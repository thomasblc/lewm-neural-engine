"""EXP-013 : appels isolés du prédicteur de LeWM selon la taille du lot (1, 8, 32, 100, 300 candidats, T = 3), le codage
des poids Core ML (fp16, int8, int4b32, pal4, int4c) et le processeur (Neural Engine, graphe réécrit « 1ctb » ; GPU,
graphe d'origine) : 50 cas. Conversion des 50 cas d'abord (placement, détail du placement par type d'opération, taille
des poids) ; puis 30 s de stabilisation et 30 s de puissance au repos de la puce ; puis 3 répétitions, chacune dans un
ordre tiré au hasard (default_rng(17)) : pour chaque cas, chargement du modèle, 10 appels de mise en route, puis des
appels en boucle pendant au moins 3 s (temps de chaque appel ; énergie de la puce sur la boucle, moins le repos), puis
déchargement ; enfin 30 s de puissance au repos ; le repos retenu est la moyenne d'avant et d'après. Appels isolés,
caches chauds, même entrée. État de la batterie au début et à la fin.
Teste la prédiction pré-enregistrée (amendement 3) : au lot de 300, aucun codage n'est plus rapide que fp16 de plus de
5 %, sur l'un ou l'autre processeur. Fichier écrit en entier à la fin seulement (jamais de fichier partiel).

Usage :
    python balayage_lot.py SORTIE.json [--essai]
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("STABLEWM_HOME", "/Users/thomasblanc/Documents/PRO/world-models-demos/stable-wm")
import numpy as np
import torch
import coremltools as ct
import stable_worldmodel as swm

import coreml_pred as C
from coreml_pred import Puissance
import energie_puce
import predicteur_ane as P

ESSAI = "--essai" in sys.argv
LOTS, T = ((1, 300) if ESSAI else (1, 8, 32, 100, 300)), 3
CODAGES = ("fp16", "int8") if ESSAI else C.CODAGES
REPETITIONS, BOUCLE_S, REPOS_S = (1, 0.3, 3.0) if ESSAI else (3, 3.0, 30.0)
sortie = Path(sys.argv[1])
ICI = Path(__file__).resolve().parent
sha = subprocess.run(["git", "-C", str(ICI), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
torch.manual_seed(0)
model = swm.wm.utils.load_pretrained("pusht/lewm").eval()
D = model.predictor.pos_embedding.shape[-1]
compteur = energie_puce.Compteur()
UNITES = {"ne": ct.ComputeUnit.CPU_AND_NE, "gpu": ct.ComputeUnit.CPU_AND_GPU}
with tempfile.TemporaryDirectory() as d:
    cas = []
    for unite, cu in UNITES.items():
        for codage in CODAGES:
            for B in LOTS:
                g = torch.Generator().manual_seed(1)
                emb, act = torch.randn(B, T, D, generator=g), torch.randn(B, T, D, generator=g)
                if unite == "ne":
                    module = P.CoeurANE(model, "1ctb")
                    ex = {"emb": P.vers_disposition(emb, "1ctb").contiguous(), "act": P.vers_disposition(act, "1ctb").contiguous()}
                else:
                    module = C._Origine(model).eval()
                    ex = {"emb": emb, "act": act}
                chemin = Path(d) / f"{unite}-{codage}-{B}.mlpackage"
                m, info = C._module_coreml(module, ex, chemin, cu, codage)
                x = {k: v.numpy() for k, v in ex.items()}
                m.predict(x)                           # première compilation, mise en cache par le système
                cas.append({"unite": unite, "codage": codage, "lot": B, "placement": info["placement"],
                            "placement_detail": C.placement_detail(m, cu), "octets_poids": info["octets_poids"],
                            "mesures": [], "_chemin": str(chemin), "_cu": cu, "_x": x})
                del m
                print(f"[conversion] {unite} {codage:8s} lot {B:3d} : placement {info['placement']}", flush=True)
    def mesure_repos():
        e0 = compteur.lire()
        time.sleep(REPOS_S)
        j, dt = compteur.ecart(e0, compteur.lire())
        return {k: v / dt for k, v in j.items()}
    batterie_debut = Puissance.lire()
    time.sleep(REPOS_S)                                # stabilisation après les conversions
    repos_avant = mesure_repos()
    rng = np.random.default_rng(17)
    for r in range(1, REPETITIONS + 1):
        for i in rng.permutation(len(cas)):
            c = cas[i]
            m, x = ct.models.MLModel(c["_chemin"], compute_units=c["_cu"]), c["_x"]
            for _ in range(10):
                m.predict(x)
            t, e0 = [], compteur.lire()
            debut = time.perf_counter()
            while time.perf_counter() - debut < BOUCLE_S:
                a = time.perf_counter()
                m.predict(x)
                t.append(time.perf_counter() - a)
            j, dt = compteur.ecart(e0, compteur.lire())
            del m
            t = np.array(t) * 1000
            c["mesures"].append({"repetition": r, "mediane_ms": float(np.median(t)), "q1_ms": float(np.percentile(t, 25)),
                                 "q3_ms": float(np.percentile(t, 75)), "appels": len(t), "duree_s": dt, "joules": j,
                                 "joules": j})
            print(f"[r{r}] {c['unite']} {c['codage']:8s} lot {c['lot']:3d} : {np.median(t):7.2f} ms ({len(t)} appels)", flush=True)
    repos_apres = mesure_repos()
    repos_w = {k: (repos_avant.get(k, 0) + repos_apres.get(k, 0)) / 2 for k in set(repos_avant) | set(repos_apres)}
    for c in cas:                                      # énergie nette du repos (moyenne d'avant et d'après), par appel
        for m in c["mesures"]:
            m["joules_nets_par_appel"] = {k: (v - repos_w.get(k, 0) * m["duree_s"]) / m["appels"] for k, v in m["joules"].items()}
    res = {"git": sha, "essai": ESSAI, "lots": list(LOTS), "codages": list(CODAGES), "repetitions": REPETITIONS,
           "boucle_s": BOUCLE_S, "repos_w": repos_w, "repos_avant_w": repos_avant, "repos_apres_w": repos_apres,
           "batterie_debut": batterie_debut, "batterie_fin": Puissance.lire(),
           "cas": [{k: v for k, v in c.items() if not k.startswith("_")} for c in cas]}
tmp = sortie.with_suffix(".tmp.json")
tmp.write_text(json.dumps(res, indent=1))
os.replace(tmp, sortie)
