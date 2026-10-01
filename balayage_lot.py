"""EXP-013 : appels isolés du prédicteur de LeWM selon la taille du lot (1, 8, 32, 100, 300 candidats, T = 3), le codage
des poids Core ML (fp16, int8, int4b32, pal4) et le processeur (Neural Engine, graphe réécrit « 1ctb » ; GPU, graphe
d'origine). Pour chaque cas : conversion, 10 appels de mise en route, puis temps de 50 appels (médiane et quartiles) et
énergie de la puce par composant sur ces 50 appels ; placement des opérations. Teste la prédiction pré-enregistrée :
à 300 candidats le calcul limite (environ 900 opérations par octet de poids), les codages des poids n'accélèrent pas ;
aux petits lots, la lecture des poids peut compter.

Usage :
    python balayage_lot.py SORTIE.json
"""
import json
import os
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
import energie_puce
import predicteur_ane as P

LOTS, T = (1, 8, 32, 100, 300), 3
sortie = Path(sys.argv[1])
torch.manual_seed(0)
model = swm.wm.utils.load_pretrained("pusht/lewm").eval()
D = model.predictor.pos_embedding.shape[-1]
compteur = energie_puce.Compteur()
res = []
with tempfile.TemporaryDirectory() as d:
    for unite, cu in (("ne", ct.ComputeUnit.CPU_AND_NE), ("gpu", ct.ComputeUnit.CPU_AND_GPU)):
        for codage in C.CODAGES:
            for B in LOTS:
                g = torch.Generator().manual_seed(1)
                emb, act = torch.randn(B, T, D, generator=g), torch.randn(B, T, D, generator=g)
                if unite == "ne":
                    module = P.CoeurANE(model, "1ctb")
                    ex = {"emb": P.vers_disposition(emb, "1ctb").contiguous(), "act": P.vers_disposition(act, "1ctb").contiguous()}
                else:
                    module = C._Origine(model).eval()
                    ex = {"emb": emb, "act": act}
                m, info = C._module_coreml(module, ex, Path(d) / f"{unite}-{codage}-{B}.mlpackage", cu, codage)
                x = {k: v.numpy() for k, v in ex.items()}
                for _ in range(10):
                    m.predict(x)
                e0, t = compteur.lire(), []
                for _ in range(50):
                    a = time.perf_counter()
                    m.predict(x)
                    t.append(time.perf_counter() - a)
                j, dt = compteur.ecart(e0, compteur.lire())
                t = np.array(t) * 1000
                ligne = {"unite": unite, "codage": codage, "lot": B, "mediane_ms": float(np.median(t)),
                         "q1_ms": float(np.percentile(t, 25)), "q3_ms": float(np.percentile(t, 75)),
                         "joules_50_appels": j, "duree_s": dt, "placement": info["placement"], "octets_poids": info["octets_poids"]}
                res.append(ligne)
                print(f"{unite} {codage:8s} lot {B:3d} : {ligne['mediane_ms']:7.2f} ms, placement {info['placement']}", flush=True)
                sortie.write_text(json.dumps(res, indent=1))
