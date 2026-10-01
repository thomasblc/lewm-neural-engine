"""EXP-013, essai avant lancement (amendement 3) : où Core ML place les opérations des prédicteurs codés, et ce que coûte
leur chargement en mémoire. Pour chaque processeur (Neural Engine, graphe « 1ctb » ; GPU, graphe d'origine) et chaque
codage (fp16, int8, int4b32, pal4), au lot de 300 et T = 3 :
- le détail du placement par type d'opération, « aucun » compris (constantes, décompression des poids au chargement) ;
- la taille du fichier des poids ;
- dans un processus neuf par cas : mémoire du processus (empreinte physique, outil footprint de macOS, et RSS) avant le
  chargement, après le chargement et après 10 appels.
Si la décompression des poids tombe au chargement (aucune opération de décompression placée), un codage ne peut changer
que la taille du fichier, pas la lecture des poids pendant le calcul : à dire dans le papier.

Usage :
    python essai_placement_codages.py SORTIE.json
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("STABLEWM_HOME", "/Users/thomasblanc/Documents/PRO/world-models-demos/stable-wm")


def memoire():
    pid = str(os.getpid())
    rss = int(subprocess.run(["ps", "-o", "rss=", "-p", pid], capture_output=True, text=True).stdout.strip()) * 1024
    out = subprocess.run(["footprint", "-p", pid], capture_output=True, text=True).stdout
    m = re.search(r"phys_footprint:\s*([\d.]+)\s*([KMG]B)", out) or re.search(r"Footprint:\s*([\d.]+)\s*([KMG]B)", out)
    fp = float(m.group(1)) * {"KB": 1e3, "MB": 1e6, "GB": 1e9}[m.group(2)] if m else None
    return {"rss_octets": rss, "empreinte_octets": fp}


def cas(chemin, unite):
    """Processus neuf : mémoire avant, après chargement, après 10 appels."""
    import numpy as np
    import coremltools as ct
    cu = ct.ComputeUnit.CPU_AND_NE if unite == "ne" else ct.ComputeUnit.CPU_AND_GPU
    avant = memoire()
    m = ct.models.MLModel(chemin, compute_units=cu)
    charge = memoire()
    spec = {x.name: tuple(x.type.multiArrayType.shape) for x in m.get_spec().description.input}
    x = {k: np.random.default_rng(0).standard_normal(v).astype(np.float32) for k, v in spec.items()}
    t = time.perf_counter()
    for _ in range(10):
        m.predict(x)
    print(json.dumps({"avant": avant, "apres_chargement": charge, "apres_10_appels": memoire(),
                      "ms_par_appel": 100 * (time.perf_counter() - t)}))


def main(sortie):
    import torch
    import stable_worldmodel as swm
    import coremltools as ct
    import coreml_pred as C
    model = swm.wm.utils.load_pretrained("pusht/lewm").eval()
    res = []
    with tempfile.TemporaryDirectory() as d:
        for unite, moteur in (("ne", "coreml-ne"), ("gpu", "coreml-gpu")):
            cu = ct.ComputeUnit.CPU_AND_NE if unite == "ne" else ct.ComputeUnit.CPU_AND_GPU
            for codage in C.CODAGES:
                chemin = Path(d) / f"{unite}-{codage}.mlpackage"
                m, info = C.convertir(model, moteur, chemin, codage)
                detail = C.placement_detail(m, cu)
                del m
                p = subprocess.run([sys.executable, __file__, "--cas", str(chemin), unite], capture_output=True, text=True)
                mem = json.loads(p.stdout.strip().splitlines()[-1]) if p.returncode == 0 else {"erreur": p.stderr[-500:]}
                res.append({"unite": unite, "codage": codage, "placement": info["placement"], "placement_detail": detail,
                            "octets_poids": info["octets_poids"], "memoire": mem})
                aucun = {k: v["aucun"] for k, v in detail.items() if "aucun" in v}
                print(f"{unite} {codage:8s} : placement {info['placement']}, sans processeur {aucun}, poids {info['octets_poids'] / 1e6:.1f} Mo, "
                      f"mémoire {mem.get('apres_10_appels', {})}", flush=True)
    sha = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    Path(sortie).write_text(json.dumps({"git": sha, "torch": torch.__version__, "coremltools": ct.__version__, "cas": res}, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "--cas":
        cas(sys.argv[2], sys.argv[3])
    else:
        main(sys.argv[1])
