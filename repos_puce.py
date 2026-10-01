"""EXP-013 : énergie par composant de la puce au repos (aucun calcul de l'expérience), compteurs IOReport
(energie_puce.py), sur DUREE secondes, après une attente de STABILISATION secondes (la puce retombe au repos après un
calcul). Écrit un fichier JSON nommé par son heure de début, jamais écrasé, avec les calculs run_grid vus pendant la mesure
(il ne doit y en avoir aucun).

Usage :
    python repos_puce.py DOSSIER ETIQUETTE [DUREE] [STABILISATION]
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import energie_puce

dossier, etiquette = Path(sys.argv[1]), sys.argv[2]
duree = float(sys.argv[3]) if len(sys.argv) > 3 else 90.0
stab = float(sys.argv[4]) if len(sys.argv) > 4 else 60.0
time.sleep(stab)
motif = "^[^ ]*[Pp]ython[^ ]* ([^ ]+ )*run_grid\\.py( |$)"
calculs = len(subprocess.run(["pgrep", "-f", motif], capture_output=True, text=True).stdout.split())
c = energie_puce.Compteur()
e0 = c.lire()
time.sleep(duree)
e1 = c.lire()
calculs = max(calculs, len(subprocess.run(["pgrep", "-f", motif], capture_output=True, text=True).stdout.split()))
j, dt = c.ecart(e0, e1)
sha = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
r = {"etiquette": etiquette, "debut": e0[0], "fin": e1[0], "duree_s": dt, "joules": j, "calculs_run_grid_vus": calculs, "git": sha}
f = dossier / f"exp013-repos-{etiquette}-{int(e0[0])}.json"
f.write_text(json.dumps(r, indent=1))
w = {k: j.get(k, 0) / dt for k in ("CPU Energy", "GPU Energy", "ANE", "DRAM")}
print(f"[repos] {etiquette} : " + ", ".join(f"{k} {v:.2f} W" for k, v in w.items()) + f" ; calculs vus {calculs}", flush=True)
