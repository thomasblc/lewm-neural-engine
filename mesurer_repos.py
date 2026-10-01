"""EXP-012 : puissance du Mac au repos (aucun calcul lancé par l'expérience), mesurée avec le compteur cumulé des
capteurs de la batterie (coreml_pred.Puissance) pendant DUREE secondes. Sert de référence pour l'énergie par plan des
fichiers calculés ensuite. Une mesure par lancement : le nom porte l'heure de début, aucune mesure n'est écrasée par une
relance, et l'analyse associe à chaque fichier la dernière mesure faite avant lui.

Usage :
    python mesurer_repos.py DOSSIER TACHE [DUREE]      # écrit DOSSIER/exp012-repos-TACHE-<horodatage>.json
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import coreml_pred as C

dossier, tache, duree = Path(sys.argv[1]), sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 130.0
debut = time.time()
with C.Puissance() as p:
    time.sleep(duree)
sha = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
r = {"tache": tache, "debut": debut, "fin": time.time(), "duree_s": duree, "git": sha, **p.resume(), "lectures": p.lectures}
sortie = dossier / f"exp012-repos-{tache}-{int(debut)}.json"
sortie.write_text(json.dumps(r, indent=1))
print(f"[repos] {r['puissance_moyenne_w']} W sur {r['secondes_couvertes']} s", flush=True)
