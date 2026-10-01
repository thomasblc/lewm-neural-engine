"""EXP-012 : puissance du Mac au repos (aucun calcul lancé par l'expérience), mesurée avec le compteur cumulé des
capteurs de la batterie (coreml_pred.Puissance) pendant DUREE secondes. Sert de référence pour l'énergie par plan.

Usage :
    python mesurer_repos.py SORTIE.json [DUREE]
"""
import json
import sys
import time

import coreml_pred as C

sortie, duree = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 130.0
with C.Puissance() as p:
    time.sleep(duree)
r = {"duree_s": duree, **p.resume(), "lectures": p.lectures}
open(sortie, "w").write(json.dumps(r, indent=1))
print(f"[repos] {r['puissance_moyenne_w']} W sur {r['secondes_couvertes']} s", flush=True)
