#!/bin/zsh
# EXP-004 (PREREG-EXP004.md) : sanité bloquante, puis la grille. caffeinate empêche la mise en veille.
# À lancer seulement quand la grille d'EXP-003 est finie (le GPU est partagé).
cd "$(dirname "$0")"
PY=/Users/thomasblanc/Documents/PRO/world-models-demos/le-wm/.venv/bin/python
set -e
[ "$(ls ../resultats/exp003/*.npz | grep -vc tmp)" -eq 55 ] || { echo "EXP-003 n'est pas finie"; exit 1; }
# 1. sanité : all:4 graine 42 recalculé avec ce code doit redonner le fichier d'EXP-003, octet pour octet
caffeinate -i $PY run_grid.py --out ../resultats/exp004-sanite --n 500 --chunk 50 --specs all:4 --seeds 42
$PY - <<'EOF'
import sys
import numpy as np
a = np.load("../resultats/exp004-sanite/all-4_s42.npz"); b = np.load("../resultats/exp003/all-4_s42.npz")
cles = ("state", "block_pose", "agent_pos", "contacts", "success", "first_success", "n_steps")
ko = [k for k in cles if not np.array_equal(a[k], b[k], equal_nan=a[k].dtype.kind == "f")]
print("[sanité]", "identique à EXP-003" if not ko else f"DIFFÉRENT sur {ko} : grille non lancée", flush=True)
sys.exit(1 if ko else 0)
EOF
# 2. la grille : 5 versions × 5 graines × 500 épisodes, graine par graine
SPECS=(autres:4 encpred:4 all:4:g64 autres:3 all:3:g64)
caffeinate -i $PY run_grid.py --out ../resultats/exp004 --n 500 --chunk 50 --specs $SPECS --seeds 42 43 44 45 46
