#!/bin/zsh
# EXP-003 (amendement 1) : grille de nuit. caffeinate empêche le Mac de dormir pendant le calcul.
cd "$(dirname "$0")"
PY=/Users/thomasblanc/Documents/PRO/world-models-demos/le-wm/.venv/bin/python
SPECS=(fp32 all:4 encoder:4 predictor:4 all:3 encoder:3 predictor:3 all:5 all:6 all:8)
caffeinate -i $PY run_grid.py --out ../resultats/exp003 --n 500 --chunk 50 --specs $SPECS --seeds 42 43 44 45 46
# réplique de la pleine précision pour les distributions nulles (amendement 1, points 6 et 7)
caffeinate -i $PY run_grid.py --out ../resultats/exp003 --n 500 --chunk 50 --specs fp32 --seeds 47 48 49 50 51
