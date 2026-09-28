#!/bin/zsh
# EXP-003 : grille de nuit. caffeinate empêche le Mac de dormir pendant le calcul.
cd "$(dirname "$0")"
PY=/Users/thomasblanc/Documents/PRO/world-models-demos/le-wm/.venv/bin/python
SPECS=(fp32 all:4 encoder:4 predictor:4 all:3 encoder:3 predictor:3 all:5 encoder:5 predictor:5 all:6 encoder:6 predictor:6 all:8 encoder:8 predictor:8)
exec caffeinate -i $PY run_grid.py --out ../resultats/exp003 --n 500 --chunk 50 --specs $SPECS --seeds 42 43 44 45 46
