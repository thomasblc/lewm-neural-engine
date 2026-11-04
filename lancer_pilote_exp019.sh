#!/bin/zsh
# EXP-019, pilote en petit format (exploratoire, PILOTE-EXP019.md, amendement du 4 novembre) : PushT, 100 premières
# positions du tirage d'EXP-003, graine 42. 1. pleine précision, comparée aux fichiers d'avant macOS 27 ; 2. ternaire
# GPTQ d'EXP-011, comparé de même ; 3. collecte des entrées du prédicteur sur 32 épisodes de calibration (tirage
# d'EXP-009) ; 4. réglage fin ; 5. 1 % gardé, 3 bits, puis 1 % gardé et réglé ; 6. analyse. caffeinate empêche la
# veille. Lancer détaché (voir outils/CALCULS-DE-NUIT.md) ; journal : ../resultats/exp019-pilote.log
set -e
ICI="${0:A:h}"
main() {
  cd "$ICI" || exit 1
  caffeinate -i -w $$ &!
  local PY=/Users/thomasblanc/Documents/PRO/world-models-demos/le-wm/.venv/bin/python
  local R=../resultats
  local CALIB=$R/exp009-calib-pusht/calib.pt
  local EVAL=(--n 500 --chunk 50 --premiers 100 --seeds 42 --gptq-calib $CALIB)
  [ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "[arrêt] arbre git modifié"; exit 1; }
  echo "[info] code $(git rev-parse --short HEAD), début $(date '+%H:%M')"

  # 1 et 2 : contrôles de dérive (informatifs : l'analyse dit si les épisodes sont identiques)
  caffeinate -i $PY run_grid.py --out $R/exp019-pilote $EVAL --specs fp32 predictor:2:clip:gptq
  $PY -B analyse_pilote_exp019.py
  # garde-fou de la pleine précision : à moins de 10 points de sa valeur d'avant sur les mêmes positions
  $PY -B - <<'EOF'
import numpy as np, sys
a = np.load("../resultats/exp019-pilote/fp32_s42.npz")["success"][:100].mean()
b = np.load("../resultats/exp003/fp32_s42.npz")["success"][:100].mean()
print(f"[contrôle] pleine précision {100 * a:.1f} % contre {100 * b:.1f} % avant macOS 27")
sys.exit(1 if abs(a - b) > 0.10 else 0)
EOF

  # 3 et 4 : données de réglage (planification en pleine précision, épisodes de calibration) et réglage fin
  [ -f $R/exp019-collecte/affinage-donnees.pt ] || caffeinate -i $PY run_grid.py --collecter-affinage \
    --out $R/exp019-collecte --n 256 --chunk 32 --premiers 32 --specs fp32 --seeds 0 --graine-episodes 9009 \
    --exclure $R/exp003/meta.json $R/sanite-n50/meta.json $R/exp005/meta.json
  [ -f $R/exp019-collecte/affinage.pt ] || caffeinate -i $PY affinage.py --spec predictor:2:clip:gptq:k1:obs \
    --calib $CALIB --donnees $R/exp019-collecte/affinage-donnees.pt --out $R/exp019-collecte/affinage.pt \
    --pas 5000 --lot 256 --lr 1e-3 --graine 0 --device mps

  # 5 : versions restantes
  caffeinate -i $PY run_grid.py --out $R/exp019-pilote $EVAL --specs predictor:2:clip:gptq:k1:obs predictor:3
  caffeinate -i $PY run_grid.py --out $R/exp019-pilote-aff $EVAL --specs predictor:2:clip:gptq:k1:obs:aff \
    --affinage $R/exp019-collecte/affinage.pt

  # 6 : analyse
  $PY -B analyse_pilote_exp019.py
  echo "[fin] $(date '+%H:%M')"
}
main "$@"
