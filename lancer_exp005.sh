#!/bin/zsh
# EXP-005 (PREREG-EXP005.md) : vérifications, attente qu'aucune autre grille ne tourne, grille de 1 000 nouveaux
# épisodes (fp32 et all:6, graines 42 à 46), sanité de clôture. caffeinate empêche la mise en veille. Tout le
# corps est dans des fonctions, lues en entier avant d'être exécutées.
set -e
ICI="${0:A:h}"                 # dossier du script ; dans une fonction, $0 vaut le nom de la fonction
main() {
  cd "$ICI" || exit 1
  local PY=/Users/thomasblanc/Documents/PRO/world-models-demos/le-wm/.venv/bin/python
  local R=../resultats
  local TIRAGE=(--n 1000 --chunk 50 --graine-episodes 2026 --exclure $R/exp003/meta.json $R/sanite-n50/meta.json)

  # 0. code figé, arbre propre ; revérifié avant chaque étape, HEAD compris
  local HEAD0=""
  verifier $PY
  HEAD0=$(git rev-parse HEAD)
  echo "[info] code vérifié : $(git rev-parse --short HEAD)"

  # 1. le GPU est partagé : attendre qu'aucune autre grille ne tourne, 24 h au plus
  local debut=$(date +%s)
  while pgrep -f '[r]un_grid\.py' > /dev/null; do
    [ $(( $(date +%s) - debut )) -lt 86400 ] || { echo "[arrêt] une autre grille tourne encore après 24 h"; exit 1; }
    sleep 60
  done
  echo "[info] GPU libre, lancement à $(date '+%H:%M')"

  # 2. la grille : 2 versions × 5 graines × 1 000 épisodes, graine par graine
  verifier $PY
  caffeinate -i $PY run_grid.py --out $R/exp005 $TIRAGE --specs fp32 all:6 --seeds 42 43 44 45 46

  # 3. sanité de clôture : all:6 graine 42 recalculé doit redonner le fichier de la grille
  verifier $PY
  caffeinate -i $PY run_grid.py --out $R/exp005-sanite-fin $TIRAGE --specs all:6 --seeds 42
  comparer $PY $R/exp005-sanite-fin/all-6_s42.npz $R/exp005/all-6_s42.npz "sanité de clôture"
  echo "[info] EXP-005 finie à $(date '+%H:%M')"
}

verifier() {
  [ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "[arrêt] arbre git modifié"; exit 1; }
  [ -z "$HEAD0" ] || [ "$(git rev-parse HEAD)" = "$HEAD0" ] || { echo "[arrêt] HEAD a changé depuis le lancement"; exit 1; }
  $1 -B -c 'import subprocess, analyse_exp005 as A5
A5.verifier_code5(subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())'
}

comparer() {
  $1 -B - "$2" "$3" "$4" <<'EOF'
import sys
import numpy as np
a, b, nom = np.load(sys.argv[1]), np.load(sys.argv[2]), sys.argv[3]
cles = ("state", "block_pose", "agent_pos", "contacts", "success", "first_success", "n_steps")
ko = [k for k in cles if not np.array_equal(a[k], b[k], equal_nan=a[k].dtype.kind == "f")]
print(f"[{nom}]", "identique" if not ko else f"DIFFÉRENT sur {ko} : arrêt", flush=True)
sys.exit(1 if ko else 0)
EOF
}

main "$@"
exit $?
