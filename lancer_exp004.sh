#!/bin/zsh
# EXP-004 (PREREG-EXP004.md) : vérifications, attente de la fin d'EXP-003, sanité d'entrée, grille, sanité
# de clôture. caffeinate empêche la mise en veille. Tout le corps est dans des fonctions, lues en entier avant
# d'être exécutées : une modification du fichier pendant le calcul ne change pas les commandes lancées.
set -e
ICI="${0:A:h}"                 # dossier du script ; dans une fonction, $0 vaut le nom de la fonction
main() {
  cd "$ICI" || exit 1
  local PY=/Users/thomasblanc/Documents/PRO/world-models-demos/le-wm/.venv/bin/python
  local R=../resultats

  # 0. le code doit être celui qui est figé, arbre propre (sinon les fichiers seraient refusés à l'analyse) ;
  #    revérifié avant chaque étape, HEAD compris
  local HEAD0=""
  verifier $PY
  HEAD0=$(git rev-parse HEAD)
  echo "[info] code vérifié : $(git rev-parse --short HEAD)"

  # 1. attendre la fin d'EXP-003 (le GPU est partagé), 24 h au plus
  local debut=$(date +%s)
  while [ "$(ls $R/exp003/*.npz 2>/dev/null | grep -vc tmp)" -lt 55 ] || pgrep -f '[r]un_grid.py --out ../resultats/exp003' > /dev/null; do
    [ $(( $(date +%s) - debut )) -lt 86400 ] || { echo "[arrêt] EXP-003 pas finie après 24 h"; exit 1; }
    sleep 60
  done
  [ "$(ls $R/exp003/*.npz | grep -vc tmp)" -eq 55 ] || { echo "[arrêt] EXP-003 n'a pas 55 fichiers"; exit 1; }
  echo "[info] EXP-003 finie, lancement à $(date '+%H:%M')"

  # 2. sanité d'entrée : all:4 graine 42 recalculé doit redonner le fichier d'EXP-003, octet pour octet
  verifier $PY
  caffeinate -i $PY run_grid.py --out $R/exp004-sanite --n 500 --chunk 50 --specs all:4 --seeds 42
  comparer $PY $R/exp004-sanite/all-4_s42.npz $R/exp003/all-4_s42.npz "sanité d'entrée"

  # 3. la grille : 5 versions × 5 graines × 500 épisodes, graine par graine
  verifier $PY
  caffeinate -i $PY run_grid.py --out $R/exp004 --n 500 --chunk 50 \
    --specs autres:4 encpred:4 all:4:g64 autres:3 all:3:g64 --seeds 42 43 44 45 46

  # 4. sanité de clôture : autres:4 graine 42 recalculé doit redonner le fichier de la grille
  verifier $PY
  caffeinate -i $PY run_grid.py --out $R/exp004-sanite-fin --n 500 --chunk 50 --specs autres:4 --seeds 42
  comparer $PY $R/exp004-sanite-fin/autres-4_s42.npz $R/exp004/autres-4_s42.npz "sanité de clôture"
  echo "[info] EXP-004 finie à $(date '+%H:%M')"
}

verifier() {
  [ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "[arrêt] arbre git modifié"; exit 1; }
  [ -z "$HEAD0" ] || [ "$(git rev-parse HEAD)" = "$HEAD0" ] || { echo "[arrêt] HEAD a changé depuis le lancement"; exit 1; }
  $1 -B -c 'import subprocess, analyse_exp004 as A4
A4.verifier_code4(subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())'
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
