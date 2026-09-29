#!/bin/zsh
# EXP-005 (PREREG-EXP005.md) : vérifications, attente qu'aucune autre grille ne tourne, garde de batterie, grille de
# 2 000 nouveaux épisodes (fp32, all:6 et le contrôle all:8, graines 52 à 56), sanité de clôture. caffeinate empêche
# la mise en veille. Tout le corps est dans des fonctions, lues en entier avant d'être exécutées.
# Relancer après une interruption avec >> pour garder le journal : ./lancer_exp005.sh >> ../resultats/exp005.log 2>&1
set -e
ICI="${0:A:h}"                 # dossier du script ; dans une fonction, $0 vaut le nom de la fonction
main() {
  cd "$ICI" || exit 1
  caffeinate -i -w $$ &!         # pas de veille pendant l'attente ni entre les étapes, jusqu'à la sortie du lanceur
  local PY=/Users/thomasblanc/Documents/PRO/world-models-demos/le-wm/.venv/bin/python
  local R=../resultats
  local TIRAGE=(--n 2000 --chunk 50 --graine-episodes 2026 --exclure $R/exp003/meta.json $R/sanite-n50/meta.json)
  local GARDE=/Users/thomasblanc/Documents/PRO/these-ia-thomas/outils/garde_batterie.sh

  # 0. code figé, arbre propre ; revérifié avant chaque étape, HEAD compris
  local HEAD0=""
  verifier $PY
  HEAD0=$(git rev-parse HEAD)
  echo "[info] code vérifié : $(git rev-parse --short HEAD)"
  # une grille commencée avec un autre commit ne se reprend pas : ses fichiers seraient de deux commits
  if [ -f $R/exp005/meta.json ]; then
    local commit_grille=$($PY -c 'import json, sys; print(json.load(open(sys.argv[1]))["git"])' $R/exp005/meta.json)
    [ "$commit_grille" = "$HEAD0" ] || { echo "[arrêt] grille commencée au commit ${commit_grille:0:7}, HEAD est ${HEAD0:0:7}"; exit 1; }
  fi

  # 1. le GPU est partagé : attendre 3 contrôles vides de suite (une grille, ou le lanceur d'une autre expérience
  #    entre deux de ses étapes), 24 h au plus
  local debut=$(date +%s) vides=0
  while [ $vides -lt 3 ]; do
    if autre_calcul; then
      vides=0
    else
      vides=$((vides + 1))
    fi
    [ $(( $(date +%s) - debut )) -lt 86400 ] || { echo "[arrêt] une autre grille tourne encore après 24 h"; exit 1; }
    [ $vides -ge 3 ] || sleep 60
  done
  echo "[info] GPU libre, lancement à $(date '+%H:%M')"

  # 2. garde de batterie (chargeur de 70 W trop faible en plein calcul), si aucune ne tourne
  garde exp005 15

  # 3. la grille : 3 versions × 5 graines × 2 000 épisodes, graine par graine
  verifier $PY
  caffeinate -i $PY run_grid.py --out $R/exp005 $TIRAGE --specs fp32 all:6 all:8 --seeds 52 53 54 55 56

  # 4. sanité de clôture : all:6 graine 52 recalculé doit redonner le fichier de la grille ; la garde de la grille
  #    peut être sortie entre les deux étapes, on en relance une si besoin
  verifier $PY
  garde exp005-sanite-fin 1
  caffeinate -i $PY run_grid.py --out $R/exp005-sanite-fin $TIRAGE --specs all:6 --seeds 52
  comparer $PY $R/exp005-sanite-fin/all-6_s52.npz $R/exp005/all-6_s52.npz "sanité de clôture"
  echo "[info] EXP-005 finie à $(date '+%H:%M')"
}

# motifs ancrés sur l'exécutable : une commande qui cite seulement ces noms (un appel de shell, un test) ne compte pas
autre_calcul() {
  pgrep -f '^[^ ]*[Pp]ython[^ ]* ([^ ]+ )*run_grid\.py( |$)' > /dev/null ||
    pgrep -f '^(/bin/)?zsh (\./|[^ ]*/)?lancer_(exp004|grille)\.sh( |$)' > /dev/null
}

garde() {
  if pgrep -f '^(/bin/)?zsh [^ ]*garde_batterie\.sh$' > /dev/null; then
    echo "[info] une garde de batterie tourne déjà"
  else
    GRILLE=$1 ATTENDUS=$2 PAUSE_INITIALE=0 REPRISE_1=80 zsh $GARDE > /dev/null 2>&1 &!
    echo "[info] garde de batterie lancée pour $1"
  fi
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
