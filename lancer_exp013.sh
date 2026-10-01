#!/bin/zsh
# EXP-013 (PREREG-EXP013.md) : vitesse, énergie et mémoire de la planification selon le moteur, le codage des poids et une
# charge de vision. Attend que le GPU soit libre (EXP-011 et EXP-012 finies), lance la garde de batterie (elle suit tous
# les calculs d'exp013 et sort quand le run final est écrit), puis vitesse.py, qui enchaîne les runs et fait les contrôles
# bloquants. caffeinate empêche la mise en veille, de la machine et de l'écran. Tout le corps est dans des fonctions, lues en entier avant d'être exécutées.
# Relancer après une interruption avec >> pour garder le journal : ./lancer_exp013.sh >> ../resultats/exp013.log 2>&1
set -e
ICI="${0:A:h}"
main() {
  cd "$ICI" || exit 1
  caffeinate -d -i -w $$ &!                 # écran allumé pendant tout le calcul, repos compris : affichage constant
  local PY=/Users/thomasblanc/Documents/PRO/world-models-demos/le-wm/.venv/bin/python
  local R=../resultats
  local GARDE=/Users/thomasblanc/Documents/PRO/these-ia-thomas/outils/garde_batterie.sh
  local HEAD0=""
  verifier $PY
  HEAD0=$(git rev-parse HEAD)
  echo "[info] code vérifié : $(git rev-parse --short HEAD)"
  local d commit_grille
  for d in $R/exp013-*(/N); do
    [ -f $d/meta.json ] || continue
    commit_grille=$($PY -c 'import json, sys; print(json.load(open(sys.argv[1]))["git"])' $d/meta.json)
    [ "$commit_grille" = "$HEAD0" ] || { echo "[arrêt] ${d:t} commencée au commit ${commit_grille:0:7}, HEAD est ${HEAD0:0:7}"; exit 1; }
  done
  rm -rf $R/exp013-fin                       # le run final d'un lancement précédent ferait sortir la garde tout de suite
  local debut=$(date +%s)
  attendre_gpu
  attendre_batterie
  if pgrep -f '(charge_vision|regulateur_batterie)\.py' > /dev/null; then
    echo "[arrêt] une charge de vision ou le régulateur de batterie tourne : il fausserait les temps"; exit 1
  fi
  echo "[info] GPU libre, batterie à $(batterie) %, lancement à $(date '+%H:%M')"
  garde exp013-fin 1
  verifier $PY
  $PY -B vitesse.py --racine $R
  echo "[info] EXP-013 finie à $(date '+%H:%M')"
}

attendre_gpu() {
  local vides=0
  while [ $vides -lt 3 ]; do
    if autre_calcul; then
      vides=0
    else
      vides=$((vides + 1))
    fi
    [ $(( $(date +%s) - debut )) -lt 86400 ] || { echo "[arrêt] une autre grille tourne encore après 24 h"; exit 1; }
    [ $vides -ge 3 ] || sleep 60
  done
}

batterie() {
  pmset -g batt | grep -o '[0-9]*%' | head -1 | tr -d '%'
}

# au moins 60 % de batterie au départ (le seuil de reprise de la garde) : une pause de la garde au milieu de l'expérience
# changerait l'état de la machine (une heure de repos et de recharge) ; 4 h d'attente au plus
attendre_batterie() {
  local t0=$(date +%s)
  while [ "$(batterie)" -lt 60 ]; do
    [ $(( $(date +%s) - t0 )) -lt 14400 ] || { echo "[arrêt] batterie sous 60 % après 4 h d'attente"; exit 1; }
    sleep 60
  done
}

# EXP-013 passe après EXP-011 et EXP-012 : leurs lanceurs sont attendus ici, et eux n'attendent pas celui-ci ;
# un calcul run_grid d'EXP-011 en cours reste vu par le premier motif.
# motifs ancrés sur l'exécutable : une commande qui cite seulement ces noms (un appel de shell, un test) ne compte pas
autre_calcul() {
  pgrep -f '^[^ ]*[Pp]ython[^ ]* ([^ ]+ )*run_grid\.py( |$)' > /dev/null ||
    pgrep -f '^(/bin/)?zsh (\./|[^ ]*/)?lancer_(exp004|exp005|exp006|exp007|exp008|exp009|exp010|exp011|exp012|grille)\.sh( |$)' > /dev/null
}

garde() {
  # une garde encore active (celle d'une autre grille, ou de l'étape précédente) sort au plus 4 min après la fin de
  # son calcul : l'attendre 6 min au plus, pour que la nouvelle étape ait sa propre garde
  local essais=0
  while pgrep -f '^(/bin/)?zsh [^ ]*garde_batterie\.sh$' > /dev/null && [ $essais -lt 6 ]; do
    essais=$((essais + 1))
    sleep 60
  done
  if pgrep -f '^(/bin/)?zsh [^ ]*garde_batterie\.sh$' > /dev/null; then
    echo "[info] une garde de batterie tourne déjà"
  else
    GRILLE=$1 ATTENDUS=$2 PAUSE_INITIALE=0 SEUIL_BAS=15 REPRISE_1=60 REPRISE=60 zsh $GARDE > /dev/null 2>&1 &!
    echo "[info] garde de batterie lancée pour $1"
  fi
}

verifier() {
  [ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "[arrêt] arbre git modifié"; exit 1; }
  [ -z "$HEAD0" ] || [ "$(git rev-parse HEAD)" = "$HEAD0" ] || { echo "[arrêt] HEAD a changé depuis le lancement"; exit 1; }
  $1 -B -c 'import subprocess, analyse_exp013 as A13
A13.verifier_code(subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())'
}

comparer() {
  $1 -B - "$2" "$3" "$4" <<'EOF'
import sys
import numpy as np
a, b, nom = np.load(sys.argv[1]), np.load(sys.argv[2]), sys.argv[3]
cles = sorted(set(a.files) - {"git", "git_sale"})
ko = ([] if set(a.files) == set(b.files) else ["clés"]) + [k for k in cles if not np.array_equal(a[k], b[k], equal_nan=a[k].dtype.kind == "f")]
print(f"[{nom}]", "identique" if not ko else f"DIFFÉRENT sur {ko} : arrêt", flush=True)
sys.exit(1 if ko else 0)
EOF
}

main "$@"
exit $?
