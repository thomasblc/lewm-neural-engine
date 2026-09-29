#!/bin/zsh
# EXP-006 (PREREG-EXP006.md) : vérifications, attente qu'aucune autre grille ne tourne, garde de batterie, test de
# fonctionnement sur GPU (50 épisodes, fp32 et all:4, bloquant), grille TwoRoom (9 versions × graines 42 à 46,
# réplique fp32 47 à 51), sanité de clôture. caffeinate empêche la mise en veille. Tout le corps est dans des fonctions, lues en entier avant d'être exécutées.
# Relancer après une interruption avec >> pour garder le journal : ./lancer_exp006.sh >> ../resultats/exp006.log 2>&1
set -e
ICI="${0:A:h}"                 # dossier du script ; dans une fonction, $0 vaut le nom de la fonction
main() {
  cd "$ICI" || exit 1
  caffeinate -i -w $$ &!         # pas de veille pendant l'attente ni entre les étapes, jusqu'à la sortie du lanceur
  local PY=/Users/thomasblanc/Documents/PRO/world-models-demos/le-wm/.venv/bin/python
  local R=../resultats
  local TIRAGE=(--tache tworoom --chunk 50 --graine-episodes 42)   # une position par trajectoire, comme EXP-005
  local GARDE=/Users/thomasblanc/Documents/PRO/these-ia-thomas/outils/garde_batterie.sh

  # 0. code figé, arbre propre ; revérifié avant chaque étape, HEAD compris
  local HEAD0=""
  verifier $PY
  HEAD0=$(git rev-parse HEAD)
  echo "[info] code vérifié : $(git rev-parse --short HEAD)"
  # une grille commencée avec un autre commit ne se reprend pas : ses fichiers seraient de deux commits
  if [ -f $R/exp006/meta.json ]; then
    local commit_grille=$($PY -c 'import json, sys; print(json.load(open(sys.argv[1]))["git"])' $R/exp006/meta.json)
    [ "$commit_grille" = "$HEAD0" ] || { echo "[arrêt] grille commencée au commit ${commit_grille:0:7}, HEAD est ${HEAD0:0:7}"; exit 1; }
  fi

  # 1. le GPU est partagé : attendre 3 contrôles vides de suite (une grille, ou le lanceur d'une autre expérience
  #    entre deux de ses étapes), 24 h au plus ; 2. garde de batterie (chargeur de 70 W trop faible en plein
  #    calcul), après la sortie de celle du calcul précédent. Si un autre calcul démarre pendant ce temps, on recommence.
  local debut=$(date +%s)
  while true; do
    attendre_gpu
    garde exp006 50
    autre_calcul || break
    echo "[info] un autre calcul a démarré pendant l'attente de la garde, nouvelle attente à $(date '+%H:%M')"
  done
  echo "[info] GPU libre, lancement à $(date '+%H:%M')"

  # 3. test de fonctionnement, bloquant : 50 épisodes sur GPU, fp32 et all:4, contrôlés comme dans l'analyse
  verifier $PY
  rm -rf $R/exp006-test                     # jamais les fichiers d'un ancien commit
  caffeinate -i $PY run_grid.py --out $R/exp006-test $TIRAGE --n 50 --specs fp32 all:4 --seeds 42
  $PY -B -c 'import sys, numpy as np, analyse_exp006 as A6
A6.N6 = 50
for f in ("fp32_s42", "all-4_s42"):
    A6.controler(dict(np.load(f"{sys.argv[1]}/{f}.npz")), f)
print("[test] contrôles passés sur 2 × 50 épisodes", flush=True)' $R/exp006-test

  # 4. la grille : 9 versions × graines 42 à 46, puis la réplique fp32 (graines 47 à 51), 500 épisodes
  verifier $PY
  caffeinate -i $PY run_grid.py --out $R/exp006 $TIRAGE --n 500 --specs fp32 all:8 all:6 all:4 all:3 encoder:4 \
    predictor:4 autres:4 all:4:g64 --seeds 42 43 44 45 46
  verifier $PY
  caffeinate -i $PY run_grid.py --out $R/exp006 $TIRAGE --n 500 --specs fp32 --seeds 47 48 49 50 51

  # 5. sanité de clôture : all:4 graine 42 recalculé doit redonner le fichier de la grille
  verifier $PY
  garde exp006-sanite-fin 1
  caffeinate -i $PY run_grid.py --out $R/exp006-sanite-fin $TIRAGE --n 500 --specs all:4 --seeds 42
  comparer $PY $R/exp006-sanite-fin/all-4_s42.npz $R/exp006/all-4_s42.npz "sanité de clôture"
  echo "[info] EXP-006 finie à $(date '+%H:%M')"
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

# motifs ancrés sur l'exécutable : une commande qui cite seulement ces noms (un appel de shell, un test) ne compte pas
autre_calcul() {
  pgrep -f '^[^ ]*[Pp]ython[^ ]* ([^ ]+ )*run_grid\.py( |$)' > /dev/null ||
    pgrep -f '^(/bin/)?zsh (\./|[^ ]*/)?lancer_(exp004|exp005|grille)\.sh( |$)' > /dev/null
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
    GRILLE=$1 ATTENDUS=$2 PAUSE_INITIALE=0 SEUIL_BAS=8 REPRISE_1=55 REPRISE=55 zsh $GARDE > /dev/null 2>&1 &!
    echo "[info] garde de batterie lancée pour $1"
  fi
}

verifier() {
  [ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "[arrêt] arbre git modifié"; exit 1; }
  [ -z "$HEAD0" ] || [ "$(git rev-parse HEAD)" = "$HEAD0" ] || { echo "[arrêt] HEAD a changé depuis le lancement"; exit 1; }
  $1 -B -c 'import subprocess, analyse_exp006 as A6
A6.verifier_code(subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())'
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
