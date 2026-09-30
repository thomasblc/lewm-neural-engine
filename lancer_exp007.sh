#!/bin/zsh
# EXP-007 (PREREG-EXP007.md) : vérifications, attente qu'aucune autre grille ne tourne, test de fonctionnement sur GPU
# (50 épisodes, all:4, replanification tous les 5 pas, bloquant), puis fp32 et all:4 × graines 42 à 44 dans deux
# conditions de planification (r5c6, r5), chacune avec sa garde de batterie, et la sanité de clôture. caffeinate empêche la mise en veille. Tout le corps est dans des fonctions, lues en entier avant d'être exécutées.
# Relancer après une interruption avec >> pour garder le journal : ./lancer_exp007.sh >> ../resultats/exp007.log 2>&1
set -e
ICI="${0:A:h}"                 # dossier du script ; dans une fonction, $0 vaut le nom de la fonction
main() {
  cd "$ICI" || exit 1
  caffeinate -i -w $$ &!         # pas de veille pendant l'attente ni entre les étapes, jusqu'à la sortie du lanceur
  local PY=/Users/thomasblanc/Documents/PRO/world-models-demos/le-wm/.venv/bin/python
  local R=../resultats
  local TIRAGE=(--chunk 50)                  # tirage par défaut : les épisodes d'EXP-003
  local GARDE=/Users/thomasblanc/Documents/PRO/these-ia-thomas/outils/garde_batterie.sh

  # 0. code figé, arbre propre ; revérifié avant chaque étape, HEAD compris
  local HEAD0=""
  verifier $PY
  HEAD0=$(git rev-parse HEAD)
  echo "[info] code vérifié : $(git rev-parse --short HEAD)"
  # une grille commencée avec un autre commit ne se reprend pas : ses fichiers seraient de deux commits
  local d commit_grille
  for d in exp007-r5c6 exp007-r5; do
    [ -f $R/$d/meta.json ] || continue
    commit_grille=$($PY -c 'import json, sys; print(json.load(open(sys.argv[1]))["git"])' $R/$d/meta.json)
    [ "$commit_grille" = "$HEAD0" ] || { echo "[arrêt] $d commencée au commit ${commit_grille:0:7}, HEAD est ${HEAD0:0:7}"; exit 1; }
  done

  # 1. le GPU est partagé : attendre 3 contrôles vides de suite (une grille, ou le lanceur d'une autre expérience
  #    entre deux de ses étapes), 24 h au plus ; 2. garde de batterie (chargeur de 70 W trop faible en plein
  #    calcul), après la sortie de celle du calcul précédent. Si un autre calcul démarre pendant ce temps, on recommence.
  local debut=$(date +%s)
  while true; do
    attendre_gpu
    garde exp007-r5c6 6
    autre_calcul || break
    echo "[info] un autre calcul a démarré pendant l'attente de la garde, nouvelle attente à $(date '+%H:%M')"
  done
  echo "[info] GPU libre, lancement à $(date '+%H:%M')"

  # 3. test de fonctionnement, bloquant : 50 épisodes sur GPU, all:4, replanification tous les 5 pas, CEM courte
  verifier $PY
  rm -rf $R/exp007-test                     # jamais les fichiers d'un ancien commit
  caffeinate -i $PY run_grid.py --out $R/exp007-test $TIRAGE --n 50 --specs all:4 --seeds 42 --replan 1 --cem-iter 6
  $PY -B -c 'import sys, numpy as np, analyse_exp003 as A
A.controler_course(dict(np.load(sys.argv[1] + "/all-4_s42.npz")), "test")
print("[test] contrôles passés sur 50 épisodes", flush=True)' $R/exp007-test

  # 3 bis. la référence r25 vient d'EXP-003 : le code actuel, réglages par défaut, doit en redonner un fichier
  #        octet pour octet (bloquant)
  verifier $PY
  rm -rf $R/exp007-sanite-r25
  caffeinate -i $PY run_grid.py --out $R/exp007-sanite-r25 $TIRAGE --n 500 --specs all:4 --seeds 42
  comparer $PY $R/exp007-sanite-r25/all-4_s42.npz $R/exp003/all-4_s42.npz "reproduction d'EXP-003"

  # 4. les deux conditions, 500 épisodes, fp32 et all:4, graines 42 à 44 ; la plus instructive d'abord
  verifier $PY
  caffeinate -i $PY run_grid.py --out $R/exp007-r5c6 $TIRAGE --n 500 --specs fp32 all:4 --seeds 42 43 44 --replan 1 --cem-iter 6
  verifier $PY
  garde exp007-r5 6
  caffeinate -i $PY run_grid.py --out $R/exp007-r5 $TIRAGE --n 500 --specs fp32 all:4 --seeds 42 43 44 --replan 1 --cem-iter 30

  # 5. sanité de clôture : all:4 graine 42 en r5c6 recalculé doit redonner le fichier de la grille
  verifier $PY
  garde exp007-sanite-fin 1
  rm -rf $R/exp007-sanite-fin               # toujours recalculée, jamais reprise
  caffeinate -i $PY run_grid.py --out $R/exp007-sanite-fin $TIRAGE --n 500 --specs all:4 --seeds 42 --replan 1 --cem-iter 6
  comparer $PY $R/exp007-sanite-fin/all-4_s42.npz $R/exp007-r5c6/all-4_s42.npz "sanité de clôture"
  echo "[info] EXP-007 finie à $(date '+%H:%M')"
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
    pgrep -f '^(/bin/)?zsh (\./|[^ ]*/)?lancer_(exp004|exp005|exp006|grille)\.sh( |$)' > /dev/null
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
  $1 -B -c 'import subprocess, analyse_exp007 as A7
A7.verifier_code(subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())'
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
