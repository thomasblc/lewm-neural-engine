#!/bin/zsh
# EXP-011 (PREREG-EXP011.md) : le prédicteur à 2 bits, GPTQ contre RTN. Par tâche (PushT puis TwoRoom) : reproduction
# bloquante de la pleine précision, vérification bloquante de l'erreur de reconstruction de GPTQ à 2 bits sur le
# prédicteur (calibration d'EXP-009, réutilisée), puis 2 versions × graines 42 à 46. caffeinate empêche la mise en
# veille. Tout le corps est dans des fonctions, lues en entier avant d'être exécutées.
# Relancer après une interruption avec >> pour garder le journal : ./lancer_exp011.sh >> ../resultats/exp011.log 2>&1
set -e
ICI="${0:A:h}"                 # dossier du script ; dans une fonction, $0 vaut le nom de la fonction
main() {
  cd "$ICI" || exit 1
  caffeinate -i -w $$ &!         # pas de veille pendant l'attente ni entre les étapes, jusqu'à la sortie du lanceur
  local PY=/Users/thomasblanc/Documents/PRO/world-models-demos/le-wm/.venv/bin/python
  local R=../resultats
  local PUSHT=(--chunk 50)                                           # les épisodes d'EXP-003
  local TWOROOM=(--tache tworoom --chunk 50 --graine-episodes 42)    # les épisodes d'EXP-006
  local VERSIONS=(predictor:2 predictor:2:gptq)
  local GARDE=/Users/thomasblanc/Documents/PRO/these-ia-thomas/outils/garde_batterie.sh

  # 0. code figé, arbre propre ; revérifié avant chaque étape, HEAD compris
  local HEAD0=""
  verifier $PY
  HEAD0=$(git rev-parse HEAD)
  echo "[info] code vérifié : $(git rev-parse --short HEAD)"
  # une grille commencée avec un autre commit ne se reprend pas : ses fichiers seraient de deux commits
  local d commit_grille
  for d in exp011-pusht exp011-tworoom; do
    [ -f $R/$d/meta.json ] || continue
    commit_grille=$($PY -c 'import json, sys; print(json.load(open(sys.argv[1]))["git"])' $R/$d/meta.json)
    [ "$commit_grille" = "$HEAD0" ] || { echo "[arrêt] $d commencée au commit ${commit_grille:0:7}, HEAD est ${HEAD0:0:7}"; exit 1; }
  done

  # 1. le GPU est partagé : attendre 3 contrôles vides de suite (une grille, ou le lanceur d'une autre expérience
  #    entre deux de ses étapes), 24 h au plus ; 2. garde de batterie, après la sortie de celle du calcul précédent.
  #    Si un autre calcul démarre pendant ce temps, on recommence.
  local debut=$(date +%s)
  while true; do
    attendre_gpu
    garde exp011-pusht 10
    autre_calcul || break
    echo "[info] un autre calcul a démarré pendant l'attente de la garde, nouvelle attente à $(date '+%H:%M')"
  done
  echo "[info] GPU libre, lancement à $(date '+%H:%M')"

  # 3. PushT : reproduction de la pleine précision (bloquante, référence de la sentinelle), vérification de GPTQ, grille
  verifier $PY
  rm -rf $R/exp011-repro-pusht
  caffeinate -i $PY run_grid.py --out $R/exp011-repro-pusht $PUSHT --n 500 --specs fp32 --seeds 42
  comparer $PY $R/exp011-repro-pusht/fp32_s42.npz $R/exp003/fp32_s42.npz "reproduction d'EXP-003"
  $PY -B verifier_gptq.py --tache pusht --calib $R/exp009-calib-pusht/calib.pt --bits 2 --groupe predictor
  verifier $PY
  caffeinate -i $PY run_grid.py --out $R/exp011-pusht $PUSHT --n 500 --specs $VERSIONS --seeds 42 43 44 45 46 \
    --gptq-calib $R/exp009-calib-pusht/calib.pt

  # 4. TwoRoom : mêmes étapes
  verifier $PY
  garde exp011-tworoom 10
  rm -rf $R/exp011-repro-tworoom
  caffeinate -i $PY run_grid.py --out $R/exp011-repro-tworoom $TWOROOM --n 500 --specs fp32 --seeds 42
  comparer $PY $R/exp011-repro-tworoom/fp32_s42.npz $R/exp006/fp32_s42.npz "reproduction d'EXP-006"
  $PY -B verifier_gptq.py --tache tworoom --calib $R/exp009-calib-tworoom/calib.pt --bits 2 --groupe predictor
  verifier $PY
  caffeinate -i $PY run_grid.py --out $R/exp011-tworoom $TWOROOM --n 500 --specs $VERSIONS --seeds 42 43 44 45 46 \
    --gptq-calib $R/exp009-calib-tworoom/calib.pt
  echo "[info] EXP-011 finie à $(date '+%H:%M')"
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
    pgrep -f '^(/bin/)?zsh (\./|[^ ]*/)?lancer_(exp004|exp005|exp006|exp007|exp008|exp009|exp010|grille)\.sh( |$)' > /dev/null
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
  $1 -B -c 'import subprocess, analyse_exp011 as A11
A11.verifier_code(subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())'
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
