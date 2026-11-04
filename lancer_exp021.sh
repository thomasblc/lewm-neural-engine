#!/bin/zsh
# EXP-021 (PREREG-EXP021.md, amendement 1) : prédicteur ternaire avec 1 % gardé et réglage fin, contre le 3 bits réglé,
# sur les 1 000 premières positions des épisodes d'EXP-005 (graines 52 à 55). 0. code figé ; 1. reproduction bloquante de
# la pleine précision d'EXP-005 (100 premières positions, graine 52) ; 2. collecte des entrées du prédicteur sur 32
# épisodes de calibration ; 3. 16 réglages (4 versions x 4 graines) ; 4. grille, graine par graine, bras A, B, B0, C, D,
# E ; 5. reproduction bloquante finale (graine 55) ; 6. analyse. Une garde de batterie par étape de run_grid (pause à
# 15 %, reprise à 60 %) et sentinelle ; caffeinate empêche la veille.
# ESSAI=1 : essai court de bout en bout sur la vraie machine (50 épisodes, 50 pas de réglage, dossiers essai-exp021-*,
# tests unitaires d'abord), pour exercer tous les contrôles bloquants avant le vrai lancement.
# Lancer détaché (outils/CALCULS-DE-NUIT.md) :
#   nohup /usr/bin/python3 -c 'import os; os.setsid(); os.execvp("zsh", ["zsh", "./lancer_exp021.sh"])' \
#     >> ../resultats/exp021.log 2>&1 < /dev/null &
# Ce lanceur n'efface rien : un dossier d'un autre commit arrête le lancement, à mettre à la corbeille à la main.
set -e
ICI="${0:A:h}"
main() {
  cd "$ICI" || exit 1
  caffeinate -i -w $$ &!
  local PY=/Users/thomasblanc/Documents/PRO/world-models-demos/le-wm/.venv/bin/python
  local R=../resultats
  local CALIB=$R/exp009-calib-pusht/calib.pt
  local PRE=exp021 PAS=5000 KREPRO=100 NPOS=1000
  if [ -n "$ESSAI" ]; then
    PRE=essai-exp021 PAS=50 KREPRO=50 NPOS=50
  fi
  local GRAINES=(52 53 54 55)
  local TIRAGE=(--n 2000 --chunk 50 --graine-episodes 2026 --exclure $R/exp003/meta.json $R/sanite-n50/meta.json)
  local VERSIONS=(predictor:2:clip:gptq:k1:obs:aff predictor:3:clip:gptq:aff predictor:3:clip:gptq predictor:2:clip:gptq:k1:obs
                  predictor:2:clip:gptq:aff predictor:2:clip:gptq:k3.1:obs:aff)
  local REGLER=(predictor:2:clip:gptq:k1:obs predictor:3:clip:gptq predictor:2:clip:gptq predictor:2:clip:gptq:k3.1:obs)
  local GARDE=/Users/thomasblanc/Documents/PRO/these-ia-thomas/outils/garde_batterie.sh
  local HEAD0=""

  # 0. code figé, arbre propre ; revérifié avant chaque étape, HEAD compris
  verifier
  HEAD0=$(git rev-parse HEAD)
  echo "[info] ${PRE} : code $(git rev-parse --short HEAD), début $(date '+%H:%M')"
  local d c
  for d in $PRE-repro-debut $PRE-collecte $PRE-pusht $PRE-repro-fin; do
    [ -f $R/$d/meta.json ] || continue
    c=$($PY -c 'import json, sys; print(json.load(open(sys.argv[1]))["git"])' $R/$d/meta.json)
    [ "$c" = "$HEAD0" ] || { echo "[arrêt] $d commencé au commit ${c:0:7}, HEAD est ${HEAD0:0:7} : le mettre à la corbeille"; exit 1; }
  done
  if [ -n "$ESSAI" ]; then
    $PY -B test_affinage.py | tail -1
    $PY -B test_analyse_exp021.py | tail -1
  fi

  # le GPU est partagé : attendre 3 contrôles vides de suite ; puis sentinelle. La garde de batterie ne suit que les
  # run_grid de son dossier et sort quand il est complet : une garde par étape (relecture du 4 novembre, P0).
  local debut=$(date +%s)
  attendre_gpu
  if ! pgrep -f "[s]entinelle\.py --boucle" > /dev/null; then
    (cd ../../outils && nohup /usr/bin/python3 -c 'import os; os.setsid(); os.execvp("/Users/thomasblanc/Documents/PRO/world-models-demos/le-wm/.venv/bin/python", ["python", "sentinelle.py", "--boucle"])' \
      >> ../papier-corl-2026/resultats/sentinelle-boucle.log 2>&1 < /dev/null &)
    echo "[info] sentinelle lancée"
  fi

  # 1. reproduction de la pleine précision d'EXP-005 (bloquante)
  verifier
  garde $PRE-repro-debut 1
  [ -f $R/$PRE-repro-debut/fp32_s52.npz ] || \
    caffeinate -i $PY run_grid.py --out $R/$PRE-repro-debut $TIRAGE --premiers $KREPRO --specs fp32 --seeds 52
  comparer $R/$PRE-repro-debut/fp32_s52.npz $R/exp005/fp32_s52.npz $KREPRO

  # 2. collecte des entrées du prédicteur (épisodes de calibration d'EXP-009, trajectoires d'EXP-005 exclues)
  verifier
  attendre_batterie
  [ -f $R/$PRE-collecte/affinage-donnees.pt ] || caffeinate -i $PY run_grid.py --out $R/$PRE-collecte \
    --collecter-affinage --n 256 --chunk 32 --premiers 32 --specs fp32 --seeds 0 --graine-episodes 9009 \
    --exclure $R/exp003/meta.json $R/sanite-n50/meta.json $R/exp005/meta.json

  # 3. réglages : un par version et par graine
  local g v f
  for g in $GRAINES; do
    for v in $REGLER; do
      f=$R/$PRE-collecte/${v//:/-}_g$g.pt
      [ -f $f ] && continue
      verifier
      attendre_batterie
      caffeinate -i $PY affinage.py --spec $v --calib $CALIB --donnees $R/$PRE-collecte/affinage-donnees.pt --out $f \
        --pas $PAS --lot 256 --lr 1e-3 --graine $g --device mps
    done
  done

  # 4. grille : graine par graine, bras A, B, B0, C, D, E
  garde $PRE-pusht $(( ${#VERSIONS} * ${#GRAINES} ))
  for g in $GRAINES; do
    verifier
    caffeinate -i $PY run_grid.py --out $R/$PRE-pusht $TIRAGE --premiers $NPOS --specs $VERSIONS --seeds $g \
      --gptq-calib $CALIB --affinage-dossier $R/$PRE-collecte
  done

  # 5. reproduction finale (bloquante)
  verifier
  garde $PRE-repro-fin 1
  [ -f $R/$PRE-repro-fin/fp32_s55.npz ] || \
    caffeinate -i $PY run_grid.py --out $R/$PRE-repro-fin $TIRAGE --premiers $KREPRO --specs fp32 --seeds 55
  comparer $R/$PRE-repro-fin/fp32_s55.npz $R/exp005/fp32_s55.npz $KREPRO

  # 6. analyse pré-enregistrée
  $PY -B analyse_exp021.py --racine $R --sortie $R/$PRE-analyse ${ESSAI:+--essai}
  echo "[fin] ${PRE} à $(date '+%H:%M')"
}

verifier() {
  [ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "[arrêt] arbre git modifié"; exit 1; }
  [ -z "$HEAD0" ] || [ "$(git rev-parse HEAD)" = "$HEAD0" ] || { echo "[arrêt] HEAD a changé depuis le lancement"; exit 1; }
}

comparer() {
  $PY -B - "$1" "$2" "$3" "$HEAD0" <<'EOF'
import sys
import numpy as np
import analyse_exp003 as A
import analyse_exp021 as A21
f, ref, k, tete = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
try:
    A21.identiques_premiers(f, ref, k, f"reproduction de {ref}")
    x = np.load(f)
    A.exiger(str(x["git"]) == tete and not bool(x["git_sale"]), f"{f} : autre commit ou arbre modifié")
except A.ControleEchoue as e:
    print(f"[arrêt] {e}", flush=True)
    sys.exit(1)
print(f"[contrôle] {f} identique à {ref} sur les {k} premières positions", flush=True)
EOF
}

attendre_gpu() {
  local vides=0 tours=0
  while [ $vides -lt 3 ]; do
    if autre_calcul; then
      vides=0
      tours=$((tours + 1))
      [ $((tours % 10)) -ne 1 ] || { echo "[info] GPU occupé ($(date '+%H:%M')) :"; pgrep -fl '(run_grid|affinage)\.py|lancer_' | grep -v "^$$ " | cut -c1-120; }
    else
      vides=$((vides + 1))
    fi
    [ $(( $(date +%s) - debut )) -lt 86400 ] || { echo "[arrêt] une autre grille tourne encore après 24 h"; exit 1; }
    [ $vides -ge 3 ] || sleep 60
  done
}

# motifs ancrés sur l'exécutable : une commande qui cite seulement ces noms (un appel de shell, un test) ne compte pas
autre_calcul() {
  pgrep -f '^[^ ]*[Pp]ython[^ ]* ([^ ]+ )*(run_grid|affinage)\.py( |$)' > /dev/null ||
    pgrep -f '^(/bin/)?zsh (\./|[^ ]*/)?lancer_(exp0[0-9][0-9]|pilote_exp019|grille)\.sh( |$)' | grep -v "^$$\$" > /dev/null
}

# la garde de batterie ne gèle que run_grid.py : avant la collecte et chaque réglage, batterie à 60 % au moins
attendre_batterie() {
  local pct
  while true; do
    pct=$(pmset -g batt | grep -Eo '[0-9]+%' | head -1 | tr -d %)
    [ -n "$pct" ] && [ "$pct" -ge 60 ] && return 0
    echo "[info] batterie à ${pct:-?} %, réglage en attente ($(date '+%H:%M'))"
    sleep 300
  done
}

garde() {
  # une garde encore active sort au plus 4 min après la fin de son calcul : l'attendre 6 min au plus
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

main "$@"
