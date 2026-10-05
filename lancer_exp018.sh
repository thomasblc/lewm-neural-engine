#!/bin/zsh
# EXP-018 (PREREG-EXP018.md) : int4c, tout sauf pred_proj (SP) et les codes de GPTQ (G), prédicteur sur le Neural Engine,
# sur les 2 000 épisodes d'EXP-005 (graines 52 à 56). Reproduction bloquante de la pleine précision d'EXP-005 (graine 52),
# puis graine par graine les trois bras ; enfin reproduction finale d'EXP-003 (graine 46).
# Relancer après une interruption avec >> pour garder le journal : ./lancer_exp018.sh >> ../resultats/exp018.log 2>&1
set -e
ICI="${0:A:h}"
main() {
  cd "$ICI" || exit 1
  caffeinate -i -w $$ &!         # pas de veille pendant l'attente ni entre les étapes, jusqu'à la sortie du lanceur
  local PY=/Users/thomasblanc/Documents/PRO/world-models-demos/le-wm/.venv/bin/python
  local R=../resultats
  local TIRAGE=(--chunk 50 --graine-episodes 2026 --exclure $R/exp003/meta.json $R/sanite-n50/meta.json --n 2000)   # les épisodes d'EXP-005
  local GARDE=$ICI/garde_batterie.sh
  local NE=(--moteur coreml-ne)
  local CALIB=(--gptq-calib $R/exp009-calib-pusht/calib.pt)          # la calibration d'EXP-009, celle d'EXP-010

  # 0. code figé, arbre propre ; revérifié avant chaque étape, HEAD compris
  local HEAD0=""
  verifier $PY
  HEAD0=$(git rev-parse HEAD)
  echo "[info] code vérifié : $(git rev-parse --short HEAD)"
  local d commit_grille
  for d in $R/exp018-*(/N); do
    [ -f $d/meta.json ] || continue
    case ${d:t} in exp018-repro-*|exp018-fin-*) continue ;; esac
    commit_grille=$($PY -c 'import json, sys; print(json.load(open(sys.argv[1]))["git"])' $d/meta.json)
    [ "$commit_grille" = "$HEAD0" ] || { echo "[arrêt] ${d:t} commencée au commit ${commit_grille:0:7}, HEAD est ${HEAD0:0:7}"; exit 1; }
  done
  mkdir $R/.lanceur-exp018 2>/dev/null || { echo "[arrêt] un autre lanceur d'EXP-018 tourne (verrou $R/.lanceur-exp018)"; exit 1; }
  trap 'rmdir $R/.lanceur-exp018 2>/dev/null' EXIT
  $PY -c 'import platform, sys, coremltools
sys.exit(0 if (platform.mac_ver()[0], coremltools.__version__) == ("26.6.2", "9.0") else 1)' || { echo "[arrêt] macOS ou coremltools différent de 26.6.2 et 9.0"; exit 1; }

  # 1. GPU libre et batterie au départ
  rm -rf $R/exp018-fin-pusht
  local debut=$(date +%s)
  while true; do
    attendre_batterie
    attendre_gpu
    [ "$(batterie)" -lt 60 ] || break
  done
  echo "[info] GPU libre, batterie à $(batterie) %, lancement à $(date '+%H:%M')"

  verifier $PY
  garde exp018-fin-pusht 1
  # 2. reproduction de la pleine précision d'EXP-005, graine 52, 2 000 épisodes (bloquante : appariement)
  rm -rf $R/exp018-repro-pusht
  caffeinate -i $PY run_grid.py --out $R/exp018-repro-pusht $TIRAGE --specs fp32 --seeds 52
  comparer $PY $R/exp018-repro-pusht/fp32_s52.npz $R/exp005/fp32_s52.npz "reproduction d'EXP-005, graine 52"
  # 3. graine par graine : int4c, SP, G ; un fichier déjà écrit par ce commit n'est pas refait
  local g
  for g in 52 53 54 55 56; do
    verifier $PY
    lancer $PY $R/exp018-int4c-pusht fp32 $g --codage int4c
    verifier $PY
    lancer $PY $R/exp018-int4c-sp-pusht fp32 $g --codage int4c-sp
    verifier $PY
    lancer $PY $R/exp018-gptq-pusht predict:4:gptq $g --codage int4c-gptq $CALIB
  done
  # 4. reproduction finale d'EXP-003, graine 46 (500 épisodes)
  verifier $PY
  caffeinate -i $PY run_grid.py --out $R/exp018-fin-pusht --chunk 50 --n 500 --specs fp32 --seeds 46
  comparer $PY $R/exp018-fin-pusht/fp32_s46.npz $R/exp003/fp32_s46.npz "reproduction finale de exp003, graine 46"
  echo "[info] EXP-018 finie à $(date '+%H:%M')"
}

# un bras sur le Neural Engine, une graine ; placement contrôlé après le run
lancer() {
  local py=$1 out=$2 spec=$3 g=$4
  shift 4
  local f=$out/${spec//:/-}_s$g
  if [ -f $f.npz ] && [ -f $f.temps.json ]; then
    echo "[info] ${out:t} graine $g déjà faite, gardée"
  else
    caffeinate -i $py run_grid.py --out $out $TIRAGE --specs $spec --seeds $g $NE "$@"
  fi
  place $py $f.temps.json
}

batterie() {
  pmset -g batt | grep -o '[0-9]*%' | head -1 | tr -d '%'
}

# au moins 60 % de batterie au départ (le seuil de reprise de la garde), 4 h d'attente au plus
attendre_batterie() {
  local t0=$(date +%s)
  while [ "$(batterie)" -lt 60 ]; do
    [ $(( $(date +%s) - t0 )) -lt 14400 ] || { echo "[arrêt] batterie sous 60 % après 4 h d'attente"; exit 1; }
    sleep 60
  done
}

place() {
  $1 -c 'import json, sys
p = json.load(open(sys.argv[1]))["coreml"].get("placement", {})
print("[placement]", sys.argv[1].split("/")[-2], p, flush=True)
sys.exit(0 if set(p) == {"NeuralEngine"} else 1)' "$2" || { echo "[arrêt] opérations hors du Neural Engine : $2"; exit 1; }
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
  pgrep -f '^[^ ]*[Pp]ython[^ ]* ([^ ]+ )*(run_grid|vitesse|balayage_lot|charge_vision|etape0b_exp015|etape0_exp017|valider_energie|test_codages_portee)\.py( |$)' > /dev/null ||
    pgrep -f '^(/bin/)?zsh (\./|[^ ]*/)?lancer_(exp011|exp012|exp013|exp014|exp015|exp017|grille)\.sh( |$)' > /dev/null
}

garde() {
  # sa propre garde : elle ne suit que les calculs dont le dossier de sortie a le préfixe exp018
  GRILLE=$1 ATTENDUS=$2 PAUSE_INITIALE=0 SEUIL_BAS=15 REPRISE_1=60 REPRISE=60 zsh $GARDE > /dev/null 2>&1 &!
  echo "[info] garde de batterie lancée pour $1"
}

verifier() {
  [ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "[arrêt] arbre git modifié"; exit 1; }
  [ -z "$HEAD0" ] || [ "$(git rev-parse HEAD)" = "$HEAD0" ] || { echo "[arrêt] HEAD a changé depuis le lancement"; exit 1; }
  $1 -B -c 'import subprocess, analyse_exp018 as A18
A18.verifier_code(subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())'
}

comparer() {
  # un fichier manquant arrête le lanceur (il ne compte pas comme « différent »)
  [ -f "$2" ] && [ -f "$3" ] || { echo "[arrêt] fichier à comparer manquant : $2 ou $3"; exit 1; }
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
