#!/bin/zsh
# EXP-017 (PREREG-EXP017.md) : les codes de GPTQ écrits dans Core ML, prédicteur sur le Neural Engine, PushT.
# Reproduction bloquante de la pleine précision (graine 42), ancres ne16 et int4c (graine 42 ; une série dont l'ancre
# diffère est refaite), puis graines 42 à 46 : G (codes de GPTQ sur la puce) et GS (simulé) ; enfin reproduction finale.
# Relancer après une interruption avec >> pour garder le journal : ./lancer_exp017.sh >> ../resultats/exp017.log 2>&1
set -e
ICI="${0:A:h}"
main() {
  cd "$ICI" || exit 1
  caffeinate -i -w $$ &!         # pas de veille pendant l'attente ni entre les étapes, jusqu'à la sortie du lanceur
  local PY=/Users/thomasblanc/Documents/PRO/world-models-demos/le-wm/.venv/bin/python
  local R=../resultats
  local PUSHT=(--chunk 50)                                           # les épisodes d'EXP-003
  local GARDE=$ICI/garde_batterie.sh
  local NE=(--moteur coreml-ne)
  local CALIB=(--gptq-calib $R/exp009-calib-pusht/calib.pt)          # la calibration d'EXP-009

  # 0. code figé, arbre propre ; revérifié avant chaque étape, HEAD compris
  local HEAD0=""
  verifier $PY
  HEAD0=$(git rev-parse HEAD)
  echo "[info] code vérifié : $(git rev-parse --short HEAD)"
  local d commit_grille
  for d in $R/exp017-*(/N); do
    [ -f $d/meta.json ] || continue
    case ${d:t} in exp017-repro-*|exp017-fin-*) continue ;; esac
    commit_grille=$($PY -c 'import json, sys; print(json.load(open(sys.argv[1]))["git"])' $d/meta.json)
    [ "$commit_grille" = "$HEAD0" ] || { echo "[arrêt] ${d:t} commencée au commit ${commit_grille:0:7}, HEAD est ${HEAD0:0:7}"; exit 1; }
  done
  [ $(pgrep -f '^[^ ]*zsh (\./|[^ ]*/)?lancer_exp017\.sh( |$)' | wc -l) -eq 0 ] || { echo "[arrêt] un autre lanceur d'EXP-017 tourne"; exit 1; }
  # étape 0 au même commit : codes de GPTQ écrits exactement, tout sur le Neural Engine
  $PY -c 'import json, sys
e = json.load(open(sys.argv[1]))
sys.exit(0 if e["git"] == sys.argv[2] and not e["git_sale"] and e["codes_differents"] == 0 and set(e["placement"]) == {"NeuralEngine"} else 1)' $R/exp017-etape0/resultats.json $HEAD0 || { echo "[arrêt] étape 0 absente, d'un autre commit ou non conforme"; exit 1; }
  $PY -c 'import platform, sys, coremltools
sys.exit(0 if (platform.mac_ver()[0], coremltools.__version__) == ("26.6.2", "9.0") else 1)' || { echo "[arrêt] macOS ou coremltools différent de 26.6.2 et 9.0"; exit 1; }

  # 1. GPU libre et batterie au départ
  rm -rf $R/exp017-fin-pusht
  local debut=$(date +%s)
  while true; do
    attendre_batterie
    attendre_gpu
    [ "$(batterie)" -lt 60 ] || break
  done
  echo "[info] GPU libre, batterie à $(batterie) %, lancement à $(date '+%H:%M')"

  verifier $PY
  garde exp017-fin-pusht 1
  # 2. reproduction de la pleine précision PyTorch, graine 42 (bloquante)
  rm -rf $R/exp017-repro-pusht
  caffeinate -i $PY run_grid.py --out $R/exp017-repro-pusht $PUSHT --n 500 --specs fp32 --seeds 42
  comparer $PY $R/exp017-repro-pusht/fp32_s42.npz $R/exp003/fp32_s42.npz "reproduction de exp003, graine 42"
  # 3. ancres ne16 et int4c, graine 42 ; une série dont l'ancre diffère est refaite sur 43 à 46
  local NE16=0 INT4C=0
  caffeinate -i $PY run_grid.py --out $R/exp017-ne16-pusht $PUSHT --n 500 --specs fp32 --seeds 42 $NE
  place $PY $R/exp017-ne16-pusht/fp32_s42.temps.json
  comparer $PY $R/exp017-ne16-pusht/fp32_s42.npz $R/exp012-ne-pusht/fp32_s42.npz "ancre 16 bits contre EXP-012" non_bloquant || NE16=1
  caffeinate -i $PY run_grid.py --out $R/exp017-int4c-pusht $PUSHT --n 500 --specs fp32 --seeds 42 $NE --codage int4c
  place $PY $R/exp017-int4c-pusht/fp32_s42.temps.json
  comparer $PY $R/exp017-int4c-pusht/fp32_s42.npz $R/exp014-int4c-pusht/fp32_s42.npz "ancre int4c contre EXP-014" non_bloquant || INT4C=1
  echo "[info] séries refaites dans la session : ne16 $NE16, int4c $INT4C"
  # 4. G (codes de GPTQ sur le Neural Engine) puis GS (simulé), graine par graine
  local g
  for g in 42 43 44 45 46; do
    verifier $PY
    caffeinate -i $PY run_grid.py --out $R/exp017-gptq-pusht $PUSHT --n 500 --specs predict:4:gptq --seeds $g $NE --codage int4c-gptq $CALIB
    place $PY $R/exp017-gptq-pusht/predict-4-gptq_s$g.temps.json
    caffeinate -i $PY run_grid.py --out $R/exp017-sim-pusht $PUSHT --n 500 --specs predict:4:gptq --seeds $g $CALIB
    if [ $g != 42 ]; then
      if [ $NE16 = 1 ]; then
        caffeinate -i $PY run_grid.py --out $R/exp017-ne16-pusht $PUSHT --n 500 --specs fp32 --seeds $g $NE
        place $PY $R/exp017-ne16-pusht/fp32_s$g.temps.json
      fi
      if [ $INT4C = 1 ]; then
        caffeinate -i $PY run_grid.py --out $R/exp017-int4c-pusht $PUSHT --n 500 --specs fp32 --seeds $g $NE --codage int4c
        place $PY $R/exp017-int4c-pusht/fp32_s$g.temps.json
      fi
    fi
  done
  # 5. reproduction finale, graine 46
  verifier $PY
  caffeinate -i $PY run_grid.py --out $R/exp017-fin-pusht $PUSHT --n 500 --specs fp32 --seeds 46
  comparer $PY $R/exp017-fin-pusht/fp32_s46.npz $R/exp003/fp32_s46.npz "reproduction finale de exp003, graine 46"
  echo "[info] EXP-017 finie à $(date '+%H:%M')"
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
    pgrep -f '^(/bin/)?zsh (\./|[^ ]*/)?lancer_(exp011|exp012|exp013|exp014|exp015|grille)\.sh( |$)' > /dev/null
}

garde() {
  # toujours sa propre garde : une garde ne suit que les calculs dont le dossier de sortie a le préfixe de sa grille
  # (exp015 ici), une garde d'une autre expérience ne gèlerait pas ceux-ci
  GRILLE=$1 ATTENDUS=$2 PAUSE_INITIALE=0 SEUIL_BAS=15 REPRISE_1=60 REPRISE=60 zsh $GARDE > /dev/null 2>&1 &!
  echo "[info] garde de batterie lancée pour $1"
}

verifier() {
  [ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "[arrêt] arbre git modifié"; exit 1; }
  [ -z "$HEAD0" ] || [ "$(git rev-parse HEAD)" = "$HEAD0" ] || { echo "[arrêt] HEAD a changé depuis le lancement"; exit 1; }
  $1 -B -c 'import subprocess, analyse_exp017 as A17
A17.verifier_code(subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())'
}

comparer() {
  # un fichier manquant arrête le lanceur (il ne compte pas comme « différent »)
  [ -f "$2" ] && [ -f "$3" ] || { echo "[arrêt] fichier à comparer manquant : $2 ou $3"; exit 1; }
  $1 -B - "$2" "$3" "$4" "$5" <<'EOF'
import sys
import numpy as np
a, b, nom = np.load(sys.argv[1]), np.load(sys.argv[2]), sys.argv[3]
cles = sorted(set(a.files) - {"git", "git_sale"})
ko = ([] if set(a.files) == set(b.files) else ["clés"]) + [k for k in cles if not np.array_equal(a[k], b[k], equal_nan=a[k].dtype.kind == "f")]
suite = "non bloquant" if sys.argv[4:5] == ["non_bloquant"] else "arrêt"
print(f"[{nom}]", "identique" if not ko else f"DIFFÉRENT sur {ko} : {suite}", flush=True)
sys.exit(1 if ko else 0)
EOF
}

main "$@"
exit $?
