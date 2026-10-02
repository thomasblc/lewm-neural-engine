#!/bin/zsh
# EXP-015 (PREREG-EXP015.md) : où se loge la perte d'int4 par canal sur le Neural Engine, PushT. Reproduction bloquante de
# la pleine précision PyTorch (graine 42) ; runs refaits de la graine 42 pour ancrer les séries réutilisées (16 bits sur
# le Neural Engine d'EXP-012, int4c d'EXP-014, predictor:4 d'EXP-003) : une série dont le run refait diffère est refaite
# sur les graines 43 à 46 ; puis graines 42 à 46, bras alternés à chaque graine (int4c-bl, int4c-pp, int4c-sp si l'étape
# 0 b le demande, puis predproj:4 et predict:4 en simulé) ; enfin reproduction finale de la graine 46. La garde de
# batterie suit tous les calculs d'exp015 et sort quand la reproduction finale est écrite. Tout le corps est dans des
# fonctions, lues en entier avant d'être exécutées.
# Relancer après une interruption avec >> pour garder le journal : ./lancer_exp015.sh >> ../resultats/exp015.log 2>&1
set -e
ICI="${0:A:h}"                 # dossier du script ; dans une fonction, $0 vaut le nom de la fonction
main() {
  cd "$ICI" || exit 1
  caffeinate -i -w $$ &!         # pas de veille pendant l'attente ni entre les étapes, jusqu'à la sortie du lanceur
  local PY=/Users/thomasblanc/Documents/PRO/world-models-demos/le-wm/.venv/bin/python
  local R=../resultats
  local PUSHT=(--chunk 50)                                           # les épisodes d'EXP-003
  local GARDE=$ICI/garde_batterie.sh                                 # copie d'EXP-013 (ABSENTS réglable, 4 ici)
  local NE=(--moteur coreml-ne)

  # 0. code figé, arbre propre ; revérifié avant chaque étape, HEAD compris
  local HEAD0=""
  verifier $PY
  HEAD0=$(git rev-parse HEAD)
  echo "[info] code vérifié : $(git rev-parse --short HEAD)"
  # une grille commencée avec un autre commit ne se reprend pas : ses fichiers seraient de deux commits
  local d commit_grille
  for d in $R/exp015-*(/N); do
    [ -f $d/meta.json ] || continue
    case ${d:t} in exp015-repro-*|exp015-fin-*) continue ;; esac   # refaites à chaque lancement
    commit_grille=$($PY -c 'import json, sys; print(json.load(open(sys.argv[1]))["git"])' $d/meta.json)
    [ "$commit_grille" = "$HEAD0" ] || { echo "[arrêt] ${d:t} commencée au commit ${commit_grille:0:7}, HEAD est ${HEAD0:0:7}"; exit 1; }
  done
  # un seul lanceur : un second effacerait la reproduction finale du premier (pgrep s'exclut lui-même et ses ancêtres,
  # donc ce lanceur : 0 attendu)
  [ $(pgrep -f '^[^ ]*zsh (\./|[^ ]*/)?lancer_exp015\.sh( |$)' | wc -l) -eq 0 ] || { echo "[arrêt] un autre lanceur d'EXP-015 tourne"; exit 1; }
  # étape 0 b : faite au même commit, arbre propre ; décide du bras int4c-sp (règle R2)
  local BRAS=(int4c-bl int4c-pp)
  local sp
  sp=$($PY -c 'import json, sys
e = json.load(open(sys.argv[1]))
if e["git"] != sys.argv[2] or e["git_sale"]:
    sys.exit("étape 0 b d un autre commit ou d un arbre modifié")
if e.get("valide") is not True:
    sys.exit("étape 0 b invalide (plancher de validité)")
print(0 if e["sp_et_bl_identiques_sur_le_neural_engine"] else 1)' $R/exp015-etape0b/resultats.json $HEAD0) || { echo "[arrêt] étape 0 b absente ou non conforme"; exit 1; }
  # versions de l'étape 0 b et des séries réutilisées (EXP-012, EXP-014) : une mise à jour rendrait les runs inanalysables
  $PY -c 'import platform, sys, coremltools
sys.exit(0 if (platform.mac_ver()[0], coremltools.__version__) == ("26.6.2", "9.0") else 1)' || { echo "[arrêt] macOS ou coremltools différent de 26.6.2 et 9.0"; exit 1; }
  if [ "$sp" = 1 ]; then BRAS+=(int4c-sp); fi
  echo "[info] étape 0 b : bras sur le Neural Engine ${BRAS[*]}"

  # 1. le GPU est partagé : attendre 3 contrôles vides de suite, 24 h au plus ; la batterie d'abord
  rm -rf $R/exp015-fin-pusht
  local debut=$(date +%s)
  while true; do                 # batterie et GPU libre vérifiés ensemble au départ
    attendre_batterie
    attendre_gpu
    [ "$(batterie)" -lt 60 ] || break
  done
  echo "[info] GPU libre, batterie à $(batterie) %, lancement à $(date '+%H:%M')"

  verifier $PY
  # 2. garde de batterie ; elle suit tous les calculs d'exp015 et sort quand la reproduction finale est écrite
  garde exp015-fin-pusht 1
  # 3. reproduction de la pleine précision PyTorch, graine 42 (bloquante)
  rm -rf $R/exp015-repro-pusht
  caffeinate -i $PY run_grid.py --out $R/exp015-repro-pusht $PUSHT --n 500 --specs fp32 --seeds 42
  comparer $PY $R/exp015-repro-pusht/fp32_s42.npz $R/exp003/fp32_s42.npz "reproduction de exp003, graine 42"
  # 4. ancres des séries réutilisées, graine 42 ; une série dont l'ancre diffère est refaite sur 43 à 46
  local NE16=0 INT4C=0 P4=0
  caffeinate -i $PY run_grid.py --out $R/exp015-ne16-pusht $PUSHT --n 500 --specs fp32 --seeds 42 $NE
  place $PY $R/exp015-ne16-pusht/fp32_s42.temps.json
  comparer $PY $R/exp015-ne16-pusht/fp32_s42.npz $R/exp012-ne-pusht/fp32_s42.npz "ancre 16 bits contre EXP-012" non_bloquant || NE16=1
  caffeinate -i $PY run_grid.py --out $R/exp015-int4c-pusht $PUSHT --n 500 --specs fp32 --seeds 42 $NE --codage int4c
  place $PY $R/exp015-int4c-pusht/fp32_s42.temps.json
  comparer $PY $R/exp015-int4c-pusht/fp32_s42.npz $R/exp014-int4c-pusht/fp32_s42.npz "ancre int4c contre EXP-014" non_bloquant || INT4C=1
  caffeinate -i $PY run_grid.py --out $R/exp015-sim-pusht $PUSHT --n 500 --specs predictor:4 --seeds 42
  comparer $PY $R/exp015-sim-pusht/predictor-4_s42.npz $R/exp003/predictor-4_s42.npz "ancre predictor:4 contre EXP-003" non_bloquant || P4=1
  echo "[info] séries refaites dans la session : ne16 $NE16, int4c $INT4C, predictor:4 $P4"

  # 5. bras alternés graine par graine (une dérive touche tous les bras pareil)
  local g c specs
  for g in 42 43 44 45 46; do
    verifier $PY
    for c in $BRAS; do
      caffeinate -i $PY run_grid.py --out $R/exp015-$c-pusht $PUSHT --n 500 --specs fp32 --seeds $g $NE --codage $c
      place $PY $R/exp015-$c-pusht/fp32_s$g.temps.json        # tout sur le Neural Engine, sinon arrêt tout de suite
    done
    specs=(predproj:4 predict:4)
    if [ $P4 = 1 ] && [ $g != 42 ]; then specs+=(predictor:4); fi
    caffeinate -i $PY run_grid.py --out $R/exp015-sim-pusht $PUSHT --n 500 --specs $specs --seeds $g
    if [ $g != 42 ]; then
      if [ $NE16 = 1 ]; then
        caffeinate -i $PY run_grid.py --out $R/exp015-ne16-pusht $PUSHT --n 500 --specs fp32 --seeds $g $NE
        place $PY $R/exp015-ne16-pusht/fp32_s$g.temps.json
      fi
      if [ $INT4C = 1 ]; then
        caffeinate -i $PY run_grid.py --out $R/exp015-int4c-pusht $PUSHT --n 500 --specs fp32 --seeds $g $NE --codage int4c
        place $PY $R/exp015-int4c-pusht/fp32_s$g.temps.json
      fi
    fi
  done
  # 6. reproduction finale, graine 46 : rien n'a dérivé pendant le calcul
  verifier $PY
  caffeinate -i $PY run_grid.py --out $R/exp015-fin-pusht $PUSHT --n 500 --specs fp32 --seeds 46
  comparer $PY $R/exp015-fin-pusht/fp32_s46.npz $R/exp003/fp32_s46.npz "reproduction finale de exp003, graine 46"
  echo "[info] EXP-015 finie à $(date '+%H:%M')"
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
  pgrep -f '^[^ ]*[Pp]ython[^ ]* ([^ ]+ )*(run_grid|vitesse|balayage_lot|charge_vision|etape0b_exp015|valider_energie|test_codages_portee)\.py( |$)' > /dev/null ||
    pgrep -f '^(/bin/)?zsh (\./|[^ ]*/)?lancer_(exp011|exp012|exp013|exp014|grille)\.sh( |$)' > /dev/null
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
  $1 -B -c 'import subprocess, analyse_exp015 as A15
A15.verifier_code(subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())'
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
