#!/bin/zsh
# EXP-014 (PREREG-EXP014.md) : le prédicteur en Core ML sur le Neural Engine, poids codés int8, int4 par canal, palettes.
# Par tâche (PushT puis TwoRoom) : reproduction bloquante de la pleine précision PyTorch (graine 42), puis graines 42 à
# 46 avec les trois codages alternés à chaque graine, enfin reproduction finale de la graine 46. La garde de batterie
# suit tous les calculs d'exp014 et sort quand la reproduction finale de la tâche est écrite. caffeinate empêche la mise
# en veille. Tout le corps est dans des fonctions, lues en entier avant d'être exécutées.
# Relancer après une interruption avec >> pour garder le journal : ./lancer_exp014.sh >> ../resultats/exp014.log 2>&1
set -e
ICI="${0:A:h}"                 # dossier du script ; dans une fonction, $0 vaut le nom de la fonction
main() {
  cd "$ICI" || exit 1
  caffeinate -i -w $$ &!         # pas de veille pendant l'attente ni entre les étapes, jusqu'à la sortie du lanceur
  local PY=/Users/thomasblanc/Documents/PRO/world-models-demos/le-wm/.venv/bin/python
  local R=../resultats
  local PUSHT=(--chunk 50)                                           # les épisodes d'EXP-003
  local TWOROOM=(--tache tworoom --chunk 50 --graine-episodes 42)    # les épisodes d'EXP-006
  local GARDE=$ICI/garde_batterie.sh                                 # copie d'EXP-013 (ABSENTS réglable, 4 ici)
  local CODAGES=(int8 int4c pal4)

  # 0. code figé, arbre propre ; revérifié avant chaque étape, HEAD compris
  local HEAD0=""
  verifier $PY
  HEAD0=$(git rev-parse HEAD)
  echo "[info] code vérifié : $(git rev-parse --short HEAD)"
  # une grille commencée avec un autre commit ne se reprend pas : ses fichiers seraient de deux commits
  local d commit_grille c
  for d in $R/exp014-*(/N); do
    [ -f $d/meta.json ] || continue
    case ${d:t} in exp014-repro-*|exp014-fin-*) continue ;; esac   # refaites à chaque lancement
    commit_grille=$($PY -c 'import json, sys; print(json.load(open(sys.argv[1]))["git"])' $d/meta.json)
    [ "$commit_grille" = "$HEAD0" ] || { echo "[arrêt] ${d:t} commencée au commit ${commit_grille:0:7}, HEAD est ${HEAD0:0:7}"; exit 1; }
  done

  # un seul lanceur : un second effacerait les reproductions finales du premier (pgrep s'exclut lui-même et ses ancêtres,
  # donc ce lanceur : 0 attendu)
  [ $(pgrep -f '^[^ ]*zsh (\./|[^ ]*/)?lancer_exp014\.sh( |$)' | wc -l) -eq 0 ] || { echo "[arrêt] un autre lanceur d'EXP-014 tourne"; exit 1; }
  # 1. le GPU est partagé : attendre 3 contrôles vides de suite, 24 h au plus
  # une reproduction finale d'un lancement précédent ferait sortir la garde tout de suite : elle est refaite
  rm -rf $R/exp014-fin-pusht $R/exp014-fin-tworoom
  local debut=$(date +%s)
  attendre_batterie                # la batterie d'abord : le GPU est vérifié libre juste avant le départ
  attendre_gpu
  echo "[info] GPU libre, batterie à $(batterie) %, lancement à $(date '+%H:%M')"

  local t opts ref g
  for t in pusht tworoom; do
    if [ $t = pusht ]; then opts=($PUSHT); ref=exp003; else opts=($TWOROOM); ref=exp006; fi
    verifier $PY
    # 2. garde de batterie de la tâche ; elle suit tous les calculs d'exp014 et sort quand la reproduction finale est écrite
    garde exp014-fin-$t 1
    # 3. reproduction de la pleine précision PyTorch, graine 42 (bloquante, référence de la sentinelle)
    rm -rf $R/exp014-repro-$t
    caffeinate -i $PY run_grid.py --out $R/exp014-repro-$t $opts --n 500 --specs fp32 --seeds 42
    comparer $PY $R/exp014-repro-$t/fp32_s42.npz $R/$ref/fp32_s42.npz "reproduction de $ref, graine 42"
    # 3 bis. témoin : le 16 bits sur le Neural Engine refait (graine 42), comparé à EXP-012 ; non bloquant (amendement 1)
    caffeinate -i $PY run_grid.py --out $R/exp014-ne16-$t $opts --n 500 --specs fp32 --seeds 42 --moteur coreml-ne
    comparer $PY $R/exp014-ne16-$t/fp32_s42.npz $R/exp012-ne-$t/fp32_s42.npz "témoin 16 bits contre EXP-012, $t" non_bloquant || \
      echo "[info] témoin 16 bits différent d'EXP-012 ($t) : rapporté, non bloquant"
    # 4. codages alternés graine par graine (une dérive touche les trois codages pareil)
    for g in 42 43 44 45 46; do
      verifier $PY
      for c in $CODAGES; do
        caffeinate -i $PY run_grid.py --out $R/exp014-$c-$t $opts --n 500 --specs fp32 --seeds $g --moteur coreml-ne --codage $c
        place $PY $R/exp014-$c-$t/fp32_s$g.temps.json        # tout sur le Neural Engine, sinon arrêt tout de suite
      done
    done
    # 5. reproduction finale, graine 46 : rien n'a dérivé pendant le calcul
    verifier $PY
    caffeinate -i $PY run_grid.py --out $R/exp014-fin-$t $opts --n 500 --specs fp32 --seeds 46
    comparer $PY $R/exp014-fin-$t/fp32_s46.npz $R/$ref/fp32_s46.npz "reproduction finale de $ref, graine 46"
  done
  echo "[info] EXP-014 finie à $(date '+%H:%M')"
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
  pgrep -f '^[^ ]*[Pp]ython[^ ]* ([^ ]+ )*(run_grid|vitesse|balayage_lot|charge_vision)\.py( |$)' > /dev/null ||
    pgrep -f '^(/bin/)?zsh (\./|[^ ]*/)?lancer_(exp011|exp012|exp013|grille)\.sh( |$)' > /dev/null
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
  $1 -B -c 'import subprocess, analyse_exp014 as A14
A14.verifier_code(subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())'
}

comparer() {
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
