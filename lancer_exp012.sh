#!/bin/zsh
# EXP-012 (PREREG-EXP012.md, amendement 1) : le prédicteur en Core ML 16 bits, sur le Neural Engine et sur le GPU.
# Par tâche (PushT puis TwoRoom) : puissance au repos, reproduction bloquante de la pleine précision PyTorch (graine
# 42), puis graines 42 à 46 avec les moteurs alternés (coreml-ne puis coreml-gpu à chaque graine, PyTorch en plus à
# la graine 43, reproduction bloquante), enfin reproduction finale de la graine 46. La garde de batterie suit tous les
# calculs d'exp012 et sort quand la reproduction finale de la tâche est écrite. caffeinate empêche la mise en veille.
# Tout le corps est dans des fonctions, lues en entier avant d'être exécutées.
# Relancer après une interruption avec >> pour garder le journal : ./lancer_exp012.sh >> ../resultats/exp012.log 2>&1
set -e
ICI="${0:A:h}"                 # dossier du script ; dans une fonction, $0 vaut le nom de la fonction
main() {
  cd "$ICI" || exit 1
  caffeinate -i -w $$ &!         # pas de veille pendant l'attente ni entre les étapes, jusqu'à la sortie du lanceur
  local PY=/Users/thomasblanc/Documents/PRO/world-models-demos/le-wm/.venv/bin/python
  local R=../resultats
  local PUSHT=(--chunk 50)                                           # les épisodes d'EXP-003
  local TWOROOM=(--tache tworoom --chunk 50 --graine-episodes 42)    # les épisodes d'EXP-006
  local GARDE=/Users/thomasblanc/Documents/PRO/these-ia-thomas/outils/garde_batterie.sh

  # 0. code figé, arbre propre ; revérifié avant chaque étape, HEAD compris
  local HEAD0=""
  verifier $PY
  HEAD0=$(git rev-parse HEAD)
  echo "[info] code vérifié : $(git rev-parse --short HEAD)"
  # une grille commencée avec un autre commit ne se reprend pas : ses fichiers seraient de deux commits
  local d commit_grille
  for d in exp012-ne-pusht exp012-gpu-pusht exp012-torch-pusht exp012-ne-tworoom exp012-gpu-tworoom exp012-torch-tworoom \
           exp012-fin-pusht exp012-fin-tworoom; do
    [ -f $R/$d/meta.json ] || continue
    commit_grille=$($PY -c 'import json, sys; print(json.load(open(sys.argv[1]))["git"])' $R/$d/meta.json)
    [ "$commit_grille" = "$HEAD0" ] || { echo "[arrêt] $d commencée au commit ${commit_grille:0:7}, HEAD est ${HEAD0:0:7}"; exit 1; }
  done

  # 1. le GPU est partagé : attendre 3 contrôles vides de suite (une grille, ou le lanceur d'une autre expérience
  #    entre deux de ses étapes), 24 h au plus ; 2. garde de batterie, après la sortie de celle du calcul précédent.
  # une reproduction finale d'un lancement précédent ferait sortir la garde tout de suite : elle est refaite
  rm -rf $R/exp012-fin-pusht $R/exp012-fin-tworoom
  local debut=$(date +%s)
  while true; do
    attendre_gpu
    garde exp012-fin-pusht 1
    autre_calcul || break
    echo "[info] un autre calcul a démarré pendant l'attente de la garde, nouvelle attente à $(date '+%H:%M')"
  done
  echo "[info] GPU libre, lancement à $(date '+%H:%M')"

  local t opts ref g
  for t in pusht tworoom; do
    if [ $t = pusht ]; then opts=($PUSHT); ref=exp003; else opts=($TWOROOM); ref=exp006; fi
    verifier $PY
    [ $t = pusht ] || garde exp012-fin-$t 1
    # 3. puissance au repos (aucun calcul de l'expérience), référence de l'énergie par plan
    $PY -B mesurer_repos.py $R $t 130     # une mesure par lancement, jamais écrasée ; la garde tolère 3 min sans calcul
    # 4. reproduction de la pleine précision PyTorch, graine 42 (bloquante, référence de la sentinelle)
    rm -rf $R/exp012-repro-$t
    caffeinate -i $PY run_grid.py --out $R/exp012-repro-$t $opts --n 500 --specs fp32 --seeds 42
    comparer $PY $R/exp012-repro-$t/fp32_s42.npz $R/$ref/fp32_s42.npz "reproduction de $ref, graine 42"
    # 5. moteurs alternés graine par graine (temps et puissance dans les mêmes conditions) ; PyTorch à la graine 43
    #    (temps dans les mêmes conditions, deuxième reproduction bloquante)
    for g in 42 43 44 45 46; do
      verifier $PY
      caffeinate -i $PY run_grid.py --out $R/exp012-ne-$t $opts --n 500 --specs fp32 --seeds $g --moteur coreml-ne
      caffeinate -i $PY run_grid.py --out $R/exp012-gpu-$t $opts --n 500 --specs fp32 --seeds $g --moteur coreml-gpu
      if [ $g = 43 ]; then
        caffeinate -i $PY run_grid.py --out $R/exp012-torch-$t $opts --n 500 --specs fp32 --seeds 43
        comparer $PY $R/exp012-torch-$t/fp32_s43.npz $R/$ref/fp32_s43.npz "reproduction de $ref, graine 43"
      fi
    done
    # 6. reproduction finale, graine 46 : rien n'a dérivé pendant le calcul
    verifier $PY
    caffeinate -i $PY run_grid.py --out $R/exp012-fin-$t $opts --n 500 --specs fp32 --seeds 46
    comparer $PY $R/exp012-fin-$t/fp32_s46.npz $R/$ref/fp32_s46.npz "reproduction finale de $ref, graine 46"
  done
  echo "[info] EXP-012 finie à $(date '+%H:%M')"
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

# EXP-011 passe après EXP-012 : son lanceur n'est pas attendu ici (il attend celui-ci), sinon les deux s'attendraient ;
# un calcul run_grid d'EXP-011 en cours reste vu par le premier motif.
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
  $1 -B -c 'import subprocess, analyse_exp012 as A12
A12.verifier_code(subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())'
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
