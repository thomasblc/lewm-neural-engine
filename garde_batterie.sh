#!/bin/zsh
# Copie pour EXP-013 de outils/garde_batterie.sh (la garde partagée n'est pas modifiée pendant qu'EXP-011 s'en sert) ; seul
# changement : ABSENTS, le nombre de contrôles d'affilée sans calcul avant de sortir (4 par défaut, 8 pour EXP-013, dont
# les mesures au repos et la vision seule laissent jusqu'à environ 3 min sans run_grid).
# Garde de batterie pour la grille EXP-003 : gèle le processus de calcul (SIGSTOP) quand la batterie
# descend à SEUIL_BAS, le relance (SIGCONT) quand elle remonte au seuil de reprise. Premier seuil de
# reprise : REPRISE_1 (test demandé par Thomas), puis REPRISE. Sort quand la grille est finie (0) ou si le
# processus disparaît avant (1), ce qui réveille la session Claude.
D=/Users/thomasblanc/Documents/PRO/these-ia-thomas/papier-corl-2026/resultats
LOG=$D/garde-batterie.log
GRILLE=${GRILLE:-exp003}          # dossier des résultats suivi, dans $D
ATTENDUS=${ATTENDUS:-55}          # nombre de fichiers .npz à la fin
SEUIL_BAS=${SEUIL_BAS:-20}
REPRISE_1=${REPRISE_1:-30}
REPRISE=${REPRISE:-80}
pause=${PAUSE_INITIALE:-1}
reprise=$REPRISE_1
debut_pause=$(date +%s)
absent=0
ABSENTS=${ABSENTS:-4}
nb_reprises=0
nb_pauses=0
arret=""                          # processus gelé par cette garde ; un autre processus n'hérite pas de la pause

reprendre() {  # la garde ne sort jamais en laissant gelé le processus qu'elle a arrêté
  if [ "$pause" = 1 ] && [ -n "$arret" ]; then
    pause=0
    if ps -o command= -p $arret 2>/dev/null | grep -q 'run_grid\.py'; then   # encore lui, pas un PID réutilisé
      kill -CONT $arret
      echo "$(date '+%H:%M') sortie de la garde : processus $arret relancé" >> $LOG
    else
      echo "$(date '+%H:%M') sortie de la garde : le processus gelé $arret a disparu" >> $LOG
    fi
  fi
}
arreter() { echo "$(date '+%H:%M') garde arrêtée par le signal $1" >> $LOG; reprendre; exit 1; }
trap reprendre EXIT
trap 'arreter HUP' HUP
trap 'arreter INT' INT
trap 'arreter TERM' TERM

courant() {  # courant de la batterie en mA, positif en charge
  local brut=$(ioreg -rn AppleSmartBattery | grep '"InstantAmperage"' | awk '{print $3}')
  python3 -c "v = int('$brut'); print(v - 2**64 if v >= 2**63 else v)"   # entier exact (awk arrondit)
}

echo "$(date '+%H:%M') garde v2 : pause à ${SEUIL_BAS} % ou moins, reprise à ${REPRISE_1} % la première fois puis à ${REPRISE} %" >> $LOG
while true; do
  pct=$(pmset -g batt | grep -Eo '[0-9]+%' | head -1 | tr -d %)
  # seuls les calculs de cette expérience (dossier de sortie qui commence par le préfixe de GRILLE, par exemple
  # exp010 pour exp010 et exp010-repro) ; un processus gelé par cette garde reste suivi tant qu'il existe
  # (1er octobre : un autre run_grid lancé à côté avait fait croire à la disparition du processus gelé)
  py=$(for p in $(pgrep -f "[r]un_grid\.py --out [^ ]*/${GRILLE%%-*}"); do ps -o pid=,comm= -p $p; done | grep -i python | awk '{print $1}' | head -1)
  if [ "$pause" = 1 ] && [ -n "$arret" ] && ps -o command= -p $arret 2>/dev/null | grep -q 'run_grid\.py'; then py=$arret; fi
  n=$(ls $D/$GRILLE/*.npz 2>/dev/null | grep -vc tmp)
  if [ -z "$py" ]; then
    absent=$((absent + 1))
    if [ "$n" -ge "$ATTENDUS" ]; then echo "$(date '+%H:%M') grille $GRILLE terminée : $n fichiers" | tee -a $LOG; exit 0; fi
    if [ "$absent" -ge "$ABSENTS" ]; then
      echo "$(date '+%H:%M') ARRÊT ANORMAL : plus de processus de grille depuis $ABSENTS min, $n fichiers sur $ATTENDUS ($GRILLE)" | tee -a $LOG; exit 1
    fi
  else
    absent=0
    # pause initiale (PAUSE_INITIALE=1) : le processus trouvé est celui qui est déjà gelé, cette garde le relancera
    if [ "$pause" = 1 ] && [ -z "$arret" ]; then arret=$py; fi
    if [ "$pause" = 1 ] && [ -n "$arret" ] && [ "$py" != "$arret" ]; then
      pause=0; reprise=$REPRISE
      echo "$(date '+%H:%M') le processus gelé $arret a disparu ; nouveau processus $py, suivi sans pause" >> $LOG
    fi
    attente=$(( ($(date +%s) - debut_pause) / 60 ))
    if [ "$pause" = 1 ] && { [ "$pct" -ge "$reprise" ] || { [ "$attente" -ge 150 ] && [ "$pct" -ge 50 ]; }; }; then
      kill -CONT $py; pause=0; reprise=$REPRISE; nb_reprises=$((nb_reprises + 1))
      echo "$(date '+%H:%M') reprise $nb_reprises : batterie $pct %, courant $(courant) mA, processus $py, $n fichiers" >> $LOG
    elif [ "$pause" = 0 ] && [ "$pct" -le "$SEUIL_BAS" ]; then
      pause=1; arret=$py; kill -STOP $py; debut_pause=$(date +%s); nb_pauses=$((nb_pauses + 1))   # état noté avant le gel
      echo "$(date '+%H:%M') pause $nb_pauses : batterie $pct %, courant $(courant) mA, processus $py, $n fichiers" >> $LOG
    fi
  fi
  sleep 60 & wait $!                # en arrière-plan : un signal est traité tout de suite
done
