# Pré-enregistrement d'EXP-006 : les résultats d'EXP-003 et d'EXP-004 tiennent-ils sur TwoRoom ?

Rédigé par Claude le 29 septembre 2026 au soir, à la demande de Thomas (élargir à une autre tâche), avant tout
calcul d'EXP-006. Figé par le commit enregistré dans les fichiers. Seul un test de fonctionnement de 4 épisodes
sur CPU a été joué (fichiers hors du dossier de résultats).

## Plan

- Tâche : TwoRoom (stable-worldmodel `swm/TwoRoom-v1`), un agent ponctuel va d'une pièce à l'autre par une
  porte ; succès si sa distance au but passe sous 16. Modèle : LeWM entraîné sur TwoRoom
  (`checkpoints/tworoom/lewm`, même architecture que celui de PushT). Protocole officiel de `le-wm/config/eval/
  tworoom.yaml` : but à 25 pas, budget de 50 pas, CEM 300 × 30, horizon 5.
- Code : `run_grid.py` avec l'option `--tache tworoom` ; sans elle, calcul identique à PushT (`test_run_grid.py`).
- Épisodes : 500, graine de tirage 42, au plus une position par trajectoire (règle d'EXP-005) : épisodes
  indépendants. Graines de la CEM : 42 à 46 ; réplique de la pleine
  précision : 47 à 51.
- Versions (RTN par canal sauf mention) : fp32, all:8, all:6, all:4, all:3, encoder:4, predictor:4, autres:4 (les
  6 couches d'EXP-004), all:4:g64 (groupes de 64). 50 fichiers de 500 épisodes, plus une sanité de clôture (all:4
  graine 42 recalculé, identique octet pour octet).

## Tests (Holm sur 4, inversion de signe comme EXP-003, unilatéraux dans le sens vu sur PushT)

x(e) = K(e, version) - K(e, référence), succès sur les 5 graines 42 à 46.

| Test | Version contre référence | Sens attendu | Sur PushT |
| --- | --- | --- | --- |
| R1 | all:4 contre fp32 | réussite plus basse | P1 d'EXP-003, -14,1 points |
| R2 | encoder:4 contre predictor:4 | réussite plus basse | P2 d'EXP-003, -5,3 points |
| R3 | autres:4 contre fp32 | réussite plus basse | Q3a d'EXP-004, -9,4 points |
| R4 | all:4:g64 contre all:4 | réussite plus haute | Q5a d'EXP-004, +6,2 points |

Chaque test est soutenu si sa p corrigée par Holm est sous 0,05. Un test non soutenu ne prouve pas l'absence
d'effet sur TwoRoom.

## Descriptif, sans test

Réussite par version et écart à fp32 (bootstrap percentile, `default_rng(2)`) ; écart de log de la marge
M = distance minimale au but / 16 (plancher 0,05) ; réplique fp32 et plancher de bruit (épisodes discordants
par 50 entre deux graines) ; écart à l'additivité à 4 bits (all:4 moins la somme des trois parties).

## Contrôles bloquants

50 fichiers attendus, un seul commit, arbre propre ; code d'analyse et de calcul identiques à ce commit ; mêmes
épisodes, départs et buts partout ; nombre de couches quantifiées ; état fini jusqu'au succès et gelé après ;
succès si et seulement si la distance minimale passe sous 16 (tolérance d'arrondi 1e-4) ; sanité de clôture
identique. Le journal du lancement, qui affiche les réussites, n'est pas lu avant l'analyse.
