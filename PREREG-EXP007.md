# Pré-enregistrement d'EXP-007 : replanifier plus souvent rattrape-t-il la compression ? (Q4, PushT)

Rédigé par Claude le 30 septembre 2026 au matin, avant tout calcul d'EXP-007 (un test de fonctionnement de 2
épisodes sur CPU, hors du dossier de résultats). Figé par le commit enregistré dans les fichiers.

## Plan

- Versions : fp32 et all:4 (RTN par canal). Graines de la CEM 42 à 44. Épisodes : les 500 d'EXP-003 (tirage par
  défaut, identique à EXP-003, vérifié par `test_run_grid.py`).
- Conditions de planification (le plan couvre toujours 5 blocs de 5 actions) :

| Condition | Replanifier tous les | Itérations de la CEM | Calcul maximal par épisode | Source |
| --- | --- | --- | --- | --- |
| r25 | 25 pas | 30 | 1 (2 plans) | fichiers d'EXP-003, graines 42 à 44 |
| r5c6 | 5 pas | 6 | 1 (10 plans courts) | nouveau |
| r5 | 5 pas | 30 | 5 | nouveau |
| r10 | 10 pas | 30 | 2,5 | nouveau, descriptif |

  18 fichiers nouveaux de 500 épisodes, plus une sanité de clôture (all:4 graine 42, r5c6, identique octet pour
  octet). Réglages par les options `--replan` et `--cem-iter` de `run_grid.py` (receding_horizon, n_steps).

## Tests (Holm sur 3, inversion de signe comme EXP-003, unilatéraux « plus grand »)

K(e, c, v) : succès de l'épisode e sur les 3 graines, condition c, version v.

- Q4a : K(all:4, r5) - K(all:4, r25) : replanifier tous les 5 pas relève la réussite à 4 bits.
- Q4b : [K(all:4, r5) - K(all:4, r25)] - [K(fp32, r5) - K(fp32, r25)] : le gain est plus grand à 4 bits qu'en
  pleine précision (la replanification compense la compression).
- Q4c : K(all:4, r5c6) - K(all:4, r25) : à calcul égal, replanifier souvent avec une CEM courte relève la réussite
  à 4 bits.

Soutenu si la p corrigée par Holm est sous 0,05. Descriptif : réussite par condition et version, gains sur r25,
écart de 4 bits à fp32 dans chaque condition (bootstrap percentile, `default_rng(2)`).

## Contrôles bloquants

Tous les fichiers présents ; protocole de chaque dossier (500 épisodes, PushT, receding_horizon et n_steps
attendus) ; mêmes épisodes, départs et buts qu'EXP-003 ; couches quantifiées ; contrôles de course d'EXP-003 ;
fichiers nouveaux d'un seul commit, dont le code d'analyse et de calcul est celui du dépôt ; sanité de clôture
identique. Le journal du lancement n'est pas lu avant l'analyse.
