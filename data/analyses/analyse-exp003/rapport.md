# EXP-003 : analyse pré-enregistrée (amendements 2 à 4)

Contrôles bloquants passés : 55 fichiers, commits 6e5ca65, c5703db, 0 cas limites d'arrondi, 328 épisodes où le T doit bouger, 70 où il part au but, courses divergentes : aucune.

## Tests pré-enregistrés (Holm sur P1, P2, P3' sur M3, P3' sur M2)

| Test | Effet | p | p Holm | Verdict |
| --- | --- | --- | --- | --- |
| P1 : all:4 contre fp32 | -14.12 points | 0.0001 | 0.0004 | soutenue |
| P2 : predictor:4 contre encoder:4 | +5.28 points | 0.0001 | 0.0004 | soutenue |
| P3' : all:5, réussite à ± 2 points et log M3 dégradée | -3.00 points, log M3 +0.075 (328 épisodes où le T doit bouger) | 0.8564 (bas 0.8564, haut 0.0001, M3 0.0001) | 1.0000 | non établie |
| P3' sur M2 : all:5, réussite à ± 2 points et log M2 dégradée (500 épisodes) | -3.00 points, log M2 +0.039 | 0.8564 (M2 0.0007) | 1.0000 | non établie |

Contrôle de P3'(a) par TOST t : p = 0.8550, intervalle à 90 % [-4.56, -1.44] points.

## Par version (5 graines, 500 épisodes ; écarts à fp32, intervalle bootstrap à 95 %)

| Version | Réussite | Écart (points) | Écart log M3 | Écart log M2 | Réussite / T placé seul / T jamais placé |
| --- | --- | --- | --- | --- | --- |
| fp32 | 83.5 % | | | | 83.5 / 10.4 / 6.0 |
| all:8 | 83.8 % | +0.3 [-0.8, +1.4] | +0.003 [-0.012, +0.018] | +0.007 [-0.006, +0.021] | 83.8 / 10.1 / 6.1 |
| all:6 | 83.2 % | -0.3 [-1.6, +0.9] | +0.021 [+0.002, +0.041] | +0.022 [+0.005, +0.039] | 83.2 / 10.3 / 6.5 |
| all:5 | 80.5 % | -3.0 [-4.8, -1.2] | +0.052 [+0.027, +0.076] | +0.039 [+0.017, +0.061] | 80.5 / 12.2 / 7.2 |
| all:4 | 69.4 % | -14.1 [-17.2, -11.2] | +0.160 [+0.116, +0.206] | +0.129 [+0.092, +0.165] | 69.4 / 19.9 / 10.7 |
| all:3 | 35.7 % | -47.8 [-51.5, -44.1] | +0.508 [+0.443, +0.576] | +0.555 [+0.499, +0.613] | 35.7 / 37.5 / 26.8 |
| encoder:4 | 78.9 % | -4.6 [-6.8, -2.6] | +0.116 [+0.081, +0.153] | +0.048 [+0.021, +0.074] | 78.9 / 12.2 / 9.0 |
| encoder:3 | 53.7 % | -29.8 [-33.4, -26.3] | +0.422 [+0.362, +0.487] | +0.318 [+0.271, +0.368] | 53.7 / 25.8 / 20.5 |
| predictor:4 | 84.2 % | +0.6 [-0.7, +1.9] | +0.001 [-0.020, +0.022] | -0.003 [-0.020, +0.014] | 84.2 / 10.0 / 5.9 |
| predictor:3 | 81.3 % | -2.2 [-4.3, -0.2] | +0.009 [-0.019, +0.038] | +0.014 [-0.014, +0.040] | 81.3 / 11.7 / 7.0 |
| fp32 replique | 83.7 % | +0.2 [-1.1, +1.4] | +0.013 [-0.007, +0.034] | +0.005 [-0.011, +0.022] | 83.7 / 10.2 / 6.2 |

predictor:4 moins encoder:4 : +5.3 [+3.2, +7.4] points.

predictor:3 moins encoder:3 : +27.6 [+23.9, +31.3] points.

## Plancher de bruit

Entre deux graines de fp32 : 5.42 [4.54, 6.32] épisodes discordants sur 50 (contre la réplique : 5.66 [4.84, 6.48]).

| Version | Perdus par 50 (graines différentes) | Gagnés par 50 | Ratio pertes du bruit / pertes de la version |
| --- | --- | --- | --- |
| fp32 | 2.71 | 2.71 | |
| all:8 | 2.66 | 2.80 | 102 % |
| all:6 | 2.89 | 2.73 | 94 % |
| all:5 | 4.36 | 2.86 | 62 % |
| all:4 | 9.89 | 2.83 | 27 % |
| all:3 | 25.03 | 1.13 | 11 % |
| encoder:4 | 5.22 | 2.91 | 52 % |
| encoder:3 | 16.71 | 1.80 | 16 % |
| predictor:4 | 2.70 | 3.02 | 100 % |
| predictor:3 | 4.26 | 3.17 | 64 % |

## Protocoles simulés (1 000 tirages d'épisodes ; m = 1 : toutes les graines énumérées)

Retenue : l'intervalle du plan complet exclut 0 (seules ces lignes portent le message). Stable : le signe du plan complet ne change pas quand on retire une graine.

### n50_graines1

| Comparaison | Retenue | Stable | Variante | Même signe | Nul | Signe contraire | Conclut même sens | Ne conclut pas | Conclut sens contraire | Significatif |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fp32 -> all:8 | non | non | apparie | 41.0 | 22.7 | 36.3 | 0.1 | 99.7 | 0.2 | 0.3 |
| fp32 -> all:8 | non | non | non_apparie | 41.9 | 17.2 | 40.9 | 0.6 | 98.9 | 0.5 | 1.1 |
| fp32 -> all:6 | non | non | apparie | 42.1 | 20.2 | 37.7 | 0.6 | 99.1 | 0.2 | 0.9 |
| fp32 -> all:6 | non | non | non_apparie | 45.3 | 16.2 | 38.5 | 0.9 | 98.7 | 0.5 | 1.3 |
| fp32 -> all:5 | oui | oui | apparie | 65.8 | 14.9 | 19.3 | 2.4 | 97.6 | 0.0 | 2.4 |
| fp32 -> all:5 | oui | oui | non_apparie | 65.1 | 12.4 | 22.4 | 3.3 | 96.5 | 0.2 | 3.5 |
| fp32 -> all:4 | oui | oui | apparie | 98.1 | 1.1 | 0.8 | 43.8 | 56.2 | 0.0 | 43.8 |
| fp32 -> all:4 | oui | oui | non_apparie | 98.2 | 1.0 | 0.9 | 42.6 | 57.4 | 0.0 | 42.6 |
| fp32 -> all:3 | oui | oui | apparie | 100.0 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 | 100.0 |
| fp32 -> all:3 | oui | oui | non_apparie | 100.0 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 | 100.0 |
| fp32 -> encoder:4 | oui | oui | apparie | 75.9 | 10.6 | 13.5 | 5.9 | 94.0 | 0.1 | 6.0 |
| fp32 -> encoder:4 | oui | oui | non_apparie | 75.9 | 9.8 | 14.2 | 6.2 | 93.7 | 0.1 | 6.3 |
| fp32 -> encoder:3 | oui | oui | apparie | 100.0 | 0.0 | 0.0 | 96.7 | 3.3 | 0.0 | 96.7 |
| fp32 -> encoder:3 | oui | oui | non_apparie | 100.0 | 0.0 | 0.0 | 96.5 | 3.5 | 0.0 | 96.5 |
| fp32 -> predictor:4 | non | oui | apparie | 45.4 | 19.8 | 34.8 | 0.4 | 99.5 | 0.1 | 0.5 |
| fp32 -> predictor:4 | non | oui | non_apparie | 45.7 | 16.6 | 37.7 | 0.7 | 98.9 | 0.3 | 1.1 |
| fp32 -> predictor:3 | oui | oui | apparie | 59.7 | 14.6 | 25.7 | 2.0 | 97.9 | 0.1 | 2.1 |
| fp32 -> predictor:3 | oui | oui | non_apparie | 59.6 | 13.6 | 26.8 | 2.6 | 97.2 | 0.3 | 2.8 |
| encoder:4 -> predictor:4 | oui | oui | apparie | 79.3 | 10.1 | 10.6 | 7.5 | 92.5 | 0.0 | 7.5 |
| encoder:3 -> predictor:3 | oui | oui | apparie | 100.0 | 0.0 | 0.0 | 90.9 | 9.1 | 0.0 | 90.9 |
| nul vrai : fp32 replique -> fp32 | non |  | non_apparie |  | 17.0 |  |  |  |  | 1.2 |

### n500_graines1

| Comparaison | Retenue | Stable | Variante | Même signe | Nul | Signe contraire | Conclut même sens | Ne conclut pas | Conclut sens contraire | Significatif |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fp32 -> all:8 | non | non | apparie | 40.0 | 0.0 | 60.0 | 0.0 | 100.0 | 0.0 | 0.0 |
| fp32 -> all:8 | non | non | non_apparie | 60.0 | 4.0 | 36.0 | 0.0 | 100.0 | 0.0 | 0.0 |
| fp32 -> all:6 | non | non | apparie | 40.0 | 0.0 | 60.0 | 20.0 | 80.0 | 0.0 | 20.0 |
| fp32 -> all:6 | non | non | non_apparie | 56.0 | 0.0 | 44.0 | 4.0 | 96.0 | 0.0 | 4.0 |
| fp32 -> all:5 | oui | oui | apparie | 100.0 | 0.0 | 0.0 | 80.0 | 20.0 | 0.0 | 80.0 |
| fp32 -> all:5 | oui | oui | non_apparie | 100.0 | 0.0 | 0.0 | 44.0 | 56.0 | 0.0 | 44.0 |
| fp32 -> all:4 | oui | oui | apparie | 100.0 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 | 100.0 |
| fp32 -> all:4 | oui | oui | non_apparie | 100.0 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 | 100.0 |
| fp32 -> all:3 | oui | oui | apparie | 100.0 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 | 100.0 |
| fp32 -> all:3 | oui | oui | non_apparie | 100.0 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 | 100.0 |
| fp32 -> encoder:4 | oui | oui | apparie | 100.0 | 0.0 | 0.0 | 80.0 | 20.0 | 0.0 | 80.0 |
| fp32 -> encoder:4 | oui | oui | non_apparie | 100.0 | 0.0 | 0.0 | 68.0 | 32.0 | 0.0 | 68.0 |
| fp32 -> encoder:3 | oui | oui | apparie | 100.0 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 | 100.0 |
| fp32 -> encoder:3 | oui | oui | non_apparie | 100.0 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 | 100.0 |
| fp32 -> predictor:4 | non | oui | apparie | 60.0 | 0.0 | 40.0 | 0.0 | 100.0 | 0.0 | 0.0 |
| fp32 -> predictor:4 | non | oui | non_apparie | 60.0 | 12.0 | 28.0 | 0.0 | 100.0 | 0.0 | 0.0 |
| fp32 -> predictor:3 | oui | oui | apparie | 100.0 | 0.0 | 0.0 | 20.0 | 80.0 | 0.0 | 20.0 |
| fp32 -> predictor:3 | oui | oui | non_apparie | 84.0 | 0.0 | 16.0 | 20.0 | 80.0 | 0.0 | 20.0 |
| encoder:4 -> predictor:4 | oui | oui | apparie | 100.0 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 | 100.0 |
| encoder:3 -> predictor:3 | oui | oui | apparie | 100.0 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 | 100.0 |
| nul vrai : fp32 replique -> fp32 | non |  | non_apparie |  | 0.0 |  |  |  |  | 4.0 |

### n10_graines3

| Comparaison | Retenue | Stable | Variante | Même signe | Nul | Signe contraire | Conclut même sens | Ne conclut pas | Conclut sens contraire | Significatif |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fp32 -> all:8 | non | non | apparie | 38.5 | 30.3 | 31.2 | 0.0 | 100.0 | 0.0 | 0.0 |
| fp32 -> all:8 | non | non | non_apparie | 36.7 | 27.5 | 35.8 | 0.1 | 99.8 | 0.1 | 0.2 |
| fp32 -> all:6 | non | non | apparie | 34.3 | 29.7 | 36.0 | 0.2 | 99.8 | 0.0 | 0.2 |
| fp32 -> all:6 | non | non | non_apparie | 40.2 | 24.6 | 35.2 | 0.5 | 99.4 | 0.1 | 0.6 |
| fp32 -> all:5 | oui | oui | apparie | 54.6 | 20.4 | 25.0 | 2.2 | 97.6 | 0.2 | 2.4 |
| fp32 -> all:5 | oui | oui | non_apparie | 57.9 | 16.9 | 25.2 | 2.6 | 97.3 | 0.1 | 2.7 |
| fp32 -> all:4 | oui | oui | apparie | 86.3 | 6.6 | 7.1 | 23.8 | 75.9 | 0.3 | 24.1 |
| fp32 -> all:4 | oui | oui | non_apparie | 89.1 | 4.0 | 6.9 | 25.0 | 75.0 | 0.0 | 25.0 |
| fp32 -> all:3 | oui | oui | apparie | 99.9 | 0.0 | 0.1 | 94.6 | 5.4 | 0.0 | 94.6 |
| fp32 -> all:3 | oui | oui | non_apparie | 100.0 | 0.0 | 0.0 | 95.7 | 4.3 | 0.0 | 95.7 |
| fp32 -> encoder:4 | oui | oui | apparie | 59.7 | 16.6 | 23.7 | 3.7 | 96.2 | 0.1 | 3.8 |
| fp32 -> encoder:4 | oui | oui | non_apparie | 63.5 | 14.3 | 22.2 | 3.5 | 96.3 | 0.2 | 3.7 |
| fp32 -> encoder:3 | oui | oui | apparie | 97.8 | 0.9 | 1.3 | 69.8 | 30.2 | 0.0 | 69.8 |
| fp32 -> encoder:3 | oui | oui | non_apparie | 98.3 | 0.8 | 0.9 | 70.6 | 29.4 | 0.0 | 70.6 |
| fp32 -> predictor:4 | non | oui | apparie | 40.4 | 27.2 | 32.4 | 0.3 | 99.6 | 0.1 | 0.4 |
| fp32 -> predictor:4 | non | oui | non_apparie | 38.7 | 23.7 | 37.6 | 0.4 | 99.3 | 0.3 | 0.7 |
| fp32 -> predictor:3 | oui | oui | apparie | 48.7 | 18.6 | 32.7 | 2.5 | 97.4 | 0.1 | 2.6 |
| fp32 -> predictor:3 | oui | oui | non_apparie | 50.8 | 18.4 | 30.8 | 2.9 | 96.9 | 0.2 | 3.1 |
| encoder:4 -> predictor:4 | oui | oui | apparie | 65.8 | 15.1 | 19.1 | 4.8 | 95.0 | 0.2 | 5.0 |
| encoder:3 -> predictor:3 | oui | oui | apparie | 96.2 | 1.4 | 2.4 | 62.2 | 37.8 | 0.0 | 62.2 |
| nul vrai : fp32 replique -> fp32 | non |  | non_apparie |  | 23.2 |  |  |  |  | 0.3 |

### n50_graines5

| Comparaison | Retenue | Stable | Variante | Même signe | Nul | Signe contraire | Conclut même sens | Ne conclut pas | Conclut sens contraire | Significatif |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fp32 -> all:8 | non | non | apparie | 52.2 | 10.4 | 37.4 | 2.3 | 97.2 | 0.5 | 2.8 |
| fp32 -> all:8 | non | non | non_apparie | 50.9 | 8.3 | 40.8 | 1.8 | 97.7 | 0.5 | 2.3 |
| fp32 -> all:6 | non | non | apparie | 52.1 | 8.9 | 39.0 | 1.6 | 97.4 | 1.0 | 2.6 |
| fp32 -> all:6 | non | non | non_apparie | 53.5 | 7.4 | 39.1 | 1.6 | 97.7 | 0.7 | 2.3 |
| fp32 -> all:5 | oui | oui | apparie | 85.5 | 2.7 | 11.8 | 13.8 | 86.1 | 0.1 | 13.9 |
| fp32 -> all:5 | oui | oui | non_apparie | 85.8 | 3.6 | 10.6 | 12.7 | 87.3 | 0.0 | 12.7 |
| fp32 -> all:4 | oui | oui | apparie | 99.9 | 0.1 | 0.0 | 81.4 | 18.6 | 0.0 | 81.4 |
| fp32 -> all:4 | oui | oui | non_apparie | 100.0 | 0.0 | 0.0 | 85.2 | 14.8 | 0.0 | 85.2 |
| fp32 -> all:3 | oui | oui | apparie | 100.0 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 | 100.0 |
| fp32 -> all:3 | oui | oui | non_apparie | 100.0 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 | 100.0 |
| fp32 -> encoder:4 | oui | oui | apparie | 91.7 | 2.0 | 6.3 | 22.1 | 77.9 | 0.0 | 22.1 |
| fp32 -> encoder:4 | oui | oui | non_apparie | 93.8 | 1.6 | 4.6 | 23.4 | 76.6 | 0.0 | 23.4 |
| fp32 -> encoder:3 | oui | oui | apparie | 100.0 | 0.0 | 0.0 | 99.9 | 0.1 | 0.0 | 99.9 |
| fp32 -> encoder:3 | oui | oui | non_apparie | 100.0 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 | 100.0 |
| fp32 -> predictor:4 | non | oui | apparie | 60.5 | 6.9 | 32.6 | 1.9 | 97.4 | 0.7 | 2.6 |
| fp32 -> predictor:4 | non | oui | non_apparie | 58.1 | 6.5 | 35.4 | 1.6 | 97.5 | 0.9 | 2.5 |
| fp32 -> predictor:3 | oui | oui | apparie | 73.2 | 4.6 | 22.2 | 6.3 | 93.5 | 0.2 | 6.5 |
| fp32 -> predictor:3 | oui | oui | non_apparie | 73.3 | 4.6 | 22.1 | 5.6 | 94.0 | 0.4 | 6.0 |
| encoder:4 -> predictor:4 | oui | oui | apparie | 95.0 | 1.3 | 3.7 | 30.0 | 70.0 | 0.0 | 30.0 |
| encoder:3 -> predictor:3 | oui | oui | apparie | 100.0 | 0.0 | 0.0 | 99.4 | 0.6 | 0.0 | 99.4 |
| nul vrai : fp32 replique -> fp32 | non |  | non_apparie |  | 6.9 |  |  |  |  | 1.9 |

## Part du planificateur dans la variance d'une étude à une graine

| Version | Appariée (borne basse) | Non appariée |
| --- | --- | --- |
| all:8 | 98 % | 100 % |
| all:6 | 94 % | 96 % |
| all:5 | 78 % | 84 % |
| all:4 | 61 % | 67 % |
| all:3 | 49 % | 51 % |
| encoder:4 | 76 % | 84 % |
| encoder:3 | 50 % | 54 % |
| predictor:4 | 94 % | 96 % |
| predictor:3 | 73 % | 78 % |
| fp32 replique |  | 100 % |

## Épisodes cassés (baisse d'au moins 3 succès sur 5 contre la réplique)

Nul (fp32 contre sa réplique) : 4 épisodes. Strates de la réplique : replique 5/5 347, replique 1 a 4 121, replique 0/5 32.

| Version | Observés | Nuls | p (unilatérale) | Réplique 5/5 | Réplique 1 à 4 | Mêmes graines (descriptif) |
| --- | --- | --- | --- | --- | --- | --- |
| all:8 | 3 | 4 | 0.8188 | 1 contre 3 | 2 contre 1 | 1 |
| all:6 | 9 | 4 | 0.1155 | 7 contre 3 | 2 contre 1 | 5 |
| all:5 | 18 | 4 | 0.0008 | 10 contre 3 | 8 contre 1 | 18 |
| all:4 | 76 | 4 | 0.0001 | 59 contre 3 | 17 contre 1 | 78 |
| all:3 | 251 | 4 | 0.0001 | 197 contre 3 | 54 contre 1 | 249 |
| encoder:4 | 25 | 4 | 0.0001 | 18 contre 3 | 7 contre 1 | 24 |
| encoder:3 | 153 | 4 | 0.0001 | 114 contre 3 | 39 contre 1 | 154 |
| predictor:4 | 4 | 4 | 0.6810 | 1 contre 3 | 3 contre 1 | 1 |
| predictor:3 | 22 | 4 | 0.0001 | 16 contre 3 | 6 contre 1 | 24 |
