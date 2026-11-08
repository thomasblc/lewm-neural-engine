# EXP-004 : analyse pré-enregistrée

Contrôles passés : EXP-003 (6e5ca65, c5703db), EXP-004 (499b603), sanité identique à EXP-003 (499b603). Courses divergentes : aucune.

## Tests (Holm sur 4)

| Test | Comparaison | Écart (points) | p | p Holm | Verdict du test | Verdict retenu (règle 2) |
| --- | --- | --- | --- | --- | --- | --- |
| Q3a | autres:4 contre fp32 | -9.44 | 0.0001 | 0.0004 | soutenue | soutenue |
| Q3b | encpred:4 contre all:4 | +10.08 | 0.0001 | 0.0004 | soutenue | soutenue |
| Q5a | all:4:g64 contre all:4 | +6.20 | 0.0002 | 0.0004 | soutenue | soutenue |
| Q5b | all:3:g64 contre all:3 | +10.00 | 0.0001 | 0.0004 | soutenue | soutenue |

## Sensibilité : graines 44 à 46 seulement (Holm sur 4)

| Test | Écart (points) | p | p Holm | Verdict | Même verdict que le test principal |
| --- | --- | --- | --- | --- | --- |
| Q3a | -10.00 | 0.0001 | 0.0004 | soutenue | oui |
| Q3b | +9.40 | 0.0001 | 0.0004 | soutenue | oui |
| Q5a | +6.20 | 0.0005 | 0.0005 | soutenue | oui |
| Q5b | +10.13 | 0.0001 | 0.0004 | soutenue | oui |

Q3 n'est affirmée que si le test principal et la sensibilité concordent (PREREG-EXP004, règle 2).

## Par version (5 graines, 500 épisodes ; écarts à fp32, intervalle bootstrap à 95 %)

| Version | Réussite | Écart (points) | Écart log M3, strate où le T doit bouger |
| --- | --- | --- | --- |
| fp32 | 83.5 % | | |
| all:4 | 69.4 % | -14.1 [-17.2, -11.2] | +0.205 [+0.142, +0.270] |
| all:3 | 35.7 % | -47.8 [-51.5, -44.1] | +0.720 [+0.634, +0.807] |
| encoder:4 | 78.9 % | -4.6 [-6.8, -2.6] | +0.155 [+0.107, +0.207] |
| predictor:4 | 84.2 % | +0.6 [-0.7, +1.9] | +0.001 [-0.027, +0.028] |
| autres:4 | 74.1 % | -9.4 [-12.1, -6.8] | +0.099 [+0.047, +0.152] |
| encpred:4 | 79.5 % | -4.0 [-6.3, -1.8] | +0.164 [+0.113, +0.218] |
| all:4:g64 | 75.6 % | -7.9 [-10.5, -5.4] | +0.136 [+0.082, +0.192] |
| autres:3 | 54.6 % | -29.0 [-32.8, -25.2] | +0.328 [+0.252, +0.405] |
| all:3:g64 | 45.7 % | -37.8 [-41.6, -34.0] | +0.574 [+0.488, +0.661] |

## Additivité et 3 bits (points, intervalle bootstrap à 95 %)

- I : tout le modèle moins la somme des trois parties (4 bits) : -0.7 [-3.9, +2.4]
- J : encodeur et prédicteur ensemble moins la somme des deux (4 bits) : -0.0 [-1.6, +1.6]
- autres:3 contre fp32 : -29.0 [-32.8, -25.2]
- autres:3 contre all:3 : +18.8 [+15.3, +22.4]

Part de la chute de tout le modèle à 4 bits que les autres couches portent en contexte, Σ [K(encpred:4) - K(all:4)] / Σ [K(fp32) - K(all:4)] : 71 [58, 86] %.

Coût en bits par poids (échelles en 16 bits) : all:4 4,06, all:4:g64 4,25, all:3 3,06, all:3:g64 3,25, autres:4 4,05, autres:3 3,05, encpred:4 4,06.
