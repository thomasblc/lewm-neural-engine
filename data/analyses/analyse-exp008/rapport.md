# EXP-008 : cases manquantes de la matrice sur TwoRoom (descriptif)

Contrôles passés : 35 fichiers, commit ab7b48f, reproduction d'EXP-006 identique. Pleine précision (EXP-006) : 84.8 %.

| Version | Réussite | Écart à fp32 (points, IC 95 %) |
| --- | --- | --- |
| all:5 | 83.5 % | -1.3 [-2.8, +0.2] |
| encoder:3 | 47.0 % | -37.8 [-41.8, -34.1] |
| predictor:3 | 83.8 % | -1.0 [-2.3, +0.2] |
| autres:3 | 82.7 % | -2.1 [-3.8, -0.4] |
| encpred:4 | 75.6 % | -9.2 [-11.6, -7.0] |
| all:3:g64 | 37.9 % | -46.9 [-51.0, -42.9] |
