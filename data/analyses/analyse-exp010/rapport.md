# EXP-010 : GPTQ à 4 et 3 bits est-il gratuit ? (PushT, 2 000 épisodes neufs)

Contrôles passés, commit af8e516, reproduction d'EXP-005 identique, calibration d'EXP-009 disjointe des évaluations. Pleine précision : 87.1 %.

| Test (séquence fixée) | Version | Perte de moins de | p | Verdict |
| --- | --- | --- | --- | --- |
| T1 | all:4:gptq | 2 points | 0.8975 | non établie |
| T2 | all:4:gptq | 1 point | 0.9996 | non testée (séquence arrêtée) |
| T3 | all:3:gptq | 2 points | 1.0000 | non testée (séquence arrêtée) |
| T4 | all:3:gptq | 1 point | 1.0000 | non testée (séquence arrêtée) |

- all:4:gptq : réussite 84.5 %, écart -2.58 [-3.48, -1.71] points ; pose (log M3, strate) +0.035 [+0.017, +0.054] ; position +0.48 [+0.23, +0.75] px, angle +0.41 [+0.17, +0.67] degrés.
- all:3:gptq : réussite 76.3 %, écart -10.77 [-12.04, -9.48] points ; pose (log M3, strate) +0.153 [+0.125, +0.180] ; position +1.57 [+1.13, +2.03] px, angle +0.93 [+0.50, +1.34] degrés.
- À côté (EXP-005, RTN, tout le modèle) : 8 bits -0.36 [-0.85, +0.12] points, 6 bits -1.26 [-1.92, -0.61] points.
