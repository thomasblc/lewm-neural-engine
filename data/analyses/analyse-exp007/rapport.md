# EXP-007 : analyse pré-enregistrée (replanification et compression, PushT)

Contrôles passés : 18 fichiers, commit dfe6ff8, courses divergentes 0.

| Test | Écart (points, 3 graines) | p | p Holm | Verdict |
| --- | --- | --- | --- | --- |
| Q4a | -42.9 [-46.9, -38.7] | 1.0000 | 1.0000 | non soutenue |
| Q4b | +10.5 [+6.1, +14.9] | 0.0001 | 0.0003 | soutenue |
| Q4c | -45.5 [-49.7, -41.2] | 1.0000 | 1.0000 | non soutenue |

| Condition, version | Réussite | Gain sur r25 (points) |
| --- | --- | --- |
| r25 fp32 | 83.2 % |  |
| r25 all:4 | 69.1 % |  |
| r5c6 fp32 | 25.9 % | -57.3 [-61.4, -53.1] |
| r5c6 all:4 | 23.7 % | -45.5 [-49.7, -41.2] |
| r5 fp32 | 29.9 % | -53.3 [-57.3, -49.5] |
| r5 all:4 | 26.3 % | -42.9 [-46.9, -38.7] |

Écart de 4 bits à fp32 par condition : r25 -14.1 [-17.3, -10.9] ; r5c6 -2.3 [-4.9, +0.5] ; r5 -3.6 [-6.7, -0.4].
