# EXP-006 : analyse pré-enregistrée (TwoRoom)

Contrôles passés : 50 fichiers, commit 40676cb.

| Test | Comparaison | Écart (points) | p | p Holm | Verdict |
| --- | --- | --- | --- | --- | --- |
| R1 | all:4 contre fp32 | -11.0 [-13.4, -8.8] | 0.0001 | 0.0004 | soutenue |
| R2 | encoder:4 contre predictor:4 | -9.2 [-11.6, -6.9] | 0.0001 | 0.0004 | soutenue |
| R3 | autres:4 contre fp32 | -1.2 [-2.3, -0.1] | 0.0177 | 0.0177 | soutenue |
| R4 | all:4:g64 contre all:4 | +6.5 [+4.9, +8.2] | 0.0001 | 0.0004 | soutenue |

| Version | Réussite | Écart à fp32 (points) | Écart de log marge |
| --- | --- | --- | --- |
| fp32 | 84.8 % |  |  |
| all:8 | 84.8 % | +0.0 [-1.0, +1.0] | -0.008 [-0.019, +0.004] |
| all:6 | 83.6 % | -1.2 [-2.2, -0.2] | +0.008 [-0.005, +0.020] |
| all:4 | 73.8 % | -11.0 [-13.4, -8.8] | +0.111 [+0.082, +0.141] |
| all:3 | 38.6 % | -46.2 [-50.2, -42.3] | +0.433 [+0.385, +0.484] |
| encoder:4 | 75.7 % | -9.1 [-11.4, -6.8] | +0.086 [+0.056, +0.116] |
| predictor:4 | 84.9 % | +0.1 [-0.8, +1.0] | -0.008 [-0.020, +0.004] |
| autres:4 | 83.6 % | -1.2 [-2.3, -0.1] | +0.014 [-0.000, +0.029] |
| all:4:g64 | 80.3 % | -4.5 [-6.5, -2.6] | +0.050 [+0.023, +0.077] |

Réplique fp32 (graines 47 à 51) : 83.9 %, écart -0.9 [-2.0, +0.1] points ; plancher de bruit : +4.02 [+3.24, +4.82] épisodes discordants sur 50 ; écart à l'additivité à 4 bits : -0.9 [-3.0, +1.2] points ; épisodes partant déjà à moins de 16 du but : 53.
