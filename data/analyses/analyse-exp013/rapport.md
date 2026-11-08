# EXP-013 : vitesse, énergie et mémoire de la planification (analyse pré-enregistrée, descriptive)

Contrôles bloquants passés, commit 0339bf1 ; runs torch identiques au pilote d'EXP-012. Médiane des 3 répétitions (min à max). Énergie de la puce : compteurs IOReport « Energy Model » (estimation de la puce, non validée contre une mesure externe ; affichage exclu), pendant la planification seule, au-dessus du repos (moyenne des repos avant et après la répétition). Règle de constance : « plus bas (ou plus haut) dans les 3 répétitions » si les 3 rapports appariés sont au-delà de 5 % (temps, latence, part de la vision) ou 10 % (énergie), du même côté ; critère descriptif, pas un test, une machine, une session.

## Signalements

- mesure au repos r1-r2 : processus de fond [[23.5, 'Claude'], [27.2, 'Claude']]
- mesure au repos r2-r3 : processus de fond [[20.1, 'Claude'], [22.3, 'Claude']]
- mesure au repos r3-apres : processus de fond [[23.6, 'Claude'], [97.6, 'duetexpertd'], [26.8, 'Claude']]
- mesure au repos sens-apres : processus de fond [[24.0, 'Claude'], [22.9, 'Claude']]
- coreml-ne-int4b32 r1 : Neural Engine à 0.62 W seulement (placement {'CPU': 211, 'NeuralEngine': 150})
- coreml-ne-int4b32 r2 : Neural Engine à 0.62 W seulement (placement {'CPU': 211, 'NeuralEngine': 150})
- coreml-ne-int4b32 r3 : Neural Engine à 0.62 W seulement (placement {'CPU': 211, 'NeuralEngine': 150})
- coreml-gpu-int4b32 r1 : processus systemstats à 98 % de CPU pendant le run
- coreml-gpu-int4c r1 : processus PerfPowerServices à 84 % de CPU pendant le run
- torch-vision r1 : processus Claude à 117 % de CPU pendant le run
- coreml-ne-int8 r3 : processus PerfPowerServices à 93 % de CPU pendant le run

## Sans charge

| Configuration | Temps par plan (ms) | Énergie par plan, puce (J) | calcul (J) | mémoire (J) | brute (J) | Poids du prédicteur (Mo) | Placement | Part de l'appel Core ML | Réussite, 50 épisodes (IC 95 %) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| torch | 768 (767 à 774) | 42.87 (41.74 à 43.22) | 33.13 (32.35 à 33.44) | 4.44 (4.24 à 4.51) | 43.84 (42.73 à 44.22) | 46.3 |  |  | 90 [79 ; 96] |
| torch-fp16 | 600 (599 à 601) | 25.15 (24.89 à 25.15) | 20.09 (19.90 à 20.11) | 2.44 (2.42 à 2.47) | 25.92 (25.64 à 25.93) | 23.2 |  |  | 88 [76 ; 94] |
| coreml-gpu | 382 (377 à 383) | 17.01 (16.63 à 17.13) | 12.64 (12.38 à 12.69) | 1.93 (1.87 à 1.99) | 17.51 (17.12 à 17.61) | 23.2 | GPU 197 | 65 (65 à 65) | 88 [76 ; 94] |
| coreml-ne | 811 (808 à 817) | 10.26 (10.16 à 10.29) | 6.19 (6.13 à 6.24) | 3.94 (3.94 à 3.97) | 11.31 (11.21 à 11.31) | 23.2 | NeuralEngine 361 | 74 (74 à 74) | 86 [74 ; 93] |
| coreml-gpu-int8 | 2438 (2354 à 2481) | 157.08 (148.11 à 170.38) | 133.10 (125.82 à 145.12) | 8.49 (7.75 à 8.62) | 160.24 (151.21 à 173.44) | 11.7 | GPU 197 | 92 (92 à 92) | 94 [84 ; 98] |
| coreml-gpu-int4b32 | 2977 (2931 à 3019) | 175.82 (175.37 à 182.14) | 152.45 (151.08 à 157.85) | 9.72 (9.31 à 10.08) | 179.68 (179.14 à 185.94) | 6.6 | GPU 197 | 93 (93 à 94) | 84 [71 ; 92] |
| coreml-gpu-pal4 | 427 (426 à 467) | 17.33 (14.84 à 17.50) | 13.09 (10.92 à 13.25) | 1.75 (1.74 à 1.84) | 17.86 (15.45 à 18.05) | 5.9 | GPU 197 | 69 (68 à 69) | 84 [71 ; 92] |
| coreml-ne-int8 | 795 (790 à 797) | 10.05 (9.69 à 10.12) | 6.13 (5.94 à 6.19) | 3.83 (3.66 à 3.83) | 11.05 (10.71 à 11.16) | 11.7 | NeuralEngine 361 | 74 (74 à 74) | 94 [84 ; 98] |
| coreml-ne-int4b32 | 2166 (2162 à 2170) | 40.18 (40.07 à 40.28) | 27.90 (27.85 à 28.06) | 12.11 (12.10 à 12.16) | 42.89 (42.89 à 43.09) | 6.6 | CPU 211, NeuralEngine 150 | 85 (85 à 85) | 90 [79 ; 96] |
| coreml-ne-pal4 | 795 (794 à 800) | 9.84 (9.80 à 10.03) | 6.10 (6.06 à 6.16) | 3.66 (3.65 à 3.76) | 10.87 (10.84 à 11.02) | 5.9 | NeuralEngine 361 | 74 (73 à 74) | 86 [74 ; 93] |
| ne-complet | 653 (648 à 655) | 8.43 (8.41 à 8.61) | 4.99 (4.97 à 5.08) | 3.43 (3.42 à 3.51) | 9.28 (9.25 à 9.42) | 23.2 | NeuralEngine 688 | 91 (91 à 92) | 94 [84 ; 98] |
| coreml-cpu | 1771 (1767 à 1772) | 34.65 (34.62 à 34.73) | 26.62 (26.61 à 26.66) | 7.93 (7.92 à 8.04) | 36.94 (36.92 à 36.95) | 23.2 | CPU 688 | 95 (95 à 95) | 96 [87 ; 99] |
| coreml-gpu-int4c | 2965 (2873 à 2975) | 167.52 (167.04 à 179.23) | 144.59 (144.46 à 154.98) | 9.70 (9.34 à 9.97) | 171.37 (170.75 à 182.96) | 5.9 | GPU 197 | 93 (93 à 94) | 84 [71 ; 92] |
| coreml-ne-int4c | 794 (790 à 802) | 9.63 (9.56 à 9.90) | 5.79 (5.78 à 6.01) | 3.75 (3.68 à 3.79) | 10.63 (10.59 à 10.94) | 5.9 | NeuralEngine 361 | 74 (74 à 74) | 84 [71 ; 92] |

Codages int8 (linéaire symétrique par canal), int4b32 (blocs de 32, arrondi sans calibration), int4c (int4 par canal) et pal4 (une palette de 16 valeurs par tenseur, k-moyennes) : fonctions de coremltools 9.0 depuis les poids 16 bits ; placement attendu, celui de l'essai de placement (int4 par blocs sur le Neural Engine : 211 opérations sur le CPU). Le moment de la décompression des poids n'est pas observé ici. Réussite non établie (50 épisodes de contrôle seulement) ; mesurée par une autre expérience. Le coût par plan ne dépend pas des valeurs des poids (CEM sans arrêt anticipé). Core ML complète les longueurs T = 1 et 2 à 3 (prédicteur de forme fixe).

| Comparaison | Temps | Énergie de la puce |
| --- | --- | --- |
| torch-fp16 contre torch | plus bas dans les 3 répétitions (0.78, 0.78, 0.78) | plus bas dans les 3 répétitions (0.58, 0.60, 0.58) |
| coreml-gpu contre torch | plus bas dans les 3 répétitions (0.49, 0.50, 0.49) | plus bas dans les 3 répétitions (0.40, 0.41, 0.38) |
| coreml-ne contre torch | plus haut dans les 3 répétitions (1.05, 1.05, 1.06) | plus bas dans les 3 répétitions (0.24, 0.24, 0.24) |
| coreml-gpu-int8 contre torch | plus haut dans les 3 répétitions (3.20, 3.18, 3.06) | plus haut dans les 3 répétitions (3.45, 3.76, 3.94) |
| coreml-gpu-int4b32 contre torch | plus haut dans les 3 répétitions (3.90, 3.88, 3.81) | plus haut dans les 3 répétitions (4.09, 4.21, 4.21) |
| coreml-gpu-pal4 contre torch | plus bas dans les 3 répétitions (0.55, 0.61, 0.56) | plus bas dans les 3 répétitions (0.40, 0.36, 0.40) |
| coreml-ne-int8 contre torch | non constant (1.03, 1.04, 1.03) | plus bas dans les 3 répétitions (0.23, 0.24, 0.22) |
| coreml-ne-int4b32 contre torch | plus haut dans les 3 répétitions (2.80, 2.82, 2.82) | non constant (0.94, 0.97, 0.93) |
| coreml-ne-pal4 contre torch | non constant (1.03, 1.04, 1.03) | plus bas dans les 3 répétitions (0.23, 0.23, 0.23) |
| ne-complet contre torch | plus bas dans les 3 répétitions (0.84, 0.85, 0.84) | plus bas dans les 3 répétitions (0.20, 0.20, 0.19) |
| coreml-cpu contre torch | plus haut dans les 3 répétitions (2.29, 2.31, 2.30) | plus bas dans les 3 répétitions (0.81, 0.83, 0.80) |
| coreml-gpu-int4c contre torch | plus haut dans les 3 répétitions (3.83, 3.75, 3.87) | plus haut dans les 3 répétitions (3.90, 4.29, 3.88) |
| coreml-ne-int4c contre torch | non constant (1.03, 1.03, 1.04) | plus bas dans les 3 répétitions (0.22, 0.23, 0.23) |
| coreml-ne contre coreml-gpu | plus haut dans les 3 répétitions (2.14, 2.11, 2.15) | plus bas dans les 3 répétitions (0.60, 0.60, 0.62) |
| ne-complet contre coreml-gpu | plus haut dans les 3 répétitions (1.71, 1.71, 1.72) | plus bas dans les 3 répétitions (0.50, 0.50, 0.51) |
| coreml-cpu contre coreml-gpu | plus haut dans les 3 répétitions (4.63, 4.62, 4.68) | plus haut dans les 3 répétitions (2.03, 2.04, 2.08) |
| coreml-gpu contre torch-fp16 | plus bas dans les 3 répétitions (0.64, 0.64, 0.63) | plus bas dans les 3 répétitions (0.69, 0.68, 0.66) |
| ne-complet contre coreml-ne | plus bas dans les 3 répétitions (0.80, 0.81, 0.80) | plus bas dans les 3 répétitions (0.84, 0.83, 0.82) |
| ne-complet contre coreml-cpu | plus bas dans les 3 répétitions (0.37, 0.37, 0.37) | plus bas dans les 3 répétitions (0.25, 0.24, 0.24) |
| coreml-gpu-int8 contre coreml-gpu | plus haut dans les 3 répétitions (6.49, 6.36, 6.24) | plus haut dans les 3 répétitions (8.65, 9.24, 10.25) |
| coreml-gpu-int4b32 contre coreml-gpu | plus haut dans les 3 répétitions (7.90, 7.76, 7.77) | plus haut dans les 3 répétitions (10.24, 10.34, 10.95) |
| coreml-gpu-pal4 contre coreml-gpu | plus haut dans les 3 répétitions (1.11, 1.22, 1.13) | non constant (1.01, 0.87, 1.05) |
| coreml-gpu-int4c contre coreml-gpu | plus haut dans les 3 répétitions (7.75, 7.49, 7.89) | plus haut dans les 3 répétitions (9.75, 10.54, 10.08) |
| coreml-ne-int8 contre coreml-ne | non constant (0.97, 0.99, 0.97) | non constant (0.98, 1.00, 0.94) |
| coreml-ne-int4b32 contre coreml-ne | plus haut dans les 3 répétitions (2.65, 2.68, 2.67) | plus haut dans les 3 répétitions (3.91, 3.97, 3.91) |
| coreml-ne-pal4 contre coreml-ne | non constant (0.97, 0.99, 0.98) | non constant (0.97, 0.96, 0.96) |
| coreml-ne-int4c contre coreml-ne | non constant (0.97, 0.98, 0.99) | non constant (0.94, 0.94, 0.97) |
| coreml-gpu-int4b32 contre coreml-gpu-int4c | non constant (1.02, 1.04, 0.99) | non constant (1.05, 0.98, 1.09) |
| coreml-ne-int4b32 contre coreml-ne-int4c | plus haut dans les 3 répétitions (2.73, 2.74, 2.70) | plus haut dans les 3 répétitions (4.17, 4.21, 4.05) |

## Avec charge de vision sur le GPU (mesures sur les fenêtres de planification)

| Configuration | Temps par plan (ms) | Puissance de la puce pendant la planification (W) | Vision (images/s) | Part de la vision seule | Latence p50 (ms) | Latence p99 (ms) | Plus grand trou sans image (s) | Échéances manquées par s | Énergie marginale de planification, 30 images/s (J par plan, n) | Bornes (puissance de la vision seule min et max) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| torch-vision | 836 (835 à 837) | 74.2 (72.4 à 74.6) | 195.1 (193.2 à 196.0) | 0.94 (0.93 à 0.95) | 5.1 (5.1 à 5.2) | 5.7 (5.6 à 7.2) | 0.01 (0.01 à 0.01) | 0.00 (0.00 à 0.00) |  |  |
| coreml-gpu-vision | 403 (399 à 418) | 61.9 (55.8 à 63.1) | 197.2 (192.2 à 198.1) | 0.96 (0.92 à 0.96) | 5.0 (5.0 à 5.2) | 5.6 (5.5 à 5.8) | 0.01 (0.01 à 0.01) | 0.00 (0.00 à 0.00) |  |  |
| coreml-ne-vision | 754 (752 à 756) | 30.5 (30.3 à 31.0) | 199.9 (199.8 à 200.0) | 0.97 (0.95 à 0.97) | 5.0 (5.0 à 5.0) | 5.4 (5.4 à 5.4) | 0.01 (0.01 à 0.01) | 0.00 (0.00 à 0.00) |  |  |
| ne-complet-vision | 672 (671 à 673) | 30.7 (30.4 à 30.8) | 199.2 (198.2 à 200.5) | 0.97 (0.95 à 0.97) | 5.0 (5.0 à 5.0) | 5.4 (5.4 à 5.5) | 0.01 (0.01 à 0.01) | 0.00 (0.00 à 0.00) |  |  |
| torch-vision30 | 786 (786 à 791) | 58.7 (58.6 à 59.3) | 30.0 (30.0 à 30.0) | 1.00 (1.00 à 1.00) | 5.6 (5.5 à 5.7) | 6.7 (6.7 à 6.7) | 0.04 (0.04 à 0.04) | 0.00 (0.00 à 0.00) | 43.42 (42.79 à 43.51), n = 3 | 42.79 à 43.70 |
| coreml-gpu-vision30 | 388 (384 à 389) | 46.4 (46.2 à 47.0) | 30.0 (30.0 à 30.0) | 1.00 (1.00 à 1.00) | 5.3 (5.3 à 5.3) | 6.8 (6.2 à 7.0) | 0.04 (0.04 à 0.05) | 0.00 (0.00 à 0.00) | 16.50 (16.22 à 16.67), n = 3 | 16.22 à 16.81 |
| coreml-ne-vision30 | 789 (785 à 792) | 17.0 (16.9 à 17.0) | 30.0 (30.0 à 30.0) | 1.00 (1.00 à 1.00) | 6.6 (6.5 à 7.0) | 10.4 (9.9 à 10.4) | 0.05 (0.05 à 0.05) | 0.00 (0.00 à 0.00) | 10.15 (10.08 à 10.47), n = 3 | 10.01 à 10.50 |
| ne-complet-vision30 | 658 (657 à 660) | 17.4 (17.3 à 17.4) | 30.0 (30.0 à 30.0) | 1.00 (1.00 à 1.00) | 7.3 (7.0 à 7.5) | 10.8 (10.5 à 10.9) | 0.05 (0.05 à 0.05) | 0.00 (0.00 à 0.00) | 8.77 (8.73 à 8.98), n = 3 | 8.69 à 9.02 |

Vision seule : vision 206.4 (205.9 à 209.4) images/s, p99 5.2 (5.2 à 5.3) ms, puce 20.1 (20.0 à 20.4) W ; vision30 30.0 (30.0 à 30.0) images/s, p99 11.4 (11.3 à 11.6) ms, puce 4.1 (3.7 à 4.1) W

### Question 2 : le planificateur hors du GPU garde-t-il son temps, et la vision sa cadence ?

| Mode | Contraste | Ralentissement du planificateur sous charge | Part de la vision gardée | Latence p99 de la vision |
| --- | --- | --- | --- | --- |
| vision | coreml-ne contre torch | plus bas dans les 3 répétitions (0.85, 0.86, 0.86) | non constant (1.02, 1.02, 1.03) | non constant (0.75, 0.96, 0.96) |
| vision | coreml-ne contre coreml-gpu | plus bas dans les 3 répétitions (0.84, 0.90, 0.87) | non constant (1.04, 1.01, 1.01) | non constant (0.94, 0.97, 0.99) |
| vision | ne-complet contre torch | non constant (0.95, 0.94, 0.95) | non constant (1.02, 1.02, 1.03) | non constant (0.76, 0.96, 0.96) |
| vision | ne-complet contre coreml-gpu | non constant (0.94, 0.99, 0.97) | non constant (1.03, 1.02, 1.01) | non constant (0.95, 0.97, 0.98) |
| vision30 | coreml-ne contre torch | non constant (0.95, 0.95, 0.95) | non constant (1.00, 1.00, 1.00) | plus haut dans les 3 répétitions (1.47, 1.54, 1.55) |
| vision30 | coreml-ne contre coreml-gpu | non constant (0.95, 0.96, 0.96) | non constant (1.00, 1.00, 1.00) | plus haut dans les 3 répétitions (1.45, 1.49, 1.68) |
| vision30 | ne-complet contre torch | non constant (0.99, 0.97, 1.00) | non constant (1.00, 1.00, 1.00) | plus haut dans les 3 répétitions (1.56, 1.63, 1.62) |
| vision30 | ne-complet contre coreml-gpu | non constant (0.99, 0.99, 1.00) | non constant (1.00, 1.00, 1.00) | plus haut dans les 3 répétitions (1.54, 1.57, 1.76) |

Ralentissement sous charge (temps avec charge / sans charge, même répétition) : vision torch 1.09 (1.08 à 1.09) ; vision coreml-gpu 1.07 (1.04 à 1.09) ; vision coreml-ne 0.93 (0.92 à 0.94) ; vision ne-complet 1.03 (1.03 à 1.04) ; vision30 torch 1.02 (1.02 à 1.03) ; vision30 coreml-gpu 1.02 (1.01 à 1.02) ; vision30 coreml-ne 0.98 (0.96 à 0.98) ; vision30 ne-complet 1.01 (1.00 à 1.02)

30 images par seconde gardées (au moins 29,7 images/s et au plus 0,3 échéance manquée par seconde, dans les 3 répétitions) : torch oui, coreml-gpu oui, coreml-ne oui, ne-complet oui

Réponse pré-enregistrée à la question 2, ne-complet contre torch (principal ; précision différente : torch en float32, ne-complet en float16) : **non** (ralentissement : non constant (0.95, 0.94, 0.95) ; part de la vision : non constant (1.02, 1.02, 1.03) ; 30 images/s gardées : ne-complet oui, torch oui).

Réponse pré-enregistrée à la question 2, ne-complet contre coreml-gpu (principal, même précision float16) : **non** (ralentissement : non constant (0.94, 0.99, 0.97) ; part de la vision : non constant (1.03, 1.02, 1.01) ; 30 images/s gardées : ne-complet oui, coreml-gpu oui).

Les autres contrastes de la question 2 sont secondaires. Le contraste principal a été choisi après l'essai de bout en bout (10 épisodes, EXP-011 en même temps) : ce qu'il montrait est consigné dans l'amendement 6.

Question 3 : par construction, encodeurs et prédicteur en Core ML sur le Neural Engine (plan de calcul), CEM et coût sur le CPU (--device cpu) ; puissance du GPU au-dessus du repos pendant la planification de ne-complet : 0.03 W, 0.00 W, -0.00 W (GPU au repos de 0.14 à 0.17 W selon la mesure au repos) ; « planificateur entièrement hors du GPU » permis (au plus 0,3 W dans les 3 répétitions).

Sensibilité à l'attente active d'OpenMP (ne-complet, 2 runs par variante) : par défaut 649 (647 à 652) ms, 8.39 (8.28 à 8.49) J, CPU 2.5 (2.4 à 2.5) W ; OMP_WAIT_POLICY=PASSIVE 649 (648 à 651) ms, 8.48 (8.48 à 8.49) J, CPU 2.5 (2.5 à 2.5) W ; rapport d'énergie 1.01, de temps 1.00.

Sensibilité à l'attente active d'OpenMP (coreml-cpu, 2 runs par variante) : par défaut 1757 (1753 à 1761) ms, 34.53 (34.38 à 34.67) J, CPU 15.1 (15.1 à 15.1) W ; OMP_WAIT_POLICY=PASSIVE 1757 (1755 à 1758) ms, 34.53 (34.51 à 34.54) J, CPU 15.1 (15.1 à 15.1) W ; rapport d'énergie 1.00, de temps 1.00.

Dérive : runs torch r1, r2, r3 et final : 774 ms, 767 ms, 768 ms, 769 ms par plan ; écart des 3 répétitions : temps 1.0 %, énergie 3.6 %.

Contrôle descriptif de réussite du planificateur sur le Neural Engine (500 épisodes d'EXP-003, 2 graines) : 83.7 % contre 83.5 % en pleine précision PyTorch, écart +0.2 [-1.5, +1.9] points (tirages de la CEM différents : sur le CPU ; aucune affirmation de non-infériorité).

## Balayage de la taille du lot (appels isolés du prédicteur, caches chauds, ms, médiane des répétitions)

| Processeur | Codage | lot 1 | lot 8 | lot 32 | lot 100 | lot 300 |
| --- | --- | --- | --- | --- | --- | --- |
| ne | fp16 | 0.37 | 0.36 | 0.50 | 1.35 | 3.71 |
| ne | int8 | 0.34 | 0.33 | 0.52 | 1.36 | 3.62 |
| ne | int4b32 | 1.48 | 1.00 | 1.90 | 4.34 | 12.05 |
| ne | pal4 | 0.35 | 0.33 | 0.52 | 1.35 | 3.62 |
| ne | int4c | 0.34 | 0.34 | 0.51 | 1.35 | 3.60 |
| gpu | fp16 | 0.68 | 1.08 | 1.11 | 1.26 | 1.58 |
| gpu | int8 | 1.07 | 1.24 | 2.18 | 5.19 | 14.13 |
| gpu | int4b32 | 1.06 | 1.34 | 2.59 | 6.53 | 18.16 |
| gpu | pal4 | 1.46 | 2.32 | 2.13 | 2.82 | 1.90 |
| gpu | int4c | 1.08 | 1.33 | 2.56 | 6.41 | 17.82 |

Prédiction enregistrée (au lot de 300, aucun codage plus rapide que fp16 de plus de 5 % dans les 3 répétitions) : tenue ; ne int8 non constant (0.98, 0.98, 0.97) ; ne int4b32 plus haut dans les 3 répétitions (3.25, 3.25, 3.24) ; ne pal4 non constant (0.97, 0.98, 0.98) ; ne int4c non constant (0.97, 0.97, 0.97) ; gpu int8 plus haut dans les 3 répétitions (8.96, 8.93, 8.81) ; gpu int4b32 plus haut dans les 3 répétitions (11.52, 11.49, 11.32) ; gpu pal4 plus haut dans les 3 répétitions (1.20, 1.20, 1.19) ; gpu int4c plus haut dans les 3 répétitions (11.30, 11.27, 11.11). Des temps par appel au lot de 300 avaient été vus avant le balayage (essai de placement) : c'est une observation, pas une prédiction confirmée. Cases informatives sur la question « un codage accélère-t-il ? » : Neural Engine int8, pal4 et int4c, GPU pal4 ; les autres cases mesurent aussi le coût propre de l'implémentation du codage dans Core ML, qu'on ne sépare pas ici.
