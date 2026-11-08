# EXP-012 : analyse pré-enregistrée (le prédicteur en Core ML 16 bits, GPU et Neural Engine)

Contrôles passés, commit 4e9d667 ; reproductions PyTorch des graines 42, 43 et 46 identiques.

| Test | Tâche | Moteur | Écart (points, IC 95 %) | Borne inférieure au niveau de Holm | p signe | p bootstrap-t | p Holm | Perte de moins de 2 points |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N1 | pusht | coreml-ne | +0.3 [-0.8, +1.4] | -0.9 (niveau 0.0125) | 0.0001 | 0.0001 | 0.0004 | soutenue |
| N2 | tworoom | coreml-ne | -0.4 [-1.2, +0.5] | -1.3 (niveau 0.0167) | 0.0001 | 0.0001 | 0.0004 | soutenue |
| N3 | pusht | coreml-gpu | -0.5 [-1.4, +0.4] | -1.3 (niveau 0.0500) | 0.0005 | 0.0010 | 0.0010 | soutenue |
| N4 | tworoom | coreml-gpu | -0.1 [-0.9, +0.7] | -0.9 (niveau 0.0250) | 0.0001 | 0.0001 | 0.0004 | soutenue |
- Erreur de Monte-Carlo des p : N1 0.0001 (signe), 0.0001 (bootstrap-t) ; N2 0.0001 (signe), 0.0001 (bootstrap-t) ; N3 0.0002 (signe), 0.0003 (bootstrap-t) ; N4 0.0001 (signe), 0.0001 (bootstrap-t)

Contrôle de la marge (deux séries de la pleine précision, graines 47 à 51 contre 42 à 46, écart vrai nul) :
- pusht : écart +0.2 [-1.1, +1.4] points, p signe 0.0006, p bootstrap-t 0.0004
- tworoom : écart -0.9 [-2.0, +0.1] points, p signe 0.0228, p bootstrap-t 0.0249

Sensibilité, référence sur les 10 graines de la pleine précision :
- N1 : écart +0.2 [-0.8, +1.3] points, p 0.0002
- N2 : écart +0.1 [-0.7, +0.9] points, p 0.0001
- N3 : écart -0.6 [-1.5, +0.3] points, p 0.0014
- N4 : écart +0.4 [-0.4, +1.2] points, p 0.0001

| Tâche | Moteur | Réussite | Temps par plan (ms, médiane des fichiers, min à max) | Puissance (W) | Repos (W) | Énergie par plan au-dessus du repos (J) |
| --- | --- | --- | --- | --- | --- | --- |
| pusht | PyTorch float32 (réussite : 5 graines ; temps : graine 43) | 83.5 % | 798 (798 à 798) | 76.0 | 84.0 | -6.31 |
| pusht | Core ML GPU | 83.0 % | 386 (385 à 418) | 67.2 | 84.0 | -6.45 |
| pusht | Core ML Neural Engine | 83.8 % | 833 (826 à 839) | 89.1 | 84.0 | 4.25 |
| pusht | Neural Engine moins GPU | +0.8 [-0.2, +1.8] points | | | | |
| tworoom | PyTorch float32 (réussite : 5 graines ; temps : graine 43) | 84.8 % | 836 (836 à 836) | 74.2 | 83.0 | -7.35 |
| tworoom | Core ML GPU | 84.7 % | 401 (395 à 426) | 63.2 | 83.0 | -8.04 |
| tworoom | Core ML Neural Engine | 84.4 % | 839 (833 à 849) | 63.8 | 83.0 | -16.25 |
| tworoom | Neural Engine moins GPU | -0.3 [-1.3, +0.7] points | | | | |
