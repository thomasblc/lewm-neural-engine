# EXP-014 : analyse pré-enregistrée (codages des poids du prédicteur dans Core ML, sur le Neural Engine)

Contrôles passés, commit cc9cd5d ; reproductions PyTorch des graines 42 et 46 identiques.

| Test | Tâche | Codage | Écart à PyTorch float32 (points, IC 95 %) | Borne inférieure au niveau de Holm | p signe | p bootstrap-t | p Holm | Perte de moins de 2 points |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N1 | pusht | int8 | +0.6 [-0.4, +1.6] | -0.5 (niveau 0.0083) | 0.0001 | 0.0001 | 0.0006 | soutenue |
| N2 | pusht | int4c | -8.6 [-11.2, -6.2] | -11.2 (niveau 0.0250) | 1.0000 | 1.0000 | 1.0000 | non soutenue |
| N3 | pusht | pal4 | -2.4 [-4.4, -0.4] | -4.3 (niveau 0.0250) | 0.6424 | 0.6431 | 1.0000 | non soutenue |
| N4 | tworoom | int8 | +0.3 [-0.6, +1.1] | -0.7 (niveau 0.0100) | 0.0001 | 0.0001 | 0.0006 | soutenue |
| N5 | tworoom | int4c | -0.6 [-1.6, +0.4] | -1.6 (niveau 0.0167) | 0.0024 | 0.0018 | 0.0072 | soutenue |
| N6 | tworoom | pal4 | +0.4 [-0.5, +1.3] | -0.7 (niveau 0.0125) | 0.0001 | 0.0001 | 0.0006 | soutenue |
- « Non soutenue » : la non-infériorité n'est pas établie, ce n'est pas une perte ; perte établie seulement si l'intervalle est entièrement sous 0 : N2, N3.
- Contrôle de la marge au niveau de Holm de chaque test (deux séries de la pleine précision, écart vrai nul) : N1 passe, N2 passe, N3 passe, N4 ne passe pas, N5 ne passe pas, N6 ne passe pas.
- Erreur de Monte-Carlo des p : N1 0.0001 (signe), 0.0001 (bootstrap-t) ; N2 0.0000 (signe), 0.0000 (bootstrap-t) ; N3 0.0048 (signe), 0.0048 (bootstrap-t) ; N4 0.0001 (signe), 0.0001 (bootstrap-t) ; N5 0.0005 (signe), 0.0004 (bootstrap-t) ; N6 0.0001 (signe), 0.0001 (bootstrap-t)

Contrôle de la marge (deux séries de la pleine précision, graines 47 à 51 contre 42 à 46, écart vrai nul) :
- pusht : écart +0.2 [-1.1, +1.4] points, p signe 0.0006, p bootstrap-t 0.0004
- tworoom : écart -0.9 [-2.0, +0.1] points, p signe 0.0228, p bootstrap-t 0.0249

Sensibilité, référence sur les 10 graines de la pleine précision :
- N1 : écart +0.5 [-0.4, +1.4] points, p 0.0001
- N2 : écart -8.7 [-11.2, -6.3] points, p 1.0000
- N3 : écart -2.4 [-4.4, -0.5] points, p 0.6850
- N4 : écart +0.7 [+0.0, +1.5] points, p 0.0001
- N5 : écart -0.1 [-1.0, +0.8] points, p 0.0001
- N6 : écart +0.9 [+0.1, +1.7] points, p 0.0001

| Tâche | Version | Réussite | Effet du codage seul (contre Core ML 16 bits sur le Neural Engine, EXP-012 ; points) | Temps par plan (ms) | Fichier des poids (Mo) |
| --- | --- | --- | --- | --- | --- |
| pusht | PyTorch float32 | 83.5 % | | | |
| pusht | Core ML 16 bits, Neural Engine (EXP-012) | 83.8 % | | | 23.2 |
| pusht | int8 | 84.1 % | +0.3 [-0.8, +1.4] | 828 | 11.7 |
| pusht | int4c | 74.9 % | -9.0 [-11.5, -6.4] | 822 | 5.9 |
| pusht | pal4 | 81.2 % | -2.7 [-4.7, -0.8] | 828 | 5.9 |
| tworoom | PyTorch float32 | 84.8 % | | | |
| tworoom | Core ML 16 bits, Neural Engine (EXP-012) | 84.4 % | | | 23.2 |
| tworoom | int8 | 85.1 % | +0.6 [-0.2, +1.5] | 829 | 11.7 |
| tworoom | int4c | 84.2 % | -0.2 [-1.2, +0.8] | 828 | 5.9 |
| tworoom | pal4 | 85.2 % | +0.8 [-0.2, +1.7] | 831 | 5.9 |

- pusht : témoin 16 bits refait dans EXP-014 (graine 42) identique octet pour octet à EXP-012 : « effet du codage seul » au sens strict.
  - int8 : 34 constantes codées : 30 dans les blocs du prédicteur, 2 de pred_proj, 2 matrices 0/1 de la réécriture « 1ctb », 0 autres ()
  - int4c : 34 constantes codées : 30 dans les blocs du prédicteur, 2 de pred_proj, 2 matrices 0/1 de la réécriture « 1ctb », 0 autres ()
  - pal4 : 34 constantes codées : 30 dans les blocs du prédicteur, 2 de pred_proj, 2 matrices 0/1 de la réécriture « 1ctb », 0 autres ()
- tworoom : témoin 16 bits refait dans EXP-014 (graine 42) identique octet pour octet à EXP-012 : « effet du codage seul » au sens strict.
  - int8 : 34 constantes codées : 30 dans les blocs du prédicteur, 2 de pred_proj, 2 matrices 0/1 de la réécriture « 1ctb », 0 autres ()
  - int4c : 34 constantes codées : 30 dans les blocs du prédicteur, 2 de pred_proj, 2 matrices 0/1 de la réécriture « 1ctb », 0 autres ()
  - pal4 : 34 constantes codées : 30 dans les blocs du prédicteur, 2 de pred_proj, 2 matrices 0/1 de la réécriture « 1ctb », 0 autres ()

Temps par plan sans mise en route et paquets Core ML dans le dossier des résultats : ne se compare pas aux temps d'EXP-013.
