# EXP-011 : analyse pré-enregistrée (le prédicteur ternaire, RTN et GPTQ à échelle réduite)

Contrôles passés, commit eddfdb5 ; reproductions des graines 42 et 46 identiques.

| Tâche | fp32 | RTN réduit | GPTQ réduit | GPTQ moins fp32 | RTN moins fp32 | GPTQ moins RTN | RTN 3 bits (échelle du max) | GPTQ ternaire moins RTN 3 bits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pusht | 83.5 % | 40.4 % | 53.0 % | -30.5 [-34.3, -26.8] | -43.1 [-46.9, -39.2] | +12.6 [+8.7, +16.6] | 81.3 % | -28.3 [-32.2, -24.5] |
| tworoom | 84.8 % | 84.5 % | 81.4 % | -3.4 [-5.0, -1.9] | -0.3 [-2.0, +1.4] | -3.2 [-4.8, -1.5] | 83.8 % | -2.4 [-4.0, -0.8] |

| Test | Tâche | Hypothèse | Écart (points) | Borne inférieure au niveau de Holm | p signe | p bootstrap-t | p Holm | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G1 | pusht | GPTQ plus haut que RTN | +12.6 [+8.7, +16.6] | 8.1 (niveau 0.0125) | 0.0001 | sans objet | 0.0004 | soutenue |
| G2 | tworoom | GPTQ plus haut que RTN | -3.2 [-4.8, -1.5] | -4.8 (niveau 0.0250) | 1.0000 | sans objet | 1.0000 | non soutenue |
| N1 | pusht | GPTQ perd moins de 2 points | -30.5 [-34.3, -26.8] | -33.6 (niveau 0.0500) | 1.0000 | 1.0000 | 1.0000 | non soutenue |
| N2 | tworoom | GPTQ perd moins de 2 points | -3.4 [-5.0, -1.9] | -5.1 (niveau 0.0167) | 0.9667 | 0.9749 | 1.0000 | non soutenue |

Contrôle de la marge (deux séries de la pleine précision, graines 47 à 51 contre 42 à 46, écart vrai nul, test N) :
- pusht : écart +0.2 [-1.1, +1.4] points, p signe 0.0006, p bootstrap-t 0.0004
- tworoom : écart -0.9 [-2.0, +0.1] points, p signe 0.0228, p bootstrap-t 0.0249

Sensibilité, référence sur les 10 graines de la pleine précision :
- N1 : écart -30.6 [-34.2, -26.9] points, p 1.0000
- N2 : écart -3.0 [-4.4, -1.6] points, p 0.9240

Part de la perte de RTN rattrapée par GPTQ (en second : elle dépend de la faiblesse de RTN) :
- pusht : +29.2 [+20.9, +37.0] %
- tworoom : sans objet
