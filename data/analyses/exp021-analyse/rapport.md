# EXP-021 : rapport de l'analyse pré-enregistrée (1000 positions d'EXP-005, PushT, graines 52, 53, 54, 55)

Commit des runs : `747f668`. Verdicts conditionnels aux quatre réglages (un par graine).

| Test | Bras contre référence | Écart moyen (points) | Borne inférieure du test (niveau de Holm) | Écart par réglage | p Holm | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| H1 (superiorite) | A contre C | +16.8 | +14.42 | +17.0, +16.3, +17.3, +16.4 | 0.0003 | soutenue |
| H2 (superiorite) | A contre D | +2.4 | +0.90 | +1.6, +1.2, +4.4, +2.4 | 0.6027 | non soutenue |
| H3 (non_inferiorite) | A contre B | -2.5 | -3.94 | -3.8, -2.4, -0.3, -3.5 | 0.7585 | non soutenue |

Lecture :

- H1 : pour ces quatre réglages, régler les échelles, les valeurs gardées et les biais sur des entrées enregistrées en planification améliore le prédicteur ternaire avec 1 % gardé de plus de 2 points ; l'écart estimé dépasse +2 points dans chacun des 4 réglages
- H2 : apport de plus de 2 points des poids gardés non établi (écart +2.4 [+0.9 ; +3.9] points, descriptif)
- H3 : non-infériorité face au 3 bits réglé non établie (écart -2.5 [-3.9 ; -1.1] points, descriptif)
- pleine_precision : écart de A à la pleine précision -3.4 [-4.8 ; -1.9] points (descriptif, aucun test)
- memoire_voisine : E moins B, mémoire voisine (3,11 contre 3,06 bits) : -2.8 [-4.2 ; -1.3] points, non testé

| Bras | Bits par poids des couches linéaires du prédicteur | Réussite |
| --- | --- | --- |
| pleine précision | 32 | 86.9 % |
| A | 2.44 | 83.6 % |
| B | 3.06 | 86.1 % |
| B0 | 3.06 | 85.9 % |
| C | 2.44 | 66.8 % |
| D | 2.06 | 81.2 % |
| E | 3.11 | 83.3 % |

Écarts descriptifs (points, intervalle percentile à 95 %, non utilisé par les tests) :

- A - fp32 : -3.4 [-4.8 ; -1.9]
- B - fp32 : -0.9 [-1.8 ; +0.1]
- B0 - fp32 : -1.1 [-2.1 ; -0.1]
- C - fp32 : -20.1 [-22.4 ; -17.8]
- D - fp32 : -5.8 [-7.4 ; -4.2]
- E - fp32 : -3.6 [-5.0 ; -2.2]
- A - B : -2.5 [-3.9 ; -1.1]
- A - C : +16.8 [+14.5 ; +18.9]
- A - D : +2.4 [+0.9 ; +3.9]
- B0 - B : -0.2 [-1.2 ; +0.7]
- E - B : -2.8 [-4.2 ; -1.3]
- E - A : -0.2 [-1.7 ; +1.2]

Pertes de contrôle du réglage (avant, après) : A_g52 6.20e-03 -> 2.03e-04 ; A_g53 6.11e-03 -> 1.93e-04 ; A_g54 6.20e-03 -> 1.86e-04 ; A_g55 6.05e-03 -> 1.80e-04 ; B_g52 5.92e-04 -> 6.20e-05 ; B_g53 5.68e-04 -> 6.11e-05 ; B_g54 5.74e-04 -> 6.07e-05 ; B_g55 5.62e-04 -> 6.02e-05 ; D_g52 9.58e-03 -> 4.54e-04 ; D_g53 9.22e-03 -> 4.40e-04 ; D_g54 9.71e-03 -> 4.24e-04 ; D_g55 9.31e-03 -> 4.31e-04 ; E_g52 4.29e-03 -> 1.19e-04 ; E_g53 4.12e-03 -> 1.10e-04 ; E_g54 4.28e-03 -> 1.10e-04 ; E_g55 4.16e-03 -> 1.09e-04
