# EXP-015 : rapport de l'analyse pré-enregistrée

Commit des runs : `78d2523`.

Contrôles : reproductions de la pleine précision identiques (bloquant) ; références réutilisées : ne16 exp012-ne-pusht, int4c exp014-int4c-pusht, predictor:4 EXP-003 ; bras int4c-sp lancé.

| Test | Bras contre référence | Écart (points) [IC 95 %] | p Holm | Borne au niveau de Holm | Verdict |
| --- | --- | --- | --- | --- | --- |
| L1 (perte) | pp contre ne16 | -8.6 [-11.1 ; -6.1] | 0.0005 | -5.77 | soutenue |
| L2 (non_inferiorite) | bl contre ne16 | +0.5 [-0.8 ; +1.9] | 0.0009 | -0.96 | soutenue |
| L3 (perte) | s_pp contre fp32 | -8.8 [-11.3 ; -6.3] | 0.0005 | -6.02 | soutenue |
| L4 (pas_mieux) | pp contre int4c | +0.3 [-1.2 ; +2.0] | 0.0203 | +1.72 | soutenue |
| L5 (non_inferiorite) | sp contre ne16 | +0.3 [-1.0 ; +1.6] | 0.0009 | -0.95 | soutenue |

Lecture pré-enregistrée : la perte d'int4c vient de pred_proj (L1, L2 et L4) : soutenue ; coder tout sauf pred_proj : appuyé sur L5 ; perte établie (intervalle sous 0) : bl non, pp oui, sp non ; D3 exclut 0 : pred_proj non, blocs non, tout non.

Étape 0 b, règle R1 : int4c H2 écartée, int4c-bl H2 écartée, int4c-pp H2 écartée, int4c-sp H2 écartée, pal4 H2 écartée.

D1, écarts :

- bl - ne16 : +0.5 [-0.8 ; +1.9]
- pp - ne16 : -8.6 [-11.1 ; -6.1]
- sp - ne16 : +0.3 [-1.0 ; +1.6]
- int4c - ne16 : -9.0 [-11.5 ; -6.4]
- bl - fp32 : +0.8 [-0.6 ; +2.2]
- pp - fp32 : -8.3 [-10.8 ; -5.8]
- sp - fp32 : +0.6 [-0.7 ; +2.0]
- int4c - fp32 : -8.6 [-11.2 ; -6.2]
- ne16 - fp32 : +0.3 [-0.8 ; +1.4]
- s_pp - fp32 : -8.8 [-11.3 ; -6.3]
- s_pr - fp32 : -9.3 [-11.9 ; -6.7]
- predictor4 - fp32 : +0.6 [-0.7 ; +1.9]

D2, interaction :

- puce : -0.8 [-2.9 ; +1.2]
- simulé : -1.1 [-3.1 ; +0.8]
- puce, plan complet avec int4c-sp : -0.6 [-2.6 ; +1.3]

D3, puce contre simulation (différence de différences) :

- pred_proj : +0.2 [-1.4 ; +1.8]
- blocs : -0.2 [-1.7 ; +1.4]
- tout : +0.3 [-1.2 ; +1.8]

D4, part de la perte d'int4c portée par pred_proj : 0.96 [0.79 ; 1.16]

D5, réussite : fp32 83.5 %, predictor4 84.2 %, ne16 83.8 %, int4c 74.9 %, bl 84.3 %, pp 75.2 %, sp 84.2 %, s_pp 74.7 %, s_pr 74.2 %

Fichier des poids (octets) : ne16 23232128, int4c 5945536, bl 7167488, pp 22057088, sp 7120576

Temps par plan médian (ms, descriptif ; ne16 et int4c lus dans exp012-ne-pusht, exp014-int4c-pusht, autres sessions) : ne16 833, int4c 822, bl 859, pp 850, sp 832 ; ancres refaites dans cette session, graine 42 : ne16 874, int4c 866
