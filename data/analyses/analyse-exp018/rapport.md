# EXP-018 : rapport de l'analyse pré-enregistrée (2 000 épisodes neufs pour la puce, PushT)

Commit des runs : `8635a12` ; modèles identiques à ceux d'EXP-014, EXP-015 et EXP-017.

| Test | Bras contre référence | Écart (points) [IC 95 %] | Borne au niveau de Holm (supérieure pour T1) | p Holm | Verdict |
| --- | --- | --- | --- | --- | --- |
| T1 (perte) | int4c contre fp32 | -7.2 [-8.4 ; -6.1] | -5.97 | 0.0004 | soutenue |
| T2 (non_inferiorite) | sp contre fp32 | -0.3 [-0.9 ; +0.3] | -0.95 | 0.0004 | soutenue |
| T3 (non_inferiorite) | g contre fp32 | -1.4 [-2.2 ; -0.6] | -2.07 | 0.0688 | non soutenue |
| T4 (superiorite) | g contre int4c | +5.9 [+4.6 ; +7.1] | +4.64 | 0.0004 | soutenue |

Lecture : la perte d'int4 et sa réparation par pred_proj en 16 bits se retrouvent sur des épisodes neufs ; GPTQ : fait mieux que l'arrondi simple de plus de 2 points ; non-infériorité face à PyTorch float32 non établie.

Descriptifs (2 000 épisodes) :

- int4c - fp32 : -7.2 [-8.4 ; -6.1]
- sp - fp32 : -0.3 [-0.9 ; +0.3]
- g - fp32 : -1.4 [-2.2 ; -0.6]
- g - int4c : +5.9 [+4.6 ; +7.1]
- g - sp : -1.1 [-1.8 ; -0.3]
- part de la perte d'int4c rattrapée par g : 0.81 [0.70 ; 0.91]
- part de la perte d'int4c rattrapée par sp : 0.96 [0.88 ; 1.04]

Rappel sur les 500 épisodes d'EXP-003 (EXP-014, EXP-015, EXP-017) :

- int4c - fp32 : -8.6 [-11.2 ; -6.2]
- sp - fp32 : +0.6 [-0.7 ; +2.0]
- g - fp32 : -0.4 [-2.0 ; +1.3]
- g - int4c : +8.3 [+5.6 ; +11.0]
- g - sp : -1.0 [-2.6 ; +0.6]

Réussite : fp32 87.1 %, int4c 79.8 %, sp 86.8 %, g 85.7 %
Temps par plan (ms, descriptif) : int4c 824, sp 821, g 825
