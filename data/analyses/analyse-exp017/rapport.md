# EXP-017 : rapport de l'analyse pré-enregistrée (codes de GPTQ dans Core ML, Neural Engine, PushT)

Commit des runs : `8d24631` ; ne16 exp012-ne-pusht, int4c exp014-int4c-pusht ; étape 0 : 0 code différent sur 11550720.

| Test | Bras contre référence | Écart (points) [IC 95 %] | Borne inférieure au niveau de Holm | p Holm | Verdict |
| --- | --- | --- | --- | --- | --- |
| G1 (non_inferiorite) | g contre ne16 | -0.7 [-2.2 ; +0.9] | -2.04 | 0.0552 | non soutenue |
| G2 (superiorite) | g contre int4c | +8.3 [+5.6 ; +11.0] | +5.62 | 0.0002 | soutenue |

Lecture (amendement 1) : fait mieux que l'arrondi au plus proche ; non-infériorité au 16 bits non établie.

Descriptifs :

- g - fp32 : -0.4 [-2.0 ; +1.3]
- gs - fp32 : -1.2 [-2.9 ; +0.4]
- gs - s_pr : +8.0 [+5.4 ; +10.8]
- puce contre simulation : +0.6 [-1.0 ; +2.1]
- g - sp : -1.0 [-2.6 ; +0.6]
- g - ne16 : -0.7 [-2.2 ; +0.9]
- g - int4c : +8.3 [+5.6 ; +11.0]

Réussite : fp32 83.5 %, sp 84.2 %, s_pr 74.2 %, ne16 83.8 %, int4c 74.9 %, g 83.2 %, gs 82.3 %
Fichier des poids (octets) : ne16 23232128, int4c 5945536, sp 7120576, g 5945536
Temps par plan (ms, descriptif, sans comparaison affirmée) : G 836 ; ancres de la session, graine 42 : ne16 851, int4c 832
Puce contre simulation : compatible ; G - int4c-sp : descriptif
Part de la perte d'int4c rattrapée : 0.92 [0.75 ; 1.11]
Étape 0, sortie de la puce contre la simulation, rapport au repère 16 bits par lot : 1.005, 1.016, 1.002
