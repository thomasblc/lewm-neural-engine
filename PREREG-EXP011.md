# Pré-enregistrement d'EXP-011 : le prédicteur ternaire (stocké sur 2 bits), RTN et GPTQ à échelle réduite

Rédigé par Claude le 1er octobre 2026 entre 11 h 20 et 11 h 28 (heure du Mac), avant tout calcul d'EXP-011 ;
amendement 1 vers 12 h (commit `fcbd573`, 11 h 56) ; amendement 2 rédigé entre 13 h 25 et 13 h 46 (commit `4053415`,
13 h 46) ; amendement 3 à 14 h 03 ; toujours avant tout calcul : le lanceur a été arrêté à 11 h 43 puis à 13 h 24
(`resultats/exp011.log`) sans avoir écrit de fichier d'évaluation. Heures tirées de l'historique git et des journaux. Seules ont été calculées, sur la calibration d'EXP-009 et sans
aucun épisode d'évaluation, des erreurs de reconstruction des couches. Figé par le commit enregistré dans les
fichiers ; les versions précédentes sont dans l'historique git (`d3f0dda`, `6c04c40`).

## Historique des amendements

- Amendement 1 (relecture du code) : heure de rédaction corrigée ; erreurs de reconstruction produites par
  `erreur_reconstruction.py` ; contrôles du protocole, de la calibration et du commit de la reproduction ; tests aux p
  intermédiaires.
- Amendement 2 (deux relectures scientifiques, statistique et protocole) :
  - Le RTN prévu à 2 bits, échelle du maximum de la ligne, met 79 à 81 % des poids du prédicteur à zéro : le
    comparer à GPTQ aurait été un combat truqué (l'échelle seule fait 91 à 93 % du gain de GPTQ en erreur de
    reconstruction, et hors échantillon sur PushT GPTQ à l'échelle du max fait moins bien qu'un RTN à échelle
    réduite, 0,099 contre 0,083). Les versions à l'échelle du max sont retirées ; RTN et GPTQ utilisent désormais la
    même échelle réduite, choisie sur la calibration : seul l'arrondi change, comme dans EXP-009.
  - Marge ramenée de 3 à 2 points, comme toutes les autres marges de l'article.
  - Non-infériorité décidée par deux tests concordants (inversion de signe et bootstrap-t).
  - Contrôle négatif, sensibilité à 10 graines de référence, reproduction finale, formulations exactes (voir plus bas).
- Amendement 3 (relecture de l'amendement 2) : heures corrigées (ci-dessus) ; la phrase sur tout le modèle à 2 bits
  s'appuie maintenant sur GPTQ à échelle réduite ; rapport à RTN 3 bits donné aussi hors échantillon ; le « contrôle
  négatif » est un contrôle de la marge (deux séries de la pleine précision, écart vrai nul), ses valeurs déjà connues
  sont écrites ci-dessous ; le RTN 3 bits de comparaison est nommé « échelle du max » ; puissance revue ; les
  reproductions finales d'un lancement précédent sont effacées au lancement.

## Motivation et question

Le prédicteur est la partie appelée à chaque pas de la CEM (150 appels par plan) et la partie robuste : à 3 bits en
RTN, il ne coûte que 2,2 points sur PushT (EXP-003, intervalle 0,2 à 4,3) et 1,0 sur TwoRoom (EXP-008). Question : avec
un format ternaire par canal (trois niveaux -s, 0, +s par ligne, log2 3 = 1,58 bit d'information, stocké sur 2 bits),
le prédicteur garde-t-il la réussite, et le choix des arrondis par GPTQ compte-t-il, à grille et échelles identiques ?

Portée : quantification simulée des poids seuls, aucune mesure de vitesse ici. Gain en mémoire du prédicteur (10,76 M
de poids) : 21,5 Mo en 16 bits, 4,04 Mo à 3 bits, 2,69 Mo à 2 bits (plus 0,06 bit par poids pour les échelles en 16
bits). Ce format n'est pas le « 2 bits » usuel de la littérature (quatre niveaux, souvent asymétrique et par groupes) :
on ne compare pas nos chiffres aux leurs.

Erreur de reconstruction du prédicteur, rapportée à celle de tout mettre à zéro (`erreur_reconstruction.py`,
`verifier_gptq.py --echelle reduite`, sorties dans `resultats/exp011-preparation/`) :

| Tâche | RTN 3 bits, échelle du max | RTN ternaire, échelle du max | RTN ternaire, échelle réduite | GPTQ ternaire, échelle réduite |
| --- | --- | --- | --- | --- |
| PushT | 0,023 | 0,248 | 0,083 | 0,032 |
| TwoRoom | 0,021 | 0,267 | 0,078 | 0,032 |

GPTQ à échelle réduite contre RTN à échelle réduite : 0,392 (PushT) et 0,415 (TwoRoom) sur la calibration, 0,504 et
0,427 hors échantillon (échelle et GPTQ réglés sur une moitié, erreur mesurée sur l'autre). Tout le modèle n'est pas
testé : à 2 bits, même avec GPTQ à échelle réduite, son erreur (0,030 sur PushT, 0,003 sur TwoRoom ; hors échantillon
0,038 et 0,003) reste au-dessus de celle de RTN à 3 bits sur tout le modèle (0,025 et 0,002), qui perd déjà 47,8 points
(EXP-003) et 46,2 points (EXP-006) ; `resultats/exp011-preparation/erreur-reconstruction-tout-le-modele.txt`. C'est une
règle de décision, pas un résultat : l'agrégat est dominé par quelques couches et ne prédit pas la réussite.

## Plan

- Grille : symétrique par canal, qmax = 1. Échelle réduite : pour chaque couche, un facteur f de 0,20 à 1,00 par pas de
  0,05 multiplie l'échelle du maximum de chaque ligne ; f minimise l'erreur de reconstruction de RTN sur la
  calibration (`gptq.facteur_reduit`) ; RTN et GPTQ utilisent les mêmes échelles (mémoire identique).
- GPTQ : `gptq.py` (amortissement 1 %, blocs de 128, ordre naturel), statistiques H calculées sur le modèle en pleine
  précision pour toutes les couches (variante non séquentielle, comme EXP-009 ; le GPTQ d'origine quantifie couche
  après couche). Calibration : celle d'EXP-009 de chaque tâche, réutilisée sans changement (fichier et SHA-256 dans le
  protocole), disjointe des évaluations.
- Versions, par tâche : `predictor:2:clip` (RTN) et `predictor:2:clip:gptq` (GPTQ), 30 couches du prédicteur ;
  graines 42 à 46 ; PushT sur les 500 épisodes d'EXP-003, TwoRoom sur les 500 d'EXP-006 ; pleine précision appariée :
  EXP-003 et EXP-006. 20 fichiers et 4 reproductions, environ 3 h 30. Lancée après EXP-012.

## Tests (4 tests, Holm, unilatéraux)

K(e, v) : nombre de succès de l'épisode e sur les 5 graines.

- G1 (PushT), G2 (TwoRoom) : GPTQ fait mieux que RTN, à échelles identiques. x(e) = K(e, clip:gptq) - K(e, clip) ;
  inversion de signe comme EXP-003 (nulle exacte), sens « plus ».
- N1 (PushT), N2 (TwoRoom) : le prédicteur ternaire avec GPTQ perd moins de 2 points de réussite. x(e) =
  K(e, clip:gptq) - K(e, fp32) ; statistique décalée y = 20 x + 2 ; décision seulement si la p par inversion de signe
  et la p d'un bootstrap-t (épisodes rééchantillonnés, B = 10 000, `default_rng(4)`) sont toutes deux sous le seuil
  (le test par inversion de signe de la statistique décalée n'est pas exact et penche vers « non inférieur »).

Holm sur les 4 (p de G, plus grande des deux p de N), seuil 0,05. Puissance attendue de N à perte nulle (écarts-types
des expériences précédentes, approximation normale, niveaux de Holm 0,0125 à 0,05) : 0,4 à 0,85 sur PushT, 0,85 à 0,95
sur TwoRoom. L'erreur de reconstruction du prédicteur ternaire avec GPTQ (1,4 à 1,5 fois celle de RTN 3 bits à
l'échelle du max sur la calibration, 1,6 à 1,8 fois hors échantillon) laisse attendre une perte de quelques points sur
PushT : N1 sera
probablement « non soutenue », ce qui veut dire « une perte de moins de 2 points n'est pas montrée ». On rapporte
d'abord l'écart estimé et son intervalle, et la borne inférieure unilatérale au niveau de Holm de chaque test.

Descriptifs (bootstrap `default_rng(2)`) : réussites ; GPTQ moins RTN ; GPTQ moins pleine précision ; GPTQ ternaire
moins RTN à 3 bits du prédicteur, échelle du max (EXP-003 sur PushT, EXP-008 sur TwoRoom ; ce RTN n'a pas l'échelle
réglée, la comparaison avantage le format ternaire et se lit comme telle) ; part de la perte de RTN rattrapée (en
second, elle dépend de la faiblesse de RTN). Sensibilité : référence sur la moyenne des 10 graines.

Contrôle de la marge (aucune décision attachée) : la deuxième série de la pleine précision (graines 47 à 51) contre la
première, par le même test N ; l'écart vrai est nul, le test doit conclure. Valeurs déjà connues avant l'expérience
(calculées le 1er octobre sur EXP-003 et EXP-006) : PushT, écart +0,16 point, p 0,0006 (signe) et 0,0004
(bootstrap-t) ; TwoRoom, écart -0,92 point, p 0,0228 et 0,0249. Sur TwoRoom, deux séries de 5 graines de la même
pleine précision diffèrent donc presque d'un point, et ne passeraient pas aux niveaux de Holm sous 0,025 : la marge de
2 points y est serrée face au seul bruit des graines. Un N2 « non soutenu » avec un écart estimé au-dessus de -1 point
se lit comme « non montré », pas comme une perte.

## Contrôles bloquants

Avant la grille, par tâche : reproduction de la pleine précision graine 42 (identique octet pour octet à EXP-003 ou
EXP-006, et référence de la sentinelle) ; erreur de reconstruction de GPTQ sous celle de RTN, à échelle réduite, sur la
calibration et hors échantillon (`verifier_gptq.py --bits 2 --groupe predictor --echelle reduite`). Après la grille :
reproduction de la pleine précision graine 46 (rien n'a dérivé). À l'analyse : tous les fichiers ; protocole identique
à celui de la référence, calibration d'EXP-009 exceptée ; calibration d'EXP-009 conforme et disjointe des évaluations ;
mêmes épisodes, départs et buts que la référence ; 30 couches quantifiées ; contrôles de course ; un seul commit dont le
code est celui du dépôt ; arbre propre ; reproductions du même commit. La sentinelle contrôle chaque fichier ; une
modification du code après une ANOMALIE annule les fichiers et tout est recalculé. Le journal n'est pas lu avant
l'analyse.

## Limites à écrire

Un modèle publié par tâche, sans variance d'entraînement ; deux tâches simulées en 2D ; une machine, un système ;
quantification simulée des poids seuls, format ternaire, GPTQ non séquentiel, une seule calibration ; comparaison au
RTN 3 bits à l'échelle du max (non réglée) ; marge serrée sur TwoRoom face au bruit des graines ; les 500 épisodes
de chaque tâche servent dans plusieurs expériences (pas de jeu neuf ici) ; formulation : « perte de moins de 2 points,
test unilatéral », jamais « aucun changement » ; « identique octet pour octet sur la même machine et la même pile
logicielle », pas « reproductible » en général.
