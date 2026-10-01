# Pré-enregistrement d'EXP-011 : le prédicteur à 2 bits avec GPTQ, sur PushT et TwoRoom

Rédigé par Claude le 1er octobre 2026 vers 11 h 45, avant tout calcul d'EXP-011 (seules ont été calculées, sur la
calibration d'EXP-009 et sans aucun épisode d'évaluation, des erreurs de reconstruction des couches : voir
Motivation). Figé par le commit enregistré dans les fichiers.

## Motivation et question

Le prédicteur est la partie appelée à chaque pas de la CEM (150 appels par plan, contre un seul pour l'encodeur) :
c'est lui qu'il faut compresser pour décider plus vite. Il est aussi la partie robuste : à 3 bits en RTN, il ne
coûte que 2,2 points sur PushT (EXP-003, intervalle 0,2 à 4,3). Sur la calibration d'EXP-009, l'erreur de
reconstruction du prédicteur (rapportée à celle de tout mettre à zéro) vaut, sur PushT, 0,023 en RTN à 3 bits, 0,248
en RTN à 2 bits et 0,070 en GPTQ à 2 bits (TwoRoom : 0,021, 0,267, 0,060). Question : à 2 bits, GPTQ rend-il le
prédicteur utilisable, et à quel prix ?

Tout le modèle à 2 bits n'est pas testé : son erreur en GPTQ (0,063 sur PushT) est 2,5 fois celle de RTN à 3 bits sur
tout le modèle (0,025), qui fait déjà perdre 47,8 points (EXP-003).

## Plan

- Grille : même définition qu'EXP-003 à EXP-010, symétrique par canal, qmax = 2^(b-1) - 1. À 2 bits, qmax = 1 : trois
  niveaux (-s, 0, +s) par ligne, codés sur 2 bits (log2 3 = 1,58 bit d'information). Échelles de RTN (maximum de la
  ligne) pour RTN et pour GPTQ : mémoire identique, seul l'arrondi change.
- GPTQ : `gptq.py` inchangé depuis EXP-009 (amortissement 1 %, blocs de 128, ordre naturel, calibration sur le modèle
  en pleine précision). Calibration : celle d'EXP-009 de chaque tâche, réutilisée sans changement (fichier et SHA-256
  inscrits dans le protocole), disjointe des évaluations.
- Versions, par tâche : predictor:2 (RTN) et predictor:2:gptq, 30 couches du prédicteur ; graines 42 à 46 ; PushT sur
  les 500 épisodes d'EXP-003, TwoRoom sur les 500 d'EXP-006 ; pleine précision appariée : EXP-003 et EXP-006.
  20 fichiers, environ 3 h.

## Tests (4 tests, Holm, inversion de signe comme EXP-003, unilatéraux)

K(e, v) : nombre de succès de l'épisode e sur les 5 graines.

- G1 (PushT), G2 (TwoRoom) : GPTQ fait mieux que RTN à 2 bits. x(e) = K(e, predictor:2:gptq) - K(e, predictor:2),
  sens « plus ».
- N1 (PushT), N2 (TwoRoom) : le prédicteur à 2 bits avec GPTQ perd moins de 3 points de réussite. x(e) =
  K(e, predictor:2:gptq) - K(e, fp32) ; test de la statistique décalée y = 20 x + 3 (moyenne de y positive si et
  seulement si l'écart moyen de réussite dépasse -3 points), sens « plus ».

Soutenue si la p corrigée par Holm (sur les 4) est sous 0,05. Marge de 3 points : demi-intervalle à 95 % attendu de
0,9 à 2,0 points à 500 épisodes et 5 graines (écarts du prédicteur à 4 et 3 bits dans EXP-003 et EXP-006).

Descriptifs, avec intervalles bootstrap (`default_rng(2)`) : réussites ; GPTQ moins RTN ; GPTQ moins pleine précision ;
part de la perte de RTN rattrapée comme dans EXP-009 ; sur PushT, GPTQ à 2 bits moins RTN à 3 bits (EXP-003, à
mémoire de 2 contre 3 bits par poids du prédicteur).

## Contrôles bloquants

Avant la grille, par tâche : reproduction de la pleine précision graine 42 (identique octet pour octet à EXP-003 ou
EXP-006, et référence de la sentinelle) ; erreur de reconstruction totale de GPTQ sous celle de RTN à 2 bits sur le
prédicteur, sur la calibration et hors échantillon (`verifier_gptq.py --bits 2 --groupe predictor`). À l'analyse :
tous les fichiers ; mêmes épisodes, départs et buts que les références ; 30 couches quantifiées ; contrôles de
course ; un seul commit dont le code est celui du dépôt ; arbre propre ; calibration disjointe des évaluations et de
SHA-256 conforme. La sentinelle contrôle chaque fichier pendant le calcul. Le journal n'est pas lu avant l'analyse.
