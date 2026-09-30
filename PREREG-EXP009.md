# Pré-enregistrement d'EXP-009 : GPTQ contre RTN, à mémoire égale, sur PushT et TwoRoom

Rédigé par Claude le 30 septembre 2026 vers 17 h 30, avant tout calcul d'EXP-009 (tests unitaires et un essai de
2 épisodes sur CPU, hors du dossier de résultats). Figé par le commit enregistré dans les fichiers.

## Question

La perte à 4 et 3 bits vient-elle du nombre de bits, ou de la méthode d'arrondi la plus simple (RTN) ? GPTQ
(Frantar et al., arXiv:2210.17323) garde exactement la même grille et les mêmes échelles que notre RTN par canal
(mémoire identique, 4,06 bits par poids à 4 bits) ; seul change le choix des arrondis, qui compense l'erreur de
chaque poids sur les suivants à partir des entrées réelles de la couche.

## Plan

- Méthode : `gptq.py`, variante « poids seuls », amortissement 1 %, blocs de 128 colonnes, ordre naturel, échelles
  de RTN. Variante déclarée : toutes les couches sont calibrées sur les entrées du modèle en pleine précision (le
  GPTQ d'origine quantifie couche après couche). Vérifiée contre une implémentation naïve indépendante et contre
  RTN (H identité) dans `test_gptq.py`.
- Calibration : par tâche, 256 épisodes en pleine précision (graine de tirage 9009, CEM graine 0), trajectoires
  disjointes de tous les épisodes d'évaluation (PushT : EXP-002, 003, 005 ; TwoRoom : EXP-006) ; H accumulé sur
  toutes les entrées vues par chaque couche pendant la planification. Le fichier de calibration et son SHA-256
  sont inscrits dans le protocole. Les couches de l'encodeur qui ne voient qu'une ligne par image (le projecteur)
  reçoivent moins de lignes que d'entrées (environ 800 pour 2 048) : GPTQ leur est appliqué quand même, H étant
  rendue inversible par l'amortissement ; `verifier_gptq.py` les liste.
- Versions : all:4:gptq, all:3:gptq, encoder:4:gptq, autres:4:gptq ; graines 42 à 46 ; PushT sur les 500
  épisodes d'EXP-003, TwoRoom sur les 500 d'EXP-006. Références RTN appariées (mêmes épisodes, mêmes graines) :
  EXP-003 et EXP-004 pour PushT, EXP-006 pour TwoRoom. 40 fichiers, environ 7 h.

## Tests (Holm sur 4, inversion de signe comme EXP-003, unilatéraux « GPTQ plus haut que RTN »)

x(e) = K(e, GPTQ) - K(e, RTN), succès sur les 5 graines.

- G1 : all:4, PushT. G2 : all:4, TwoRoom. G3 : all:3, PushT. G4 : all:3, TwoRoom.

Soutenu si la p corrigée par Holm est sous 0,05. Descriptif : encoder:4 et autres:4 GPTQ contre RTN ; part de la
perte de RTN rattrapée, (K GPTQ - K RTN) / (K fp32 - K RTN), avec intervalle bootstrap (`default_rng(2)`), sans
objet quand RTN perd moins de 2 points.

## Contrôles bloquants

Avant la grille, par tâche : reproduction de la pleine précision graine 42 (identique octet pour octet à EXP-003
ou EXP-006, et référence de la sentinelle) ; calibration complète (toutes les couches vues) ; erreur de
reconstruction totale de GPTQ sous celle de RTN à 4 et 3 bits, sur la calibration et hors échantillon (GPTQ réglé
sur la moitié des paquets de calibration, erreur mesurée sur l'autre moitié ; `verifier_gptq.py`). À l'analyse : tous les
fichiers ; mêmes épisodes, départs et buts que les références ; couches quantifiées ; contrôles de course ; un
seul commit dont le code est celui du dépôt ; calibration disjointe des évaluations et de SHA-256 conforme. La
sentinelle contrôle chaque fichier pendant le calcul. Le journal n'est pas lu avant l'analyse.
