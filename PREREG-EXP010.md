# Pré-enregistrement d'EXP-010 : GPTQ à 4 et 3 bits est-il gratuit ? (PushT, 2 000 épisodes neufs)

Rédigé par Claude le 1er octobre 2026 vers 7 h 15, à la demande de Thomas, après EXP-009 et avant tout calcul
d'EXP-010. Figé par le commit enregistré dans les fichiers.

## Question

EXP-009 : à mémoire égale, GPTQ rattrape 83 à 98 % de la perte de RTN ; sur PushT, GPTQ à 4 bits perd encore 1,5
point et à 3 bits 8,3 points (500 épisodes, ce qui ne tranche pas). EXP-010 est une réplication sur d'autres
épisodes et répond à « jusqu'où c'est gratuit », avec des marges fixées d'avance. Vérifié avant : les poids GPTQ
ont au plus 15 valeurs par ligne à 4 bits (109 couches), 99,7 % diffèrent de la pleine précision.

## Plan

- Les 2 000 épisodes d'EXP-005 (graine de tirage 2026, trajectoires d'EXP-003 et d'EXP-002 exclues, une position
  par trajectoire), jamais vus par EXP-009 ni par sa calibration ; graines 52 à 56 ; pleine précision de référence
  d'EXP-005 (appariement par épisode). Versions : all:4:gptq et all:3:gptq, avec la calibration PushT d'EXP-009
  (`exp009-calib-pusht/calib.pt`, trajectoires disjointes de celles d'EXP-005, vérifié à l'analyse). 10 fichiers,
  environ 6 h.
- Contrôle bloquant avant la grille : la pleine précision graine 52 recalculée doit redonner le fichier d'EXP-005
  octet pour octet (référence de la sentinelle).

## Tests : séquence fixée, chacun à 5 %, arrêt au premier échec

x(e) = K(e, version) - K(e, fp32), succès sur les 5 graines ; d = moyenne(x) / 5. Non-infériorité à la marge m :
d > -m, test par inversion de signe (10 000 vecteurs `default_rng(0)`, comme EXP-003 et EXP-005) sur
y = 10 x + 1 pour m = 2 points et y = 20 x + 1 pour m = 1 point, unilatéral « plus
grand », p = (1 + nombre de sommes permutées au moins aussi grandes) / 10 001.

1. T1 : GPTQ 4 bits perd moins de 2 points.
2. T2 : GPTQ 4 bits perd moins de 1 point.
3. T3 : GPTQ 3 bits perd moins de 2 points.
4. T4 : GPTQ 3 bits perd moins de 1 point.

Un test n'est conclu que si tous les précédents l'ont été (la famille garde 5 % de risque d'erreur). Puissance
(écart-type par épisode 11,2 points, erreur type 0,25 point, estimés sur EXP-005) : T2 environ 88 % si l'écart
réel vaut -0,3 point, 50 % s'il vaut -0,6 ; si GPTQ 4 bits perd vraiment 1,5 point, T2 échouera (c'est la
réponse).

## Descriptif, sans test

Écarts de réussite avec intervalles bootstrap (`default_rng(2)`) pour les deux versions ; pose du T (log M3,
strate des 1 333 épisodes, `default_rng(3)`) et traduction en pixels et degrés, comme EXP-005 ; à côté, les écarts
de RTN 8 et 6 bits d'EXP-005 et les résultats d'EXP-009.

## Contrôles bloquants

Protocole identique à celui d'EXP-005, à la calibration près (fichier et SHA-256 inscrits) ; calibration disjointe
des épisodes d'EXP-005, d'EXP-003 et d'EXP-002 ; mêmes épisodes, départs et buts que les fichiers d'EXP-005 ; 109
couches quantifiées ; contrôles de course ; un seul commit, dont le code est celui du dépôt ; reproduction
identique. La sentinelle contrôle chaque fichier. Le journal n'est pas lu avant l'analyse.
