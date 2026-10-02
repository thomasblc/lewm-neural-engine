# Pré-enregistrement d'EXP-016 : la compression casse-t-elle des épisodes précis, ou baisse-t-elle la réussite partout ?

Rédigé par Claude le 2 octobre 2026 à partir de 9 h 47 (heure du Mac, lue avec `date`), avant tout calcul de cette
analyse. Analyse seule, sans nouveau run : elle relit les fichiers d'EXP-003, 004, 006, 008, 009, 011 et 014. Figé par
le commit enregistré dans la sortie. Existant : K-029.

## Ce qui a déjà été vu (divulgation)

- Le 2 octobre au matin, le script de l'audit (`code-exp015/verifications_audit.py`) a compté, avec la pleine précision
  des graines 42 à 46 comme référence, les épisodes que la pleine précision réussit 5 fois sur 5 et qu'une version ne
  réussit jamais : PushT RTN 3 bits 132 sur 353, GPTQ 3 bits 15, RTN 4 bits 20, GPTQ 4 bits 1 ; TwoRoom RTN 3 bits 137
  sur 377, GPTQ 3 bits 0, RTN 4 bits 16, GPTQ 4 bits 0. Aucun modèle nul n'a été calculé, aucune p.
- Les pertes de réussite moyennes de toutes les versions sont connues (analyses de chaque expérience et audit).
- Rien n'a été calculé avec la référence de cette analyse (pleine précision des graines 47 à 51).

## Question

Quand la compression baisse la réussite du planificateur, les échecs ajoutés se concentrent-ils sur des épisodes
précis (les mêmes pour les 5 graines du planificateur, qu'un nouvel essai ne rattrape pas), au-delà de ce qu'expliquerait
une perte égale sur tous les épisodes ? Et ces épisodes sont-ils des épisodes que la pleine précision réussit toujours
(échecs nouveaux) ou des épisodes déjà fragiles (échecs amplifiés, comme chez Jang et al. pour des agents LLM, K-029) ?

## Données

Par tâche (PushT : 500 épisodes d'EXP-003 ; TwoRoom : 500 épisodes d'EXP-006), pour chaque épisode e :
- k_R(e) : nombre de réussites de la pleine précision PyTorch float32 sur les graines 47 à 51 (référence, jamais
  appariée aux versions) ;
- k_V(e) : nombre de réussites d'une version sur les graines 42 à 46.
Chaque fichier lu est contrôlé (épisodes, graine, version, arbre propre), comme dans les analyses précédentes.

## Modèle nul (perte égale sur tous les épisodes)

- Difficulté de chaque épisode : p(e), probabilité de réussite de la pleine précision, inconnue. Loi a priori bêta
  (alpha, beta) ajustée par maximum de vraisemblance (bêta-binomiale) sur les k_R de la tâche ; loi a posteriori de p(e) :
  bêta (alpha + k_R(e), beta + 5 - k_R(e)).
- Nul principal, amincissement : la version réussit l'épisode e avec la probabilité q(e) = min(1, r p(e)), r le même pour
  tous les épisodes, réglé à chaque tirage pour que la moyenne des q(e) égale la réussite moyenne observée de la version
  (chaque réussite de la pleine précision est perdue avec la même probabilité, quel que soit l'épisode). C'est le nul le
  plus favorable aux cassures d'épisodes faciles parmi les pertes « égales » : il est prudent.
- Nul de sensibilité, décalage logistique : logit q(e) = logit p(e) + delta, delta réglé de même.
- Tirages : 10 000 ; à chaque tirage, p(e) tiré de sa loi a posteriori, r (ou delta) réglé par dichotomie, puis
  k(e) tiré d'une binomiale (5, q(e)). Graine du générateur : 16.
- Les graines du planificateur sont appariées entre la pleine précision des graines 42 à 46 et les versions ; la
  référence 47 à 51 ne l'est pas : le nul ignore l'appariement, ce qui le rend plus prudent (il prévoit plus de cassures
  qu'un nul apparié).

## Statistique et tests

T1 = nombre d'épisodes que la référence réussit 5 fois sur 5 et que la version ne réussit jamais (0 sur 5).
p unilatérale = (1 + nombre de tirages avec T1 simulé au moins égal à T1 observé) / 10 001.

Famille principale (7 tests, Holm), fixée par une règle : les versions dont la perte moyenne contre la pleine précision
est établie (intervalle à 95 % entièrement sous 0) parmi les familles centrales du papier :
- PushT : RTN 4 bits et 3 bits (EXP-003), GPTQ 3 bits (EXP-009), int4 par canal et palettes sur le Neural Engine
  (EXP-014) ;
- TwoRoom : RTN 4 bits et 3 bits (EXP-006).
Hypothèse S, pour chacune : T1 dépasse le nul principal. « Soutenue » au niveau de Holm : la compression casse des
épisodes que la pleine précision réussit toujours, plus qu'une perte égale sur tous les épisodes ne l'expliquerait.

Contrôles négatifs (le nul doit les accepter, p au moins 0,05) : la pleine précision des graines 42 à 46 traitée comme
une version (PushT et TwoRoom) ; RTN 8 bits (PushT et TwoRoom). Si un contrôle négatif rejette, le modèle nul est jugé
mal calibré et toutes les conclusions de S se lisent « non concluant ».

Versions secondaires (p rapportées, sans conclusion) : GPTQ 4 bits (PushT, TwoRoom), GPTQ 3 bits (TwoRoom), les
portées d'EXP-003 et 004 (encodeur, prédicteur, autres, encodeur et prédicteur à 4 bits), le prédicteur ternaire
(EXP-011, RTN et GPTQ), int8 sur le Neural Engine (EXP-014, deux tâches), int4 par canal et palettes sur TwoRoom
(EXP-014).

## Descriptifs fixés

- E1 : pour chaque version, T1, nombre d'épisodes réussis 5 fois sur 5 par la référence, part cassée, T1 attendu sous
  chaque nul (moyenne des tirages) et rapport observé sur attendu ; sensibilité au nul logistique (p).
- E2, nouveaux ou amplifiés : parmi les épisodes que la version ne réussit jamais, répartition de k_R (0 à 5) ; part des
  « nouveaux » (k_R = 5) et des « déjà fragiles » (k_R au plus 2) ; même répartition pour la pleine précision des
  graines 42 à 46 (repère du bruit des graines).
- E3, nouvel essai : réussite « au moins une fois en 5 graines » de la version et de la pleine précision (graines 42 à
  46) ; perte au meilleur des 5 rapportée à la perte moyenne (proche de 1 si les échecs sont systématiques, proche de 0
  s'ils sont aléatoires).
- E4, recouvrement : indice de Jaccard des ensembles d'épisodes cassés pour les paires (RTN 4, RTN 3), (RTN 3, GPTQ 3),
  (RTN 4, int4 sur la puce), (int4, palettes sur la puce), comparé à son attendu pour des ensembles tirés au hasard de
  mêmes tailles parmi les épisodes réussis 5 fois sur 5 (10 000 tirages).
- Calibration du nul : pour chaque contrôle négatif, comptes observés et attendus des épisodes (k_R = 5, k_V = j), j de
  0 à 5.
- Ajustement de la loi a priori : alpha, beta, et comptes observés contre attendus de k_R (0 à 5).

## Lecture écrite d'avance

- S soutenue et contrôles négatifs acceptés : la compression casse des épisodes précis ; E2 dit si ce sont des échecs
  nouveaux ou amplifiés, E3 combien un nouvel essai en rattrape.
- S non soutenue : la concentration n'est pas établie (pas « la perte est aléatoire »).
- Le nul principal ne modélise pas l'appariement des graines ; un rejet reste prudent, une acceptation peut venir de
  là. Les deux nuls sont des modèles : un rejet dit « plus concentré que ces deux nuls », pas plus.

## Limites

Une machine, les épisodes d'EXP-003 et d'EXP-006, 5 graines par version. Les graines 47 à 51 de la pleine précision
servent à la fois de référence de difficulté et de définition des épisodes « toujours réussis ».

## Analyse

`analyse_exp016.py` (tests `test_analyse_exp016.py` : ajustement bêta-binomial retrouvé sur données simulées,
calibration du test sous le nul simulé, puissance sur cassures plantées, recomptage indépendant de T1 et des p, refus des
fichiers non conformes). Relecture : une passe, puis une sur les corrections.
