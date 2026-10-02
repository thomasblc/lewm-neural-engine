# Pré-enregistrement d'EXP-016 : la compression casse-t-elle des épisodes précis, ou baisse-t-elle la réussite partout ?

Version 2, rédigée par Claude le 2 octobre 2026 à partir de 10 h 34 (heure du Mac, lue avec `date`). Elle remplace la
version 1 (commit `72fb194`, 9 h 47) après une relecture (1 P0, 8 P1), avant tout calcul de cette analyse sur les données
réelles : aucune statistique de réussite des versions n'a été calculée par ce code. Analyse seule, sans nouveau run :
elle relit les fichiers d'EXP-003, 004, 006, 008, 009, 011, 012 et 014. Existant : K-029.

Changements depuis la version 1 : le nul logistique est retiré (mal posé avec une loi a priori en U : il prévoyait plus
de cassures que l'amincissement) ; le test principal est calibré dans la strate des épisodes réussis 5 fois sur 5 (un nul
« perte plafonnée » faisait rejeter le test global sans effet propre aux épisodes) ; la loi a priori des difficultés est
non paramétrique et estimée sur 10 graines (la queue d'une bêta n'est pas identifiée par 5 essais) ; des contrôles
synthétiques sur la pleine précision réelle vérifient la calibration avant toute lecture ; un excès minimal est exigé ;
la phrase de conclusion est réduite à ce que le test porte ; la remarque sur l'appariement est corrigée.

Amendement 1 (2 octobre, à partir de 11 h 15, avant tout calcul réel ; relecture de la version 2 : aucun P0, 3 P1). Il
l'emporte sur le texte qui suit.
1. Lecture (P1) : une perte qui dépend seulement de la difficulté, plus forte sur les épisodes fragiles que l'amincissement
   (décalage logistique), fait aussi rejeter le test (simulation de la relecture, ad hoc : 60 à 75 % de « soutenue » à
   r = 0,83, 100 % à r = 0,45, 0 % à partir de r = 0,90). Un rejet dit donc seulement : « plus d'épisodes réussis 5 fois
   sur 5 par la référence ne sont jamais réussis par la version qu'une perte proportionnelle de même moyenne ne le
   prévoit ». Il ne sépare pas des épisodes précis d'une perte concentrée sur des épisodes fragiles que 5 essais de la
   référence ne révèlent pas. Les verdicts ne se comparent pas entre versions de pertes différentes (« RTN casse, GPTQ
   non » n'est pas une conclusion de cette analyse). E2 en k_R = 5 se nomme « k_R = 5 », pas « nouveaux ».
2. Moteur (P1) : pour int4 et palettes sur le Neural Engine, l'analyse rapporte l'excès moins celui du 16 bits sur le
   Neural Engine, avec un intervalle de fluctuation tiré de nuls indépendants ; « codage » seulement si cet écart vaut au
   moins 5 épisodes et que sa borne basse dépasse 0, sinon « codage et moteur, ou non séparés ». RTN 8 bits en excès (p
   sous 5 % et au moins 5 épisodes) : les versions PyTorch de la tâche se lisent « à confirmer ».
3. E4 (P1) : le nul partage p entre les deux versions d'une paire (même épisode) et tire leurs binomiales
   indépendamment.
4. Libellés : l'intervalle de l'excès est un intervalle de fluctuation sous le nul, pas un intervalle de confiance du
   nombre d'épisodes cassés ; l'excès est biaisé vers le bas (r est réglé sur une moyenne qui contient les épisodes
   cassés ; relecture : 30 cassures plantées donnent environ +20). p de Holm sous 5 % avec un excès de moins de 5 :
   « excès sous le minimum ».
5. Puissance, corrigée avec une loi a priori ajustée à la strate de PushT (relecture, ad hoc) : 30 cassures à r = 0,43
   détectées 89 % du temps, 2 cassures à r = 0,90 70 %. RTN 3 bits sur PushT (r environ 0,43) est sous le bas de la grille
   de calibration (0,45).
6. Calibration synthétique : elle vérifie le test sur la pleine précision réelle amincie, avec la loi a priori ajustée sur
   ces mêmes données ; elle ne voit pas l'erreur d'estimation de la loi a priori. Prudente, elle peut refuser à tort.

## Ce qui a déjà été vu (divulgation)

- Le 2 octobre au matin, l'audit (`code-exp015/verifications_audit.py`) a compté, avec la pleine précision des graines
  42 à 46 comme référence et sans modèle nul, les épisodes réussis 5 fois sur 5 par la pleine précision et jamais par une
  version : PushT RTN 3 bits 132 sur 353, GPTQ 3 bits 15, RTN 4 bits 20, GPTQ 4 bits 1 ; TwoRoom RTN 3 bits 137 sur 377,
  GPTQ 3 bits 0, RTN 4 bits 16, GPTQ 4 bits 0. Cinq des sept versions de la famille principale ont donc été vues sous
  cette forme ; seules int4 par canal et palettes sur le Neural Engine (PushT) ne l'ont pas été.
- Les pertes de réussite moyennes de toutes les versions sont connues (analyses de chaque expérience et audit).

## Question et phrase visée

Phrase visée, à écrire seulement si S est soutenue : « Parmi les épisodes que la pleine précision réussit 5 fois sur 5
(graines 47 à 51), la version en rate un nombre, 5 fois sur 5, qui dépasse ce que prévoit une perte proportionnelle de
même moyenne sur ces épisodes, de X épisodes [intervalle] ». L'interprétation « la compression casse des épisodes précis,
qu'un nouvel essai ne rattrape pas » n'est écrite qu'avec cette phrase et ses limites.

## Données

Par tâche (PushT : 500 épisodes d'EXP-003 ; TwoRoom : 500 épisodes d'EXP-006), pour chaque épisode e :
- k_R(e) : réussites de la pleine précision PyTorch float32 sur les graines 47 à 51 (référence) ;
- k_42(e) : réussites de la pleine précision sur les graines 42 à 46 (sert à la loi a priori et aux contrôles) ;
- k_V(e) : réussites d'une version sur les graines 42 à 46.
Chaque fichier est contrôlé (épisodes, graine, version, arbre propre) ; pour les runs sur le Neural Engine, moteur
« coreml-ne » et un seul modèle Core ML par dossier.

## Loi a priori des difficultés

p(e), probabilité de réussite de la pleine précision sur l'épisode e. Loi a priori non paramétrique : maximum de
vraisemblance sur une grille de 201 valeurs de 0 à 1 (algorithme EM, 5 000 itérations), ajustée sur k_R + k_42 (10
essais par épisode). Loi a posteriori d'un épisode : à partir de k_R seul (la référence). Rapportés : poids de la grille,
et ajustement d'une bêta-binomiale (alpha, beta, khi-deux d'adéquation sur k_R et sur k_R + k_42), pour comparaison.

## Statistique et test principal (calibré dans la strate)

Strate : les épisodes avec k_R = 5. T1 = nombre d'épisodes de la strate avec k_V = 0.
Nul (perte proportionnelle dans la strate) : chaque épisode de la strate a p(e) tiré de la loi a posteriori (la même
pour toute la strate, puisque k_R = 5 partout) ; la version réussit avec q(e) = min(1, r p(e)), r réglé à chaque tirage
pour que la moyenne des q dans la strate égale la réussite moyenne observée de la version dans la strate ; k(e) tiré
d'une binomiale (5, q(e)). 10 000 tirages, graine 16. p unilatérale = (1 + tirages avec T1 simulé au moins égal à T1
observé) / 10 001. Excès = T1 - moyenne des T1 simulés, intervalle à 95 % = quantiles 2,5 et 97,5 de T1 - T1 simulé.
Sous ce nul, la loi de T1 ne dépend que de k_R et de k_V ; l'appariement des graines 42 à 46 entre versions et pleine
précision n'y entre pas.

## Calibration sur la pleine précision réelle (avant toute lecture)

Pour chaque version de la famille principale : 1 000 versions synthétiques obtenues en gardant chaque réussite de la
pleine précision des graines 42 à 46 avec la probabilité rho = (réussite moyenne de la version) / (réussite moyenne de la
pleine précision 42 à 46), toutes deux sur les 500 épisodes ; test exact ci-dessus sur chacune (1 000 tirages).
Calibration acceptée si le taux de rejet est au plus 6,4 % au seuil de 5 % et au plus 1,2 % au seuil de 5 % / 7.
Sinon le verdict de cette version est « non concluant ». Rapportés aussi : T1 synthétique moyen contre T1 attendu.

## Famille principale (Holm sur 7) et hypothèse S

Règle : les versions des familles qui portent les affirmations du papier (RTN contre GPTQ à 4 et 3 bits ; codages sur
le Neural Engine), sur les deux tâches, dont la perte moyenne contre la pleine précision est établie (intervalle à
95 % entièrement sous 0) :
- PushT : RTN 4 bits et 3 bits (EXP-003), GPTQ 3 bits (EXP-009), int4 par canal et palettes sur le Neural Engine
  (EXP-014) ;
- TwoRoom : RTN 4 bits et 3 bits (EXP-006).
Exclues de cette famille (perte non établie) : GPTQ 4 bits (PushT, TwoRoom), GPTQ 3 bits (TwoRoom), int8, int4 par
canal et palettes sur TwoRoom. Toutes les autres versions sont secondaires (liste plus bas).
S soutenue pour une version si : p au niveau de Holm sous 5 %, excès d'au moins 5 épisodes, calibration acceptée, et
contrôle négatif de la tâche accepté. Sinon « non soutenue » (concentration non établie) ou « non concluant »
(calibration ou contrôle en échec).

## Contrôles

- Contrôles négatifs bloquants : la pleine précision des graines 42 à 46 traitée comme une version, PushT et TwoRoom
  (Holm sur 2) ; s'il rejette pour une tâche, les verdicts de cette tâche sont « non concluants ».
- Contrôle du moteur : Core ML 16 bits sur le Neural Engine (EXP-012), PushT et TwoRoom. S'il montre un excès (p sous
  5 % et excès d'au moins 5), les verdicts d'int4 et des palettes sur le Neural Engine se lisent « codage et moteur ».
- RTN 8 bits : descriptif seulement (un vrai effet y serait possible).

## Versions secondaires (p et excès rapportés, sans conclusion)

Toutes les autres versions à 5 graines (42 à 46) sur les mêmes épisodes, dans les dossiers exp003, exp004, exp006,
exp008, exp009-pusht, exp009-tworoom, exp011-pusht, exp011-tworoom et exp014-*, trouvées par le code (liste écrite dans
la sortie), hors famille principale, contrôles et pleine précision.

## Descriptifs fixés

- E1 : pour chaque version, T1, taille de la strate, réussite dans la strate, T1 attendu, excès et son intervalle, p.
- E2, nouveaux ou amplifiés : pour chaque valeur j de k_R (0 à 5), nombre d'épisodes jamais réussis par la version,
  observé et attendu sous une perte proportionnelle globale (même nul, r réglé sur les 500 épisodes, loi a posteriori de
  chaque épisode) ; excès en j = 5 (« nouveaux ») et en 1 à 4 (« amplifiés »). Repère : la pleine précision 42 à 46.
- E3, nouvel essai (famille principale) : réussite « au moins une fois en 5 graines » de la version et de la pleine
  précision 42 à 46 ; perte au meilleur des 5 rapportée à la perte moyenne.
- E4, recouvrement : indice de Jaccard des épisodes de la strate jamais réussis, pour les paires (RTN 4, RTN 3), (RTN 3,
  GPTQ 3), (RTN 4, int4 sur la puce), (int4, palettes sur la puce) de PushT, comparé à son attendu sous le nul (tirages
  indépendants des deux versions).
- Calibration : comptes observés et attendus de k_V (0 à 5) dans la strate, pour chaque version et chaque contrôle.

## Lecture écrite d'avance

- S soutenue : la phrase visée, avec l'excès et son intervalle ; E2 dit si les épisodes ratés sont plutôt nouveaux ou
  amplifiés, E3 combien un nouvel essai en rattrape.
- S non soutenue : la concentration n'est pas établie (pas « la perte est aléatoire »).
- Le nul est un modèle : un rejet dit « plus concentré qu'une perte proportionnelle dans la strate », pas plus ; la loi
  a priori est estimée, et la calibration synthétique en est le contrôle.

## Limites

Une machine, les épisodes d'EXP-003 et d'EXP-006, 5 graines par version ; les graines 47 à 51 définissent à la fois la
strate et la difficulté a posteriori ; cinq versions principales vues sous une autre forme (divulgué).

## Analyse et tests

`analyse_exp016.py`. Tests `test_analyse_exp016.py` sur données simulées : loi a priori retrouvée ; calibration du test
sur une grille (r de 0,45 à 0,97 ; loi a priori bêta réaliste et mélange à trois points ; vérité amincissement et perte
plafonnée), aux seuils 5 % et 5 % / 7 ; puissance (5 cassures à r = 0,97, 10 à r = 0,45) ; recomptage indépendant ;
refus des fichiers non conformes. Relecture : une passe sur cette version, puis l'analyse.

Résultats des tests, avant toute lecture réelle (commit de cette version) : rejets du test de strate sous le nul, sur la
grille de 16 cases (300 jeux chacune), de 0 à 3,3 % au seuil de 5 % et de 0 à 0,7 % au seuil de 5 % / 7 : le test est
prudent, y compris sous une perte plafonnée et un mélange à trois points. Puissance au seuil de 5 % / 7 : 2 ou 5 cassures
à r = 0,97 et 10 à r = 0,85, détectées dans 100 % des cas ; à r = 0,45 (perte proche de celle de RTN 3 bits), 10 cassures
dans 10 % des cas, 30 dans 96 % : à forte perte, seul un excès important se voit. Une S « non soutenue » pour RTN 3 bits
se lit donc « excès de moins d'une trentaine d'épisodes non exclu ».
