# Pré-enregistrement d'EXP-013 : vitesse, énergie et mémoire de la planification de LeWM selon le moteur, le codage des poids et une charge de vision

Brouillon rédigé par Claude le 1er octobre 2026 à partir de 18 h 02 (heure du Mac, lue avec `date`), avant tout calcul
d'EXP-013. Figé par le commit enregistré dans les fichiers. Les mesures d'EXP-012 (temps par plan, 50 épisodes de pilote,
journal d'énergie par composant pendant TwoRoom) sont connues et ont motivé ce plan ; aucun fichier d'EXP-013 n'existe.

## Plan en vigueur (résumé des amendements 1 à 7 ; en cas de doute, les amendements font foi)

- Machine : MacBook Pro M5 Max, macOS 26.6.2, coremltools 9.0, PyTorch 2.14 (OpenMP, 6 fils). PushT, 50 épisodes de la
  sonde, graine 42, CEM 300 x 30, horizon 5, LeWM publié.
- Sans charge, 14 configurations : `torch` (float32, GPU), `torch-fp16`, `coreml-gpu`, `coreml-ne`, `ne-complet`
  (encodeurs et prédicteur sur le Neural Engine, CEM sur le CPU), `coreml-cpu` (mêmes graphes que `ne-complet`, CPU
  seul), et les codages Core ML int8, int4b32, pal4, int4c sur le GPU et sur le Neural Engine.
- Avec charge (ResNet-50 sur le GPU, démarrée après la mise en route du run) : `torch`, `coreml-gpu`, `coreml-ne`,
  `ne-complet`, à pleine vitesse et à 30 images/s ; vision seule dans chaque mode. 24 éléments par répétition.
- 3 répétitions, ordre contrebalancé ; repos par composant avant, entre et après ; sensibilité OpenMP (`ne-complet`,
  `coreml-cpu`) ; contrôle de réussite descriptif de `ne-complet` (500 épisodes, 2 graines) ; run torch final ;
  balayage du lot (50 cas, 3 répétitions en ordre aléatoire).
- Mesures : temps par plan ; énergie de la puce (IOReport, affichage exclu) pendant la planification au-dessus du repos ;
  par composant ; taille du fichier des poids du prédicteur (pas la mémoire à l'exécution) ; vision sur les fenêtres de planification (images/s, part
  de la vision seule, latences, échéances, plus grand trou) ; énergie marginale à 30 images/s si la cadence est gardée.
- Décisions : règle de constance (3 répétitions du même côté, 5 % temps et vision, 10 % énergie), descriptive ;
  question 2 : réponse fixée par les contrastes principaux `ne-complet` contre `torch` et contre `coreml-gpu`
  (amendement 6, point 4) ; question 3 : « entièrement hors du GPU » par construction, avec le GPU au plus à 0,3 W ; balayage : au lot de 300, aucun codage plus rapide
  que fp16 de plus de 5 %.
- Contrôles bloquants : runs torch identiques au pilote d'EXP-012 ; un commit ; placement attendu ; Neural Engine actif ;
  ordre ; repos à leur place, sans calcul ni vision ; vision prête et couvrant les fenêtres ; canaux complets ; aucun
  autre calcul.
  Le reste est signalé.

Amendement 1 (1er octobre, 18 h 14, avant tout calcul) : la question 3 est retenue. L'essai de faisabilité de l'encodeur
(`essais/encodeur_ane.py`, `test_encodeur_ane.py`, `essai_encodeur_placement.py`) montre l'encodeur d'images réécrit en
(B, C, 1, S) équivalent à 1,9e-6 en float32 et ses 323 opérations toutes sur le Neural Engine (plan de calcul), de même
que l'encodeur d'actions ; l'encodeur d'images n'est appelé que 2 fois par plan (but et départ, en cache). Configuration
ajoutée : `ne-complet` (encodeur, encodeur d'actions et prédicteur sur le Neural Engine, en 16 bits ; CEM et coût sur le
CPU, `--device cpu`), sans et avec la charge de vision. Contrôle de réussite ajouté : `ne-complet` sur les 500 épisodes
d'EXP-003, graines 42 et 43, comparé par épisode à la pleine précision d'EXP-003 (5 graines), intervalle bootstrap ;
descriptif, les tirages de la CEM étant faits sur le CPU (différents de ceux du GPU). Environ 25 min de plus.

Amendement 2 (1er octobre, 18 h 37, avant tout calcul d'EXP-013 ; code aux commits 069f7f2 à 195a035, tests passés,
16 refus vérifiés). Deux relectures du plan (code ; méthode, verdict « à retravailler », 2 P0 et 13 P1) le changent
ainsi. Ce qui suit remplace les sections Plan, Mesures et Contrôles bloquants là où il les contredit.

1. Configurations (22 par répétition). Sans charge, 12 : les 11 précédentes et le témoin `coreml-cpu` (prédicteur et
   encodeurs en Core ML sur le CPU seul, `--device cpu`), qui sépare l'effet de Core ML de celui du processeur.
   `torch-fp16` : poids convertis une fois en float16 (`model.half()`) et autocast, au lieu de l'autocast seul. Avec
   charge : `torch`, `coreml-gpu`, `coreml-ne`, `ne-complet` en 16 bits, dans deux modes de vision : pleine vitesse
   (saturation) et 30 images par seconde (une caméra, travail fixe) ; et la vision seule dans chaque mode (45 s : 10 s
   de chauffe, 35 s de mesure d'énergie). La vision démarre 15 s avant chaque run avec charge.
2. Ordre contrebalancé : répétition 1 tirée au hasard (`default_rng(13)`), répétition 2 dans l'ordre inverse (la dérive
   lente, chaleur ou charge de la batterie, s'annule en moyenne), répétition 3 tirée au hasard ; la vision seule jamais en
   tête. Mise en route hors mesure avant chaque run (`echauffer` : appels du prédicteur et des encodeurs, puis compteurs
   remis à zéro). Un run pendant lequel le fil de lecture a été gelé plus de 12,5 s (pause de la garde de batterie) est
   effacé et refait, 3 essais au plus ; l'analyse refuse tout run avec un tel trou.
3. Énergie. Canaux IOReport « Energy Model » lus par `energie_puce.py` (sans droits administrateur) : CPU, GPU, ANE,
   DRAM, AMCC, DCS, FAB, AFR, ISP, AVE, MSR, DISP, DISPEXT (le canal « GPU », double de « GPU Energy », est écarté).
   Groupes : calcul (CPU, GPU, ANE), mémoire (DRAM, AMCC, DCS, FAB), puce (tous sauf l'affichage DISP et DISPEXT, qui ne
   calcule pas le plan). Mesure principale : énergie de la puce pendant la planification seule (compteurs lus avant et
   après chaque appel de solve, hors du temps chronométré ; une lecture coûte environ 5 ms pour des appels de 10 à 130
   s), moins la puissance au repos multipliée par la durée de planification, divisée par le nombre de plans. Repos :
   4 mesures (avant la répétition 1, entre 1 et 2, entre 2 et 3, après 3 ; 30 s de stabilisation, 60 s de mesure, aucun
   calcul run_grid) ; chaque run utilise la moyenne des repos qui encadrent sa répétition. Sensibilité : repos d'avant
   seul et repos d'après seul ; une comparaison d'énergie n'est dite robuste au repos que si les trois calculs donnent
   le même verdict. L'énergie du run entier (simulation et rendu compris) est rapportée, descriptive. Écran allumé
   pendant tout le calcul, repos compris (`caffeinate -d -i` sur tout le lanceur), luminosité au minimum, inchangée. Le
   journal d'énergie exploratoire et la sentinelle tournent pendant toute l'expérience (charge constante, soustraite par
   le repos).
4. Avec charge de vision. À pleine vitesse, la vision prend tout le GPU laissé libre : son travail varie avec la
   configuration, donc aucune énergie par plan n'est attribuée ; on rapporte le temps par plan, les images par seconde,
   leur part de la vision seule de la même répétition (référence principale), les latences médiane et 99e centile et la
   puissance de la puce. À 30 images par seconde (travail fixe) : échéances manquées par seconde, latences, et énergie de
   planification estimée = (énergie de la puce pendant la planification moins la puissance de la vision seule à 30
   images par seconde de la même répétition multipliée par la durée de planification) divisée par le nombre de plans ;
   estimation valable seulement si la vision garde ses 30 images par seconde, ce qui est rapporté à côté. Les mesures de
   la vision dans les 60 s avant un run (chargement et mise en route du run compris) sont descriptives.
5. Contrôles bloquants ajoutés. Usage réel du Neural Engine, à l'exécution : puissance de l'ANE au-dessus du repos
   pendant la planification supérieure à 1,0 W pour `coreml-ne`, ses codages et `ne-complet`, inférieure à 0,3 W pour les
   autres moteurs (un repli silencieux de Core ML vers le CPU ou le GPU, que le plan de calcul ne montre pas, est ainsi
   refusé). Déterminisme : chaque configuration donne des fichiers identiques (hors champs git) d'une répétition à
   l'autre et avec ou sans vision ; les runs torch sont identiques au pilote d'EXP-012. Placement : toutes les
   opérations sur le processeur demandé, CPU pour `coreml-cpu` ; pour les codages compressés, le placement des opérations
   de décompression des poids est vérifié par un essai avant le lancement, et si certaines tombent sur le CPU la règle
   est ajustée par un amendement daté, avant tout calcul. Configurations, ordre et commit uniques vérifiés.
6. Règle de décision (n = 3, aucun test, aucune valeur p) : une différence de temps par plan est dite établie si les 3
   rapports appariés (même répétition) sont tous sous 0,95 ou tous au-dessus de 1,05 ; pour l'énergie par plan, 0,90 et
   1,10. Comparaisons : chaque configuration sans charge contre `torch` (temps et énergie) et contre `coreml-gpu`
   (énergie). Sinon : « non établi ». Rapportée comme une règle de constance, pas comme un test.
7. Balayage de la taille du lot (`balayage_lot.py`, après les runs) : appels isolés du prédicteur, lots de 1, 8, 32,
   100 et 300 candidats (T = 3), 4 codages, Neural Engine (graphe « 1ctb ») et GPU (graphe d'origine) ; 10 appels de
   mise en route, puis médiane et quartiles de 50 appels, énergie de la puce sur ces 50 appels, placement. Prédiction
   enregistrée : au lot de 300, le calcul limite (environ 900 opérations par octet de poids en 16 bits), et les codages
   des poids n'accélèrent pas le prédicteur (écart sous 5 % contre fp16, sur les deux processeurs). Aux petits lots, la
   lecture des poids peut compter : aucune prédiction de sens.
8. Codages des poids : fonctions de coremltools appliquées aux poids 16 bits (int8 linéaire symétrique par canal, int4
   par blocs de 32, palettes de 16 valeurs par k-moyennes), sans calibration. Leur réussite n'est pas établie ici (50
   épisodes de contrôle) ; aucun chiffre de vitesse d'un codage n'est présenté sans cette mention ; une autre expérience
   mesure leur réussite.
9. Validation des compteurs : IOReport donne l'estimation d'énergie de la puce, pas une mesure à la prise. Plan : sur
   batterie, environ 20 min, une charge fixe ; comparer la décharge de la batterie (mWh) à la somme IOReport plus une
   constante pour le reste du Mac. Faite si Thomas débranche le chargeur ; sinon le papier écrit « énergie estimée par
   les compteurs de la puce (IOReport), non validée contre une mesure externe ».
10. Durée estimée : environ 2 h (3 répétitions d'environ 28 min, contrôle de réussite d'environ 25 min, balayage
    d'environ 10 min), à confirmer par l'essai court.

Amendement 3 (1er octobre, 19 h 14, avant tout calcul d'EXP-013). Relecture des corrections par deux sous-agents
(code : 11 P1 ; méthode : « prêt avec corrections », 1 P0 et 23 P1). Ce qui suit complète et corrige l'amendement 2 ;
en cas de contradiction, cet amendement l'emporte.

1. Question 2, contrastes pré-enregistrés (P0). Pour chaque moteur e et mode de vision, par répétition k :
   ralentissement S = temps par plan avec charge / temps par plan sans charge ; part de la vision V = images par seconde
   pendant la planification / images par seconde de la vision seule de la même répétition. Contrastes : `coreml-ne` et
   `ne-complet` contre `torch` et contre `coreml-gpu`, sur S, V et la latence p99 de la vision, par la règle de
   constance à 5 %. À 30 images par seconde, la cadence est dite gardée si les images par seconde pendant la
   planification sont au moins 29,7 et les échéances manquées au plus 0,3 par seconde, dans les 3 répétitions.
2. Vision mesurée sur les fenêtres de planification : chaque appel de solve enregistre son début et sa fin (horloge
   `time.time`, la même que la vision) ; la vision enregistre la fin, la latence et le retard de chaque image ; images
   par seconde, latences médiane et 99e centile (sur toutes les images, plus le maximum des centiles par tranche de 5 s)
   et échéances manquées sont calculés sur les images finies dans ces fenêtres. Contrôle bloquant : aucun trou de plus
   de 2 s sans image dans une fenêtre ; la vision a écrit sa ligne « prête » avant le run ; sa cadence visée est celle du
   mode. La vision démarre, attend d'être prête, puis 10 s de chauffe avant le run.
3. Énergie à 30 images par seconde : estimée seulement si la cadence est gardée dans cette répétition ; sinon « non
   attribuable » (la vision en retard fait moins de travail, la soustraction serait fausse). Sensibilité : repos de la
   répétition, plus l'énergie d'une image de la vision seule ((puissance de la vision seule moins repos) / ses images par
   seconde) multipliée par les images produites pendant la planification. Formulation : énergie marginale de la puce pour
   planifier en plus d'une charge ResNet-50 à 30 images par seconde.
4. Vision seule : mesure d'énergie (35 s) après la ligne « prête » et 10 s de chauffe ; images par seconde sur exactement
   la même fenêtre ; fichier neuf à chaque lancement (jamais de mélange avec un lancement interrompu) ; commit et cadence
   enregistrés et vérifiés.
5. Contrôles bloquants ajoutés : canaux IOReport de la puce tous présents et finis (runs, repos, vision seule) ; moteur et
   codage enregistrés égaux au nom du dossier ; ordre réel des runs égal à l'ordre tiré, recalculé par l'analyse ;
   chaque mesure au repos placée entre les runs qu'elle encadre ; trou de plus de 7,5 s entre deux lectures (au lieu de
   12,5 s : pauses courtes du régulateur de batterie comprises) ou un autre calcul run_grid pendant le run : run refait
   par vitesse.py, refusé par l'analyse ; mesure au repos refaite sur-le-champ si un calcul ou une charge de vision
   tournait ; balayage du lot complet (40 cas, 3 répétitions) et du même commit ; run torch final présent.
6. Signalements (rapportés, non bloquants) : déterminisme par configuration (identique d'une répétition à l'autre et
   avec ou sans vision ; le contrôle bloquant reste l'identité des runs torch au pilote d'EXP-012, car Core ML et les
   calculs sur CPU n'ont jamais été vérifiés en répétition) ; puissance du GPU au-dessus du repos supérieure à 0,3 W pour
   `ne-complet` et `coreml-cpu` (question 3 : planificateur hors du GPU) ; énergie par plan au-dessus du repos négative ou
   nulle ; alerte thermique ou de performance (`pmset -g therm`, lue toutes les 5 s) ; chargeur débranché ; processus
   autre que le calcul à plus de 50 % d'un cœur ; écart des 3 runs torch de plus de 2,5 % (règle du temps alors dite peu
   informative). Le contrôle du Neural Engine (supérieur à 1,0 W, ou inférieur à 0,3 W) s'applique aussi aux runs avec
   vision.
7. Contrastes sans charge (temps et énergie, règle de constance) : chaque configuration contre `torch` ; `coreml-ne`,
   `ne-complet` et `coreml-cpu` contre `coreml-gpu` ; `coreml-gpu` contre `torch-fp16` (le cadre à précision égale) ;
   `ne-complet` contre `coreml-ne` (encodeur et CEM hors du GPU) ; `ne-complet` contre `coreml-cpu` (Neural Engine contre
   CPU pour les mêmes graphes Core ML) ; chaque codage contre fp16 sur le même processeur. L'énergie brute par plan
   (sans soustraction du repos) est rapportée à côté.
8. Règle de constance, formulation : « plus bas (ou plus haut) dans les 3 répétitions » si les 3 rapports appariés sont
   tous sous 0,95 (ou au-dessus de 1,05) pour le temps, la latence et la part de la vision, 0,90 et 1,10 pour l'énergie ;
   sinon « non constant ». Critère descriptif, sans taux d'erreur, sur une machine et une session, pour environ 40
   comparaisons et 3 variantes du repos. Une dérive linéaire pousse les répétitions 1 et 2 en sens contraires (ordre
   inverse) : elle ne peut pas passer la règle seule.
9. Mise en route : deux appels du prédicteur à chacune des longueurs T = 1, 2 et 3 du déroulé (au lieu de T = 3
   seulement), un de l'encodeur d'actions, un de l'encodeur.
10. Fils et OpenMP : PyTorch 2.14 utilise OpenMP (6 fils par défaut, `OMP_NUM_THREADS` non fixé) ; réglage par défaut
    retenu comme partie de l'implémentation, fils et variables `OMP_*` enregistrés par run. Sensibilité ajoutée après la
    répétition 3 : `ne-complet` avec `OMP_WAIT_POLICY` par défaut et `PASSIVE`, 2 runs chacun, ordre ABBA, entre deux
    mesures au repos ; descriptive.
11. Batterie, chargeur, état de la machine : départ seulement avec au moins 60 % de batterie (4 h d'attente au plus) ;
    pendant une pause de la garde, la charge de vision est gelée avec le calcul ; courant, tension, charge et chargeur
    enregistrés toutes les 5 s ; `pmset -g`, `pmset -g custom`, `pmset -g therm`, fils, canaux IOReport et processus
    actifs écrits au début et à la fin ; départ refusé si une charge de vision ou le régulateur de batterie tourne. Les
    paquets Core ML sont écrits dans un dossier `.noindex` (Spotlight ne les indexe pas pendant la mesure suivante).
    Processus les plus actifs enregistrés à chaque lecture et dans les mesures au repos. À Thomas, s'il le veut bien :
    luminosité automatique et True Tone coupés, pas d'usage du Mac pendant l'expérience.
12. Arrêts propres : vitesse.py arrête la vision et le calcul en cours sur SIGTERM ; la vision s'arrête seule si son
    parent disparaît ; un fichier de plan (`exp013-ordre.json`) différent du code arrête la reprise.
13. Balayage du lot, refait : les 40 cas convertis et chargés d'abord (détail du placement par type d'opération) ;
    puissance au repos de la puce 30 s ; 3 répétitions, chacune dans un ordre tiré au hasard (`default_rng(17)`) ; par
    cas, 10 appels de mise en route puis des appels en boucle pendant au moins 3 s, énergie nette du repos. Prédiction
    reformulée : au lot de 300, aucun codage n'est plus rapide que fp16 de plus de 5 % dans les 3 répétitions, sur l'un ou
    l'autre processeur ; les ralentissements sont rapportés à part. Justification corrigée : à 300 candidats et T = 3,
    chaque poids sert 900 fois par appel, et la lecture des poids (23 Mo en fp16) n'est qu'une petite part du trafic
    mémoire (les activations des couches linéaires font environ 5 fois les poids) ; l'argument « le calcul limite »
    n'est pas établi et n'est pas avancé. Formulation : appels isolés, caches chauds, même entrée ; aux petits lots, la
    disposition « 1ctb » aligne mal le dernier axe (2 octets au lot de 1), si bien que les temps du Neural Engine y
    mesurent aussi la disposition.
14. Essai de placement des codages (`essai_placement_codages.py`, avant le lancement) : détail du placement par type
    d'opération, « aucun » compris, et mémoire du processus (empreinte physique, RSS) avant et après chargement, pour
    fp16, int8, int4b32 et pal4 sur les deux processeurs. Si la décompression se fait au chargement, le papier dit qu'un
    codage ne change que la taille du fichier des poids, pas la lecture des poids pendant le calcul.
15. Poids : la colonne compare le seul prédicteur (prédicteur et projection de sortie), de la même façon pour tous
    (fichier weight.bin du prédicteur Core ML, ou paramètres PyTorch au type utilisé) ; la taille de chaque réseau est
    rapportée à part ; « taille du fichier des poids », pas mémoire à l'exécution.
16. Codages, formulations : int8 linéaire symétrique par canal ; int4b32 arrondi par blocs de 32 sans calibration (le
    schéma qui perdait environ 14 points à 4 bits dans EXP-003 est de cette famille) ; pal4 une palette de 16 valeurs par
    tenseur (réglage par défaut de coremltools 9.0). La réussite sur 50 épisodes est affichée avec son intervalle de
    Wilson à côté de chaque codage, avec la mention « réussite non établie, mesurée par une autre expérience ». Le coût
    par plan ne dépend pas des valeurs des poids (la CEM n'a pas d'arrêt anticipé) : temps et énergie d'un codage sont
    des mesures de coût valides sans sa réussite.
17. `coreml-cpu`, formulation corrigée : il fait tourner les mêmes graphes Core ML que `ne-complet` sur le CPU seul ; le
    contraste `ne-complet` contre `coreml-cpu` isole le Neural Engine contre le CPU pour ces graphes. Il n'isole ni Core
    ML contre PyTorch (aucun run PyTorch sur CPU) ni le meilleur déploiement sur CPU.
18. Question 3, contrôle de réussite : descriptif, sans marge ni affirmation de non-infériorité ; le contrôle de pause
    n'y est pas appliqué (la réussite ne dépend pas du temps).
19. Validation des compteurs (remplace l'amendement 2, point 9), faite seulement si Thomas débranche le chargeur : sur
    batterie, trois segments d'environ 5 min (repos, `coreml-gpu` en boucle, `ne-complet` en boucle) ; puissance de la
    batterie intégrée (InstantAmperage fois Voltage) ; constante du reste du Mac estimée sur le segment au repos seul,
    puis prédiction des deux segments chargés. Acceptation fixée d'avance : chaque prédiction à 10 % près, et le rapport
    entre les deux configurations à 10 % près (les affirmations du papier sont des rapports). Sinon : « estimée par les
    compteurs de la puce, non validée ». Un recoupement avec `sudo powermetrics` (par Thomas) vérifierait le choix des
    canaux, pas le modèle d'énergie.
20. Corrections du texte : l'exclusion des appels de plus de 3 fois la médiane (Mesures) n'est pas appliquée, la règle
    « run refait après une pause » la remplace ; « mêmes tirages » est faux pour `coreml-cpu` comme pour `ne-complet`
    (CEM sur le CPU) ; les appels de solve durent de 5 à 130 s.
21. Essai de bout en bout avant le lancement : `vitesse.py --essai` (10 épisodes, mesures brèves, dossier de travail hors
    de resultats/) puis `analyse_exp013.py --essai` sur ses fichiers.

Amendement 4 (1er octobre, 19 h 30, avant tout calcul d'EXP-013). L'essai de placement des codages
(`essai_placement_codages.py`, `resultats/preparation-exp013/placement-codages.json`, commit `54faef4`) montre :
- int4 par blocs de 32 sur le Neural Engine : 211 des 361 opérations du prédicteur sur le CPU, 150 sur le Neural Engine
  (même résultat par blocs de 64) ; int4 par canal, int8 par canal et palettes : toutes sur le Neural Engine ; sur le
  GPU, toutes les opérations sur le GPU pour les quatre codages ;
- les opérations de décompression des poids (`constexpr_blockwise_shift_scale`, `constexpr_lut_to_dense`) n'ont aucun
  processeur dans le plan de calcul : la décompression se fait au chargement ; l'empreinte mémoire du processus après
  chargement et 10 appels ne baisse pas avec le codage (Neural Engine : 354 à 365 Mo ; GPU : 477 à 521 Mo).
Changements :
1. Codage ajouté : `int4c` (int4 linéaire symétrique par canal), sur le GPU et sur le Neural Engine, sans charge, et dans
   le balayage du lot : 14 configurations sans charge, 24 éléments par répétition. Contraste ajouté : int4 par blocs
   contre int4 par canal, sur chaque processeur.
2. Placement des codages : consigné et rapporté comme un résultat, non bloquant. Le contrôle bloquant de placement reste
   pour les configurations en 16 bits (`coreml-gpu`, `coreml-ne`, `ne-complet`, `coreml-cpu`). Le contrôle bloquant du
   Neural Engine actif (plus de 1,0 W) reste pour `coreml-ne` et `ne-complet` en 16 bits ; pour les codages sur le Neural
   Engine, une puissance de 1,0 W ou moins est signalée.
3. Formulation : un codage peut faire tomber une partie du prédicteur sur le CPU (int4 par blocs sur ce Neural Engine,
   macOS 26.6.2, coremltools 9.0) ; un codage ne réduit que la taille du fichier des poids, pas l'empreinte mémoire du
   processus, dans ce montage.
4. Durée estimée : environ 2 h 45.

Amendement 5 (1er octobre, 19 h 59, avant tout calcul d'EXP-013 ; code au commit `8398d18`, tests passés, 34 refus).
Relecture des corrections des amendements 3 et 4 par deux sous-agents (code : 2 P0, 4 P1 ; méthode : 2 P0, 11 P1) et
premier essai de bout en bout (`vitesse.py --essai`, arrêté après la répétition 1, 10 épisodes, EXP-011 en même temps :
temps indicatifs seulement). Ce qui suit l'emporte sur les amendements précédents.
1. Vision, écriture : la charge de vision écrit à l'arrêt les images depuis sa dernière ligne (elles étaient perdues,
   jusqu'à 5 s ; l'essai le montrait : 21,9 images/s au lieu de 30 et un trou de 2,2 s à la fin d'un run). Contrôle
   bloquant : les lignes de la vision, contiguës, couvrent toutes les fenêtres de planification (la vision tournait et a
   tout écrit) ; le plus grand trou sans image dans une fenêtre est un résultat (vision affamée par le planificateur),
   rapporté, plus un refus.
2. Compteur des autres calculs : `pgrep` exclut le processus qui l'appelle et ses ancêtres, donc le run lui-même ; la
   valeur attendue est 0 (et non 1). Sans cette correction, l'analyse aurait refusé tous les runs réels.
3. Démarrage de la vision : run_grid charge et convertit le modèle, fait sa mise en route, écrit `go.pret` et attend ;
   la vision démarre alors (prête, puis 10 s de chauffe), puis le signal `go` lance la planification. La chauffe de la
   vision ne dépend plus du temps de conversion du moteur.
4. Placement : contrôle bloquant pour toutes les configurations Core ML, contre le placement de l'essai de placement
   (`resultats/preparation-exp013/placement-codages-70ecada.json`, commit `70ecada`) : int4 par blocs sur le Neural
   Engine 211 opérations sur le CPU et 150 sur le Neural Engine ; tous les autres cas sur le processeur visé. Contrôle
   bloquant du Neural Engine actif (plus de 1,0 W) pour toutes les configurations sur le Neural Engine sauf int4 par
   blocs (signalé). Les chiffres de l'amendement 4 sur la mémoire venaient d'un fichier écrasé depuis ; les chiffres en
   vigueur sont ceux de `70ecada` : empreinte du processus après chargement et 10 appels, Neural Engine 360 à 370 Mo,
   GPU 470 (pal4) à 526 Mo (fp16 501 Mo) ; le témoin int4 par blocs de 64 est ajouté à l'essai de placement.
5. Décompression, formulation corrigée (l'amendement 4 concluait trop) : le plan de calcul n'assigne aucun processeur
   aux constantes ni aux opérations de décompression ; l'int4 par blocs change le placement sur le Neural Engine ; le
   moment de la décompression n'est pas observé ; l'empreinte mémoire mesurée est celle du processus client (2
   mesures, environ 1 Mo de résolution), la mémoire côté Neural Engine n'est pas observée. Rien sur la vitesse ne vient
   de l'essai de placement. Observation de l'essai de bout en bout, exploratoire : sur le GPU, les codages linéaires
   (int8, int4) rendent le prédicteur Core ML environ 10 fois plus lent (5 900 à 8 000 ms par plan contre 589 ms en fp16,
   pal4 657 ms) ; sur le Neural Engine, int8 et int4 par canal ne changent rien (760 à 773 ms) et int4 par blocs le rend
   3 fois plus lent (repli sur le CPU). EXP-013 le mesure ; durée allongée d'environ 1 h.
6. Énergie à 30 images par seconde : la « sensibilité par image » est retirée (égale à l'estimation principale quand la
   cadence est gardée). À la place, bornes : puissance de la vision seule la plus basse et la plus haute des 3
   répétitions ; même condition (cadence gardée) ; le nombre de répétitions attribuables est affiché. Descriptive, sans
   contraste.
7. Question 2, réponse pré-enregistrée. Contrastes principaux : `ne-complet` contre `torch`, ralentissement S et part de
   la vision V à pleine vitesse, et cadence de 30 images/s gardée. « Oui » si S est plus bas et V plus haut dans les 3
   répétitions, et si `ne-complet` garde les 30 images/s quand `torch` ne les garde pas (ou si les deux les gardent) ;
   « non » si ni S ni V ne sont en faveur de `ne-complet` et que la cadence n'est pas mieux gardée ; « en partie »
   sinon. Les autres contrastes de la question 2 sont secondaires.
8. Question 3 : « planificateur entièrement hors du GPU » n'est écrit que si la puissance du GPU au-dessus du repos
   pendant la planification de `ne-complet` reste au plus 0,3 W dans les 3 répétitions ; sinon « GPU à X W ».
9. OpenMP : la sensibilité couvre aussi `coreml-cpu` (ABBA, 2 runs par variante) ; un moteur dont l'énergie par plan
   change de plus de 10 % avec `OMP_WAIT_POLICY=PASSIVE` est dit « sensible à OpenMP », et chaque contraste qui le met en
   jeu porte cette mention.
10. Mesures au repos : refaites une fois sur-le-champ si un processus hors de l'expérience dépasse 20 % d'un cœur au
    début ou à la fin, gardées et signalées s'il est encore là ; refus si un calcul ou une vision tourne encore.
11. Ordre : la vision seule n'est plus jamais en dernier dans une répétition (la garde de batterie sort après 4 contrôles
    sans calcul, un par minute).
12. Reprise : refusée si un fichier d'EXP-013 vient d'un autre commit ; un run déjà écrit est recontrôlé (pause,
    autre calcul) et refait s'il le faut ; un run dont la vision s'arrête est effacé et refait ; écritures atomiques.
13. Balayage du lot : modèles chargés cas par cas (pas 50 en mémoire) ; 30 s de stabilisation, repos de 30 s avant et
    après, moyenne ; état de la batterie au début et à la fin ; canaux contrôlés ; 50 cas (int4c compris). Son échec
    n'empêche ni les contrôles finaux ni le reste de l'analyse (signalé). Point 13 de l'amendement 3 : l'argument du
    trafic des activations est retiré (le cache du système peut les garder) ; seule la prédiction reste.
14. Divers : règle de constance « non calculable » si une valeur est nulle ou négative ; mention « référence bruitée »
    sur les verdicts de temps si les runs torch s'écartent de plus de 2,5 % ; écart des runs torch en énergie rapporté ;
    puissance de la puce sur les fenêtres de planification dans le tableau avec charge ; maximum des 99e centiles par
    tranche de 5 s calculé ; vision seule à 30 images/s vérifiée contre la règle de cadence (signalé) ; Core ML complète
    les longueurs T = 1 et 2 à 3 (environ 25 % de calcul du prédicteur en plus dans 2 appels sur 5), à dire dans la
    comparaison des moteurs ; int4 par blocs contre int4 par canal sur le Neural Engine est un contraste de placement,
    sur le GPU un contraste de granularité. Environ 90 verdicts au total (pas 40).
15. Validation des compteurs : protocole de l'amendement 3 (point 19) complété par un segment de calibration (charge
    torch sur le GPU) pour ajuster une perte de conversion proportionnelle à la charge ; script écrit avant la mesure ;
    faite avec Thomas (chargeur débranché, environ 20 min), sinon la mention « non validée » reste.
16. Durée estimée : environ 3 h 45.

Amendement 6 (1er octobre, 20 h 20, avant tout calcul d'EXP-013 ; code au commit `af37479`, tests passés : 34 refus, table de
12 cas pour la réponse à la question 2). Relecture des corrections de l'amendement 5 (méthode : « prêt avec
corrections », 7 P1). Ce qui suit l'emporte sur les amendements précédents.
1. Mémoire, retrait : la phrase de l'amendement 4 (point 3) « un codage ne réduit que la taille du fichier des poids, pas
   l'empreinte mémoire du processus » est retirée. Ce qui reste : l'empreinte du processus client après chargement et
   10 appels n'est pas plus basse pour les modèles codés (1 processus par cas ; GPU int8 520 à 526 Mo contre fp16 499 à
   501 Mo selon le passage) ; la mémoire côté Neural Engine n'est pas observée ; aucune affirmation sur la mémoire à
   l'exécution.
2. Blocs de 64 : le témoin est maintenant mesuré (`resultats/preparation-exp013/placement-codages-1b0123c.json`) : int4 par
   blocs de 64 sur le Neural Engine, 211 opérations sur le CPU et 150 sur le Neural Engine, comme par blocs de 32.
3. Observations de l'essai de bout en bout (amendement 5, point 5), formulation : essai, 1 run par cas, 10 épisodes,
   EXP-011 en même temps sur le GPU ; ce sont des temps par plan, pas des temps du prédicteur ; aucune cause n'est
   avancée ; rien n'en est cité. Divulgation : des temps par appel au lot de 300 ont été vus avant le balayage du lot
   (essai de placement : Neural Engine int4c 7,97 ms contre fp16 8,00 ms ; GPU pal4 15,8 contre 16,0 ms) ; le balayage
   est donc rapporté comme une observation, pas comme une prédiction confirmée.
4. Question 2, divulgation et réponse. Le contraste principal (point 7 de l'amendement 5) a été choisi après le premier
   essai, qui avait produit des runs `torch-vision`, `ne-complet-vision` et à 30 images/s (10 épisodes, EXP-011 en même
   temps) ; il suit la question (le seul planificateur entièrement hors du GPU, contre le déploiement par défaut). Biais
   de précision : `torch` est en float32, `ne-complet` en float16 ; la même règle de réponse est donc appliquée aussi
   contre `coreml-gpu` (float16), principal au même titre, et les deux réponses sont rapportées ensemble. Règle de
   réponse : « compromis » si un critère est en faveur de `ne-complet` et un autre contre (ralentissement plus haut,
   part de la vision plus basse, ou la référence garde les 30 images/s quand `ne-complet` ne les garde pas) ; « oui » si
   ralentissement plus bas et part de la vision plus haute dans les 3 répétitions et si `ne-complet` garde les
   30 images/s ; « non » si aucun critère n'est en faveur ; « en partie » sinon. Si la vision seule ne tient pas sa
   cadence de 30 images/s dans les 3 répétitions, le bras à 30 images/s n'est pas interprétable et la réponse se lit
   sur le ralentissement et la part de la vision seuls. Les autres contrastes de la question 2 sont rapportés dans le
   même tableau, comme secondaires.
5. Vision entièrement affamée (aucune image finie dans les fenêtres) : 0 image/s, latences absentes, rapports « non
   calculables » ; un résultat, pas un refus (la couverture par les lignes prouve que la vision tournait). vitesse.py
   contrôle aussi la couverture après chaque run avec vision et refait le run sinon.
6. Énergie à 30 images/s : estimée sur les répétitions qui gardent la cadence, avec leur nombre n ; bornes sur ces mêmes
   répétitions ; les bornes ne couvrent que la dérive de la puissance de la vision seule, pas la non-additivité,
   inhérente à une mesure marginale.
7. Validation des compteurs : obligatoire avant qu'un chiffre d'énergie entre dans le papier (sinon, aucun chiffre
   d'énergie dans le papier) ; segments : repos, `coreml-gpu`, `ne-complet`, `ne-complet` avec la vision à 30 images/s,
   et la charge de calibration (torch sur le GPU) ; script écrit avant la mesure. L'énergie marginale sous charge n'est
   jamais répartie par composant.
8. Neural Engine sous charge de vision : sa puissance baisse (essai : environ 4,3 W sans charge, 2,4 W avec) ; le
   contrôle bloquant de plus de 1,0 W ne s'applique qu'aux runs sans charge ; sous charge, une puissance de 1,0 W ou
   moins est signalée.
9. OpenMP : « sensible à OpenMP » seulement si l'énergie par plan change de plus de 10 % et si les intervalles des deux
   runs de chaque variante sont disjoints ; le rapport des temps est rapporté aussi.
10. Question 3 : « entièrement hors du GPU » repose d'abord sur la construction (encodeurs et prédicteur en Core ML sur le
    Neural Engine d'après le plan de calcul, CEM et coût sur le CPU), la puissance du GPU servant de contrôle (au plus
    0,3 W au-dessus du repos dans les 3 répétitions) ; l'écart de la puissance du GPU entre les mesures au repos est
    rapporté.
11. Divers : fenêtre « vision avant le run » réduite à la chauffe (de la ligne « prête » au début de la mesure) ;
    stabilisation de 30 s avant le repos final du balayage ; formulations corrigées dans le code (moment de la
    décompression non observé). Limite connue, non corrigée : le filtre des processus de fond ignore tout processus
    python (un autre calcul python que run_grid serait invisible) ; EXP-011 sera finie avant le lancement.
12. Relecture du code de l'amendement 5 (1 P0, 10 P1), corrigée au commit `a29483e` (tests : 35 refus) : garde de batterie
    propre à EXP-013 (copie de `outils/garde_batterie.sh`, inchangée pendant qu'EXP-011 s'en sert), qui ne sort qu'après
    8 contrôles d'affilée sans calcul au lieu de 4 (une mesure au repos refaite au départ l'aurait fait sortir pour la
    nuit) ; une pause de la garde pendant la mise en route d'un run avec vision gèle aussi la vision et fait refaire le
    run ; une vision qui ne démarre pas fait refaire le run au lieu d'arrêter la nuit ; la couverture de la vision est
    recontrôlée à la reprise ; les processus de fond sont reconnus par leur script (les scripts de l'expérience sont
    exclus, tout autre processus python compte), ce qui corrige la limite du point 11 ; écritures atomiques des mesures
    au repos et de meta.json ; le lanceur attend aussi vitesse.py, balayage_lot.py et charge_vision.py ; mentions
    OpenMP et « référence bruitée » portées aussi sur la question 2 ; rapport OpenMP non calculable si une énergie est
    nulle ou négative ; canaux contrôlés dans chaque mesure du balayage.

Amendement 7 (1er octobre, 20 h 50, avant tout calcul d'EXP-013 ; code au commit `4563fe2`, tests passés : 36 refus).
Dernière relecture du code (aucun P0 ; 1 P1, 7 P2) :
1. Une pause de la garde pendant le chargement, la conversion ou la mise en route d'un run (avant le début des mesures)
   n'était pas vue : le run aurait été mesuré sur une puce refroidie après la recharge. Tout gel du calcul, à tout moment
   et pour tous les runs, fait maintenant refaire le run.
2. Avant un nouvel essai d'un run, l'ancien calcul est attendu jusqu'à sa fin (jamais deux calculs à la fois).
3. Vision seule : refaite jusqu'à 3 fois si elle ne démarre pas ou s'arrête ; sans aucune image pendant la mesure, refus
   bloquant clair.
4. Une mesure au repos encore contaminée au second essai est renommée avant l'arrêt (jamais reprise comme faite).
5. Question 2 : si la vision de la référence est entièrement affamée (part 0) et pas celle de `ne-complet`, le rapport
   est infini, donc en faveur de `ne-complet` (au lieu de « non calculable ») ; la mention OpenMP figure aussi dans le
   tableau des contrastes secondaires.
6. Le lanceur refuse de partir si une charge de vision tourne déjà (orpheline gelée).

## Questions

1. Que fait gagner chaque choix de déploiement, en temps par plan, en énergie par plan et en mémoire des poids : le
   moteur (PyTorch sur le GPU en 32 et 16 bits, Core ML sur le GPU, Core ML sur le Neural Engine) et le codage des poids
   dans Core ML (16 bits, 8 bits par canal, 4 bits par blocs de 32, palettes de 16 valeurs) ?
2. Avec une charge de vision en continu sur le GPU (un robot voit et agit en même temps), le planificateur sur le
   Neural Engine garde-t-il son temps par plan, et la vision ses images par seconde, mieux que le planificateur qui
   partage le GPU ?
3. Conditionnelle à l'essai de faisabilité de l'encodeur (`essais/essai_encodeur_placement.py`) : un planificateur
   entièrement hors du GPU (encodeur et prédicteur sur le Neural Engine, CEM sur le CPU) ; sinon cette partie est retirée
   par un amendement daté, avant tout calcul.

Existant (recherches faites, citées) : K-021 (co-exécution GPU et NPU établie en général : HeteroInfer, HaX-CoNN ;
jamais pour un world model qui planifie), K-023 (aucun world model qui planifie mesuré sur le Neural Engine), K-024
(sebspel : vitesse et réussite de LeWM en FP16, FP8, INT8 sur GPU NVIDIA avec TensorRT, 50 épisodes). Nouveau ici : le
Neural Engine, les codages Core ML, la charge de vision, l'énergie par composant.

## Plan

- Tâche PushT, les 50 épisodes de la sonde (EXP-002), graine de la CEM 42 pour toutes les configurations (mêmes tirages ;
  seuls les calculs changent), CEM 300 x 30, horizon 5, tout le plan exécuté. LeWM publié, pleine précision des poids
  avant codage.
- Configurations sans charge : `torch` (float32, GPU), `torch-fp16` (autocast float16, GPU), `coreml-gpu` et `coreml-ne`
  en 16 bits (prédicteur en Core ML, le reste en PyTorch sur le GPU, comme EXP-012), et `coreml-gpu`, `coreml-ne` avec les
  codages `int8`, `int4b32`, `pal4`, et `ne-complet` (amendement 1) : 11 configurations.
- Configurations avec charge de vision (`charge_vision.py` : ResNet-50 à poids aléatoires, 224 x 224, lot de 1, float16,
  en boucle sur le GPU, démarré 30 s avant le calcul) : `torch`, `coreml-gpu`, `coreml-ne`, `ne-complet` en 16 bits, et la charge seule
  pendant 60 s (sa vitesse de référence, placée au hasard dans l'ordre comme une configuration) : 3 configurations et une
  mesure de référence (4 configurations avec l'amendement 1).
- Trois répétitions ; dans chaque répétition, ordre des configurations tiré au hasard (`default_rng(13)`), pour que
  l'ordre et la chaleur ne se confondent pas avec la configuration. Avant chaque répétition : énergie au repos par
  composant (`repos_puce.py`, 45 s de stabilisation puis 75 s de mesure, aucun calcul en cours ; moins de 3 min sans
  calcul, pour que la garde de batterie ne croie pas le calcul arrêté).
- Environ 40 runs de 50 épisodes (1 min environ chacun), soit environ 1 h avec les mesures au repos.

## Mesures (descriptives, aucun test d'hypothèse)

- Temps par plan : somme des durées des appels de solve sur le nombre de plans, par run (appels allongés par une pause
  de la garde écartés, plus de 3 fois la médiane) ; résumé par configuration : médiane des 3 répétitions, minimum et
  maximum.
- Énergie par plan : énergie de la puce par composant (CPU, GPU, Neural Engine, mémoire ; compteurs IOReport, sans droits
  administrateur) sur la durée du run, moins l'énergie au repos de la répétition sur la même durée, divisée par le nombre
  de plans. Le run comprend la simulation et le rendu de l'environnement, identiques entre configurations. Avec charge,
  l'énergie mesurée est celle de la planification et de la vision ensemble.
- Vision : images par seconde de la charge pendant le run, contre sa vitesse seule.
- Mémoire : taille des poids Core ML de chaque codage (octets de weight.bin) et du modèle PyTorch.
- Réussite des 50 épisodes de chaque run : contrôle seulement (les codages ne sont pas évalués en réussite ici).

Comparaisons annoncées : `coreml-ne` contre `torch` et `coreml-gpu` (temps et énergie, sans charge) ; avec charge,
temps par plan et images par seconde de la vision pour les trois moteurs ; temps, énergie et taille par codage.
Formulations : « sur un MacBook Pro M5 Max, macOS 26.6.2 » ; énergie de la puce, pas de tout le Mac.

## Contrôles bloquants

Au lancement : arbre propre, commit unique ; le run `torch` de chaque répétition et un run `torch` final doivent être
identiques octet pour octet entre eux et au pilote d'EXP-012 (`resultats/pilote-exp012/torch/fp32_s42.npz`, mêmes 50
épisodes, même graine ; seuls les champs git diffèrent). Pour chaque run Core ML : toutes les opérations sur le processeur demandé (plan de calcul). Mesure au repos :
aucun calcul run_grid pendant la mesure. La charge de vision doit écrire au moins une mesure pendant chaque run avec charge.
