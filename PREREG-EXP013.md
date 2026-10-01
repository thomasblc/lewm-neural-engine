# Pré-enregistrement d'EXP-013 : vitesse, énergie et mémoire de la planification de LeWM selon le moteur, le codage des poids et une charge de vision

Brouillon rédigé par Claude le 1er octobre 2026 à partir de 18 h 02 (heure du Mac, lue avec `date`), avant tout calcul
d'EXP-013. Figé par le commit enregistré dans les fichiers. Les mesures d'EXP-012 (temps par plan, 50 épisodes de pilote,
journal d'énergie par composant pendant TwoRoom) sont connues et ont motivé ce plan ; aucun fichier d'EXP-013 n'existe.

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
