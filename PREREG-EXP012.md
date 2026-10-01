# Pré-enregistrement d'EXP-012 : le prédicteur en Core ML 16 bits, sur le GPU et sur le Neural Engine

Rédigé par Claude le 1er octobre 2026 à partir de 12 h 27 (heure du Mac, lue avec `date`), avant tout calcul
d'EXP-012. Déjà calculés, hors du dossier de résultats de l'expérience : les tests de `test_coreml_pred.py` (écart de
sortie à PyTorch float32 de 1,3e-3 à 2,4e-3 sur le GPU et de 2,7e-3 à 8,3e-3 sur le Neural Engine, PushT et
TwoRoom, T = 1 à 3) et un pilote de 50 épisodes de PushT, graine 42, sur les trois moteurs (`resultats/pilote-exp012/`,
réussite et temps lus, pour vérifier que le calcul tourne). Figé par le commit enregistré dans les fichiers.

## Question

Faire tourner le prédicteur de LeWM (prédicteur et pred_proj) en Core ML, en 16 bits, sur le GPU ou sur le Neural
Engine, change-t-il la réussite du planificateur ? C'est la condition préalable à tout déploiement sur le Neural
Engine. La vitesse et l'énergie sont mesurées ici à titre descriptif : dans ce montage, l'encodeur, la CEM et le coût
restent en PyTorch sur le GPU (un planificateur entièrement hors du GPU est l'objet de l'expérience de vitesse).

## Plan

- Moteurs : `coreml-gpu` (graphe d'origine, CPU_AND_GPU) et `coreml-ne` (graphe réécrit, lot en dernier axe,
  `predicteur_ane.py`, CPU_AND_NE) ; mlprogram, float16, macOS 15 minimum ; formes fixes (300, 3), séquences plus
  courtes complétées puis recoupées ; conversion au lancement depuis les poids du modèle publié, empreinte des poids
  Core ML dans chaque fichier. macOS 26.6.2, coremltools 9.0.
- Version du modèle : pleine précision (aucune quantification) ; le calcul du prédicteur se fait en 16 bits.
- PushT sur les 500 épisodes d'EXP-003, TwoRoom sur les 500 d'EXP-006 ; graines 42 à 46 ; référence appariée : la
  pleine précision PyTorch float32 d'EXP-003 et d'EXP-006. 20 fichiers, environ 3 h.

## Tests (4 tests de non-infériorité, Holm, inversion de signe comme EXP-003, unilatéraux)

K(e, v) : nombre de succès de l'épisode e sur les 5 graines ; x(e) = K(e, moteur) - K(e, fp32 PyTorch).

- N1 (PushT, Neural Engine), N2 (TwoRoom, Neural Engine), N3 (PushT, GPU), N4 (TwoRoom, GPU) : le moteur perd moins
  de 2 points de réussite. Statistique décalée y = 20 x + 2 (moyenne positive si et seulement si l'écart moyen de
  réussite dépasse -2 points), sens « plus ».

Soutenue si la p corrigée par Holm (sur les 4) est sous 0,05. Marge de 2 points : demi-intervalle à 95 % attendu de
0,9 à 1,3 point pour un effet nul (prédicteur à 4 bits dans EXP-003 et EXP-006).

Descriptifs : réussites et écarts à la pleine précision avec intervalles bootstrap (`default_rng(2)`) ; écart du
Neural Engine au GPU ; temps par plan (médiane et quartiles, fichiers `.temps.json`), à côté de celui de PyTorch
mesuré sur la reproduction de la graine 42 ; puissance du Mac (médiane de SystemLoad) et énergie par plan estimée
(puissance médiane fois temps médian), mesure grossière de tout le système, batterie en charge ou non.

## Contrôles bloquants

Avant la grille, par tâche : reproduction de la pleine précision PyTorch graine 42 (identique octet pour octet à
EXP-003 ou EXP-006, et référence de la sentinelle). À l'analyse : tous les fichiers ; protocole identique à celui de
la référence, clé `moteur` exceptée, et moteur conforme ; mêmes épisodes, départs et buts que la référence ; aucune
couche quantifiée ; même empreinte des poids Core ML pour toutes les graines d'un moteur et d'une tâche ; contrôles
de course ; un seul commit dont le code est celui du dépôt ; arbre propre ; reproduction du même commit. La
sentinelle contrôle chaque fichier pendant le calcul. Le journal n'est pas lu avant l'analyse.
