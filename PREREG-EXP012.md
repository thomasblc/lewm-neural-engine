# Pré-enregistrement d'EXP-012 : le prédicteur en Core ML 16 bits, sur le GPU et sur le Neural Engine

Rédigé par Claude le 1er octobre 2026 à partir de 12 h 27 (heure du Mac, lue avec `date`), avant tout calcul
d'EXP-012. Déjà calculés, hors du dossier de résultats de l'expérience : les tests de `test_coreml_pred.py` (écart de
sortie à PyTorch float32 de 1,3e-3 à 2,4e-3 sur le GPU et de 2,7e-3 à 8,3e-3 sur le Neural Engine, PushT et
TwoRoom, T = 1 à 3) et un pilote de 50 épisodes de PushT, graine 42, sur les trois moteurs (`resultats/pilote-exp012/`,
réussite et temps lus, pour vérifier que le calcul tourne). Figé par le commit enregistré dans les fichiers.

Amendement 1 (1er octobre, 13 h 37, avant tout calcul de l'expérience ; lanceur pas encore démarré). Suite à deux
relectures scientifiques (statistique ; protocole et reproductibilité) :
- Décision de non-infériorité : soutenue seulement si la p par inversion de signe (inchangée) et la p d'un
  bootstrap-t (épisodes rééchantillonnés, B = 10 000, `default_rng(4)`) sont toutes deux sous le seuil ; Holm sur la
  plus grande des deux. Raison : le test par inversion de signe de la statistique décalée n'est pas exact et penche
  vers « non inférieur » (4,7 à 6,3 % pour 5 % nominal en simulation). Erreur de Monte-Carlo de la p rapportée ; borne
  inférieure unilatérale de l'écart rapportée au niveau de Holm de chaque test.

Amendement 2 (1er octobre, 13 h 55, avant tout calcul ; relecture des corrections) :
- Puissance : les lectures sont coupées là où le fil de lecture a été gelé (pause de la garde, plus de 2,5 pas sans
  lecture) ; seules comptent les mises à jour du capteur faites pendant le calcul, pauses exclues.
- Repos : une mesure par lancement, nommée par son heure, jamais écrasée ; chaque fichier est associé à la dernière
  mesure faite avant lui (même commit, au moins 50 s couvertes, aucun calcul run_grid pendant la mesure).
- Borne inférieure : par bootstrap-t (même rééchantillonnage que la p), aux niveaux de la procédure de Holm (après le
  premier test gardé, tous les suivants au niveau de ce test) ; tout désaccord entre la borne et le verdict est signalé
  (le verdict suit la règle des deux p). Erreur de Monte-Carlo rapportée pour les deux p, dans le rapport.
- Temps par plan : somme des durées des appels de solve, divisée par le nombre de plans (pas la durée du fichier).
- Reproductions finales d'un lancement précédent effacées au lancement (la garde de batterie ne sort pas trop tôt).
- Les descriptifs de la section « Tests » ci-dessous sont remplacés par ceux des amendements 1 et 2.

Amendement 3 (1er octobre, 14 h 05, avant tout calcul) : le « contrôle négatif » de l'amendement 1 est un contrôle de
la marge (deux séries de la pleine précision, écart vrai nul : le test doit conclure), sans décision attachée. Valeurs
déjà connues avant l'expérience (EXP-003 et EXP-006) : PushT, écart +0,16 point, p 0,0006 (signe) et 0,0004
(bootstrap-t) ; TwoRoom, écart -0,92 point, p 0,0228 et 0,0249. Sur TwoRoom, deux séries de 5 graines de la même pleine
précision ne passeraient donc pas aux niveaux de Holm sous 0,025 : un N2 ou N4 « non soutenu » avec un écart estimé
au-dessus de -1 point se lit comme « non montré », pas comme une perte. Le lanceur n'attend plus celui d'EXP-011 (qui
l'attend), pour éviter une attente mutuelle.
- Contrôle négatif rapporté : la deuxième série de la pleine précision (graines 47 à 51, déjà calculée) contre la
  première, par le même test. Sensibilité rapportée : référence sur la moyenne des 10 graines. La décision reste sur
  la référence à 5 graines (choix d'origine).
- Ordre des calculs : moteurs alternés graine par graine (Neural Engine puis GPU à chaque graine), PyTorch en plus à
  la graine 43, pour mesurer les temps dans les mêmes conditions ; reproductions bloquantes PyTorch octet pour octet des
  graines 42 (au début), 43 (au milieu) et 46 (à la fin) de chaque tâche.
- Énergie : puissance moyenne lue sur le compteur cumulé des capteurs de la batterie (mises à jour faites pendant le
  calcul seulement), puissance au repos mesurée 130 s avant chaque tâche, énergie par plan au-dessus du repos ; temps
  par plan = temps total / nombre de plans par fichier, appels allongés par une pause de batterie écartés (plus de
  3 fois la médiane) ; conditions enregistrées (charge du système, batterie, état thermique).
- Contrôles ajoutés : toutes les opérations Core ML sur le processeur demandé (plan de calcul enregistré par fichier) ;
  empreinte du programme Core ML (spécification sans date) en plus de celle des poids, constante sur les graines.
- Une modification du code après une ANOMALIE de la sentinelle annule les fichiers du moteur concerné : tout est
  recalculé.
- Formulations : « perte de moins de 2 points, test unilatéral », pas « aucun changement » ; « identique octet pour
  octet sur la même machine et la même pile logicielle », pas « reproductible » en général ; l'écart Neural Engine
  moins GPU mêle le processeur et la réécriture du graphe (exacte en float32, 2,1e-6) et le 16 bits n'est pas isolé
  (pas de témoin PyTorch en 16 bits) : aucune affirmation générale sur le coût du 16 bits ou du Neural Engine.

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
  pleine précision PyTorch float32 d'EXP-003 et d'EXP-006. 20 fichiers et 6 reproductions, environ 4 h.

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
