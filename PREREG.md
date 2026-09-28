# Pré-enregistrement d'EXP-003 (figé par le commit qui contient ce fichier)

Rédigé par Claude le 28 septembre 2026, à la demande de Thomas (lancer les calculs avant la
compréhension complète), avant tout résultat de la grille. Thomas peut ajouter des hypothèses avant
l'analyse : elles seront marquées « ajoutées le JJ/MM, avant analyse ». Tout ce qui est décidé après
avoir vu les résultats est exploratoire.

## Question

Quand on quantifie les poids de LeWM, qu'est-ce qui casse la planification sur PushT, une fois le
bruit du planificateur retiré ? Et une mesure continue de l'état (erreur de pose du T) voit-elle la
dégradation avant le taux de réussite ?

## Plan

- Modèle : LeWM PushT publié (`quentinll/lewm-pusht`), protocole d'évaluation de `le-wm/eval.py`
  (but à 25 pas, budget de 50 pas, CEM 300 candidats, 30 itérations, horizon 5, frameskip 5).
- Épisodes : 500, tirés comme `eval.py` avec la graine 42, les mêmes pour toutes les versions, en 10
  paquets de 50 de composition fixe.
- Graines de la CEM : 42, 43, 44, 45, 46, les mêmes pour toutes les versions (appariement complet).
- Versions (16) : pleine précision ; quantification des poids par arrondi au plus proche, symétrique,
  une échelle par canal de sortie, à 8, 6, 5, 4 et 3 bits, sur tout le modèle (« all », 109 couches),
  l'encodeur seul (73) ou le prédicteur seul (30). Activations en pleine précision.
- Unité d'analyse : l'épisode. Pour chaque épisode e et version v, p(e, v) = part des 5 graines où
  l'épisode réussit.

## Métriques

- M1, principale : taux de réussite moyen (500 épisodes, 5 graines).
- M2 : erreur de position minimale du T sur l'épisode, distance entre la position du T et celle du
  but, en pixels du simulateur, minimum sur les pas de l'épisode, moyenne sur les graines.
- M3 : erreur d'angle minimale du T, en radians, même construction.
- Bruit du planificateur : en pleine précision, part des épisodes « fragiles » (0 < p < 1) et taux de
  bascule entre deux graines.

## Hypothèses et tests

Tests appariés par épisode, Wilcoxon signé bilatéral, correction de Holm sur P1 à P3, α = 0,05.

- P1 : « all:4 » baisse M1 par rapport à la pleine précision (test sur p(e, all:4) moins p(e, fp32)).
- P2 : à 4 bits, l'encodeur est plus sensible que le prédicteur, sens attendu d'après Fu et al.
  (2602.02110) : p(e, predictor:4) moins p(e, encoder:4) est positif.
- P3 : à 5 bits sur tout le modèle (« all:5 »), M2 est significativement dégradée alors que M1 ne
  l'est pas. Réfutée si M2 n'est pas dégradée, ou si M1 l'est aussi.

## Exploratoire (sans test pré-enregistré)

- Courbes M1, M2, M3 selon les bits, par composant.
- Épisodes cassés par la compression : ceux dont p baisse d'au moins 0,6 entre la pleine précision et
  une version quantifiée. Les décrire par l'état exact : distance et angle initiaux au but, contacts,
  pas du premier succès en pleine précision.
- Comparer ces épisodes aux épisodes déjà fragiles en pleine précision.

## Tests de sanité

- Le code redonne EXP-002 épisode par épisode (pleine précision, graine 42, 50 épisodes) : vérifié le
  28 septembre 2026 avant ce commit.
- Cohérence du succès : un épisode réussi doit avoir, au pas du succès, une erreur de position sous 20
  et une erreur d'angle sous π/9 (seuils de l'environnement).
- Déterminisme : relancer une (version, graine) doit redonner le même fichier.

## Amendement 1 (28 septembre 2026, 19 h 40, avant toute analyse)

Motif : revue du code par deux relecteurs indépendants. Seul résultat vu de la grille abandonnée :
le taux de réussite agrégé de la pleine précision à la graine 42 (84,6 %), sous l'ancien protocole.
Aucune version quantifiée n'avait tourné.

1. **Hasard apparié par épisode.** Dans la config officielle, un générateur unique est tiré épisode
   après épisode, et le second plan (au pas 25) ne concerne que les épisodes pas encore réussis : le
   hasard reçu par un épisode dépendait donc des autres épisodes du paquet, et de la version. Désormais
   chaque plan d'un épisode reçoit un tirage fixé par (graine, position de l'épisode dans
   l'échantillon, numéro du plan). La réinitialisation de l'environnement reçoit aussi une graine fixe
   par position (sans elle, pymunk gardait la trace d'une position aléatoire). Tests passés avant ce
   commit : sur CPU, lancement répété identique et résultat indépendant de la taille des paquets ; sur
   le GPU du Mac, lancement répété identique. Le protocole ne reproduit donc plus `eval.py` tirage pour
   tirage, mais il reste équivalent en loi.
2. **Versions réduites à 10**, pour tenir le calcul (environ 10 min par version et par graine) :
   pleine précision ; « all » à 8, 6, 5, 4, 3 bits ; « encoder » et « predictor » à 4 et 3 bits. Plus
   une réplique de la pleine précision avec les graines 47 à 51, pour les distributions nulles.
3. **Métriques redéfinies.** Après un succès, l'environnement est gelé : on prolonge son dernier état
   jusqu'au pas 50. Avec d4 = distance (agent x, y, T x, y) au but, dT = distance du T seul, dA =
   distance de l'agent seul, dθ = écart d'angle du T ramené dans [0, π] :
   - M1 (inchangée) : taux de réussite ; p(e, v) = part des 5 graines où l'épisode réussit.
   - M2 : marge de succès, minimum sur les 50 pas de max(d4 / 20, dθ / (π / 9)). Marge < 1 équivaut au
     succès de l'environnement : M1 en est la version seuillée.
   - M3 : marge du T seul, minimum sur les 50 pas de max(dT / 20, dθ / (π / 9)).
   - M4 (descriptive) : marge de l'agent, minimum sur les 50 pas de dA / 20. Le succès de PushT exige
     que l'agent revienne près de son but : M3 et M4 séparent la physique du T du retour de l'agent.
   - Moyennes par épisode sur les 5 graines, puis sur les épisodes.
4. **Tests.** Test de permutation par inversion de signe sur la moyenne des différences appariées par
   épisode (10 000 tirages, graine 0), bilatéral sauf mention ; intervalle bootstrap à 95 % par
   épisode (10 000 tirages), descriptif. Remplace Wilcoxon (trop d'ex aequo et de zéros).
   - P1 : p(e, all:4) moins p(e, fp32) différent de 0.
   - P2 : p(e, predictor:4) moins p(e, encoder:4) différent de 0, sens attendu positif.
   - P3 remplacée par P3' : à « all:5 », (a) M1 équivalente à la pleine précision à ± 2 points près
     (deux tests unilatéraux, TOST, par permutation, α = 0,05) et (b) M2 plus grande que la pleine
     précision (unilatéral). P3' est un test d'intersection-union : sa p-valeur est le maximum des
     trois p-valeurs. Réfutée si l'une des trois conditions échoue.
   - Correction de Holm sur P1, P2, P3', α = 0,05.
5. **Message principal, pré-enregistré en descriptif.** On simule le protocole des études existantes :
   1 000 tirages d'une seule graine et de 50 épisodes parmi les 500. Pour chaque version, part des
   tirages où le signe de (réussite fp32 moins réussite de la version) est opposé à celui du plan
   complet, et part où un test de McNemar à 0,05 conclut autrement que le plan complet (P1 pour
   « all:4 », test équivalent pour les autres versions).
6. **Plancher de bruit.** Bascules entre fp32 à la graine a et la version v à la graine b, comparées
   aux bascules entre fp32 à la graine a et fp32 à la graine b, avec a différent de b (graines 42 à 46).
7. **Épisodes cassés par la compression.** Fragilité (0 < p < 1) classée sur la réplique fp32
   (graines 47 à 51). Baisse de p mesurée entre fp32 (42 à 46) et la version (42 à 46). Distribution
   nulle : même calcul entre fp32 (42 à 46) et la réplique (47 à 51). Seuil de baisse : 0,6.
8. **Unité.** La position 0 à 499 dans l'échantillon (6 épisodes du jeu y figurent deux fois, à des
   pas de départ différents : gardés et signalés).
9. **Stratification** (descriptive) : épisodes où le T doit bouger (marge du T au départ au moins 1)
   contre les autres.
10. **Limite connue de P2** : l'encodeur a 5,42 M poids quantifiés, le prédicteur 10,76 M. On rapporte
    l'erreur relative des poids par composant (exploratoire).

## Amendement 2 (28 septembre 2026, avant toute analyse)

Heure de référence : celle du commit qui introduit cet amendement. Les commits de revue qui le suivent
et précèdent la première lecture des fichiers de la grille en font partie ; la fiche EXP-003 donne le
commit final, celui qui fige l'analyse.

Motif : revue du pré-enregistrement amendé (`revues/2026-09-28_revue-prereg-amendement1.md`, 2 P0 et
7 P1), qui porte sur l'analyse et non sur le code qui tourne. Déjà vu à cette date : fp32 graine 42 sur
les 500 épisodes, 82,2 % (nouveau protocole, seule ligne du journal de la grille lue, à 19 h 54) ; fp32
graine 42 sous l'ancien protocole, 84,6 % (`resultats/exp003-bs1/`) ; les lancements de test fp32
graine 42 sur 4 à 50 épisodes (`sanite-*`, `couplage-*`, `mps-*`, `test-c5703db-*`) ; le pilote EXP-002
(50 autres épisodes, 6 versions). Aucune version quantifiée de la grille n'a été vue ; l'avancement est
suivi au nombre de fichiers. L'amendement 1 dit 19 h 40, son commit `c5703db` est de 19 h 36 (heure de la
machine) : c'est le commit qui fait foi. Les sections initiales « Métriques », « Hypothèses et tests » et
le seuil exploratoire de 0,6 sont remplacés ; les points 5, 6 et 7 de l'amendement 1 sont remplacés ou
précisés par les points 9, 10 et 11 ci-dessous.

1. **Code.** `analyse_exp003.py` applique exactement cet amendement ; `test_analyse_exp003.py` le teste
   sur des grilles simulées (effet nul, effets connus, fichiers corrompus). Tout calcul ajouté après la
   première lecture des fichiers de la grille est exploratoire et vit dans un autre script.
2. **Contrôles bloquants.** 55 fichiers (10 versions × graines 42 à 46, fp32 × graines 47 à 51), sans
   doublon. Pour chaque fichier : arbre git propre au lancement ; commit `c5703db`, ou un commit
   postérieur où `run_grid.py`, `quant.py` et `lancer_grille.sh` sont identiques à `c5703db` (la réplique
   démarre après le commit de cet amendement) ; positions, épisodes, pas de départ et taille des paquets
   identiques à `meta.json` ; 50 pas ; état de départ et but identiques entre fichiers. Pour chaque course
   (épisode, version, graine) : succès si et seulement si un pas de premier succès est noté ; aucun NaN
   dans l'état jusqu'au premier succès, NaN ensuite ; M2 et M3 finies et strictement positives ;
   M2 < 1 si et seulement si succès, sauf les courses où |M2 - 1| ≤ 1e-4 (état enregistré en float32,
   écart jusqu'à 1,5e-5 avec l'état interne de l'environnement), comptées et rapportées. Minima sur les
   50 pas enregistrés, état initial exclu.
3. **Arithmétique.** K(e, v) = succès sur les graines 42 à 46 ; x(e) = K(e, v) - K(e, réf), entier de
   -5 à 5. Inversion de signe : 10 000 vecteurs de signes tirés une fois par
   `numpy.random.default_rng(0)` (forme 10 000 × 500), réutilisés partout ; un sous-échantillon de n
   épisodes prend les n premières colonnes. p bilatérale = (1 + nombre de |Σ s x| ≥ |Σ x|) / 10 001 ;
   unilatérale, même forme dans le sens de l'alternative ; égalités comprises. Calcul en float64 sur des
   entiers, vérifié identique au calcul en entiers. Une différence non finie arrête l'analyse.
4. **P1 et P2**, bilatérales. P1 soutenue si p ajustée (Holm) < 0,05 et Σ x < 0 ; P2 soutenue si p
   ajustée < 0,05 et Σ [K(e, predictor:4) - K(e, encoder:4)] > 0. Un effet significatif de sens
   contraire contredit l'hypothèse.
5. **P3'(a)**, marge 0,02, soit un épisode sur 50, la résolution du protocole officiel de LeWM (50
   épisodes, une graine) : y(e) = 10 x(e) + 1, p_bas = (1 + nombre de Σ s y ≥ Σ y) / 10 001 ;
   z(e) = 10 x(e) - 1, p_haut = (1 + nombre de Σ s z ≤ Σ z) / 10 001. Contrôle rapporté, sans valeur de
   décision : TOST t apparié et intervalle à 90 %.
6. **P3'(b) porte sur M3**, la marge du T seul, en log : w(e) = moyenne sur g de log M3(e, all:5, g)
   moins moyenne sur g de log M3(e, fp32, g), unilatérale « plus grand ». Motif : la question
   pré-enregistrée porte sur l'erreur de pose du T, et M2 contient le retour de l'agent. Choix de Claude,
   à confirmer par Thomas avant la lecture des résultats. p(P3') = max(p_bas, p_haut, p_b). P3' établie si
   sa p ajustée (Holm) < 0,05, non établie sinon. M2 devient descriptive, même transformation.
7. **Descriptifs.** Par version : taux de réussite ; écart à fp32 avec intervalle bootstrap à 95 %
   (percentile, 10 000 rééchantillons des 500 positions tirés par `default_rng(2)`, sur les différences
   appariées par épisode) ; de même pour log M3 et log M2. Décomposition par graine en classes SS, SF,
   FS, FF (fp32 d'abord, la version ensuite) : nombre de courses et part de chaque classe dans l'écart
   moyen de log M3 et de log M2 ; pour SS, écart moyen du pas du premier succès.
8. **M4 remplacée.** Chaque course est classée : réussite ; T placé sans réussite (M3 < 1) ; T jamais
   placé (M3 ≥ 1). Parts par version, aussi dans la strate « le T doit bouger » (marge du T à l'état de
   départ, copie du jeu de données, au moins 1), où l'écart de réussite et l'écart de log M3 ont leur
   propre bootstrap (`default_rng(3)`). Les affirmations sur la physique du T se limitent à cette strate.
9. **Point 5 remplacé : protocoles simulés.** Réglages (n épisodes, m graines) : (30, 1), (50, 1),
   (500, 1), (30, 5), (50, 5). (50, 1) est le protocole officiel de LeWM (`le-wm/config/eval/pusht.yaml` :
   50 épisodes, graine 42) ; n = 30 est l'effectif apparié de Where Bits Matter (2602.11882 : 10 épisodes
   × 3 graines au budget bA) ; Fu et al. (2602.02110) suivent le protocole de DINO-WM sans donner le
   nombre d'épisodes dans leur PDF : pas de réglage tiré d'eux. (500, 1) isole le bruit du planificateur,
   (50, 5) isole le tirage des épisodes. 1 000 tirages par `default_rng(1)` : par tirage, n positions
   sans remise, puis un indice de graine côté version et un côté réplique (tirés même si m = 5).
   Variante appariée : fp32 et la version aux mêmes graines (42 à 46). Variante non appariée : fp32 dans
   47 à 51, la version dans 42 à 46. Comparaisons : fp32 contre chacune des 9 versions ; encoder:b contre
   predictor:b (b = 4, 3), appariée seulement ; nul vrai, fp32 (47 à 51) contre fp32 (42 à 46). Signe de
   référence : celui du plan complet. Parts de même signe, de différence nulle, de signe contraire ;
   test à 0,05 (McNemar exact bilatéral si m = 1, inversion de signe sur K si m = 5) : conclut au même
   sens, ne conclut pas, conclut au sens contraire. Le message ne s'appuie que sur les versions dont
   l'intervalle du plan complet exclut 0 ; les autres sont rapportées. Part du planificateur :
   σ²_plan = moyenne sur e de la variance entre graines des différences par graine ; σ²_ep = variance
   entre épisodes de la différence moyenne, moins σ²_plan / 5 ; on rapporte
   σ²_plan / (σ²_plan + max(σ²_ep, 0)).
10. **Point 6 précisé : plancher de bruit.** 20 paires ordonnées a ≠ b parmi 42 à 46 ; perdus (succès de
    fp32 à la graine a, échec de v à la graine b) et gagnés séparés, par 50 épisodes ; attribution =
    pertes moyennes fp32 contre fp32 / pertes moyennes fp32 contre v. Aux mêmes graines (a = b) :
    descriptif. Épisodes discordants par 50 entre deux graines de fp32, avec intervalle bootstrap : ce
    chiffre remplace A-01 ; de même contre la réplique.
11. **Point 7 remplacé : épisodes cassés.** D_v(e) = K(e, fp32, 47 à 51) - K(e, v, 42 à 46) ;
    D_0(e) = K(e, fp32, 47 à 51) - K(e, fp32, 42 à 46) ; cassé si D ≥ 3. Strates de la réplique :
    K = 5, 1 à 4, 0. Nombres observés et nuls, au total et par strate ; test descriptif par inversion de
    signe de 1[D_v ≥ 3] - 1[D_0 ≥ 3], unilatéral (plus de cassés que le nul). Les baisses aux mêmes
    graines restent descriptives. Épisodes cassés, robustes (réplique 5/5, non cassés) et fragiles
    (réplique 1 à 4) décrits par les médianes de la distance et de l'angle initiaux au but, de la marge
    du T au départ, des contacts et du pas du premier succès en fp32.
12. La réplique n'entre jamais dans P1, P2, P3'.
13. Recouvrement avec les 50 épisodes d'EXP-002 : aucun (calculé sur les `meta.json`, avant tout
    résultat). Six épisodes du jeu figurent deux fois parmi les 500, à des pas de départ différents.

## Amendement 3 (28 septembre 2026, avant toute analyse)

Heure de référence : celle du commit qui l'introduit. Motif : revue du script d'analyse par deux
relecteurs indépendants (justesse statistique : 0 P0, 5 P1 ; intégration aux vrais fichiers : 1 P0,
2 P1). Aucun fichier de la grille n'a été lu ; les défauts ont été trouvés sur les fichiers de test fp32
graine 42 déjà déclarés vus et sur les données de protocole (états de départ et buts).

1. **M3 nulle.** Dans 70 des 500 épisodes, le T part exactement à sa pose cible (copie du jeu de
   données) ; s'il n'est pas touché, M3 vaut exactement 0 et log M3 n'existe pas. M3 = 0 est admise
   seulement pour ces épisodes, bloquante ailleurs ; M2 = 0 reste bloquante. Tous les logs de marge
   deviennent log(max(M, 0,05)) : 0,05 correspond à 1 px ou 1 degré, sous toute précision physique utile.
2. **P3'(b) dans la strate « le T doit bouger »** (marge du T au départ au moins 1 ; 328 épisodes,
   disjoints des 70 ci-dessus) : w(e) défini au point 6 de l'amendement 2, avec le plancher, moyenne sur
   ces 328 épisodes ; inversion de signe sur les 328 premières colonnes. Motif : hors de la strate, le T
   n'a pas à bouger et log M3 mesure surtout si l'agent l'effleure ; le point 8 limitait déjà les
   affirmations sur la physique du T à cette strate. P3'(a) reste sur les 500 épisodes. L'écart de
   log M3 sur les 500 épisodes reste rapporté, en descriptif.
3. **Courses divergentes.** Un échec dont l'état devient NaN sur ses 7 colonnes à un pas j, et le reste
   jusqu'au pas 50, compte comme échec ; ses marges sont prises sur les pas finis (sur l'état de départ si
   j = 0) ; leur nombre est rapporté par fichier. Tout autre motif de NaN reste bloquant.
4. **Contrôles précisés.** Un paquet peut s'arrêter avant 50 pas quand tous ses épisodes ont réussi
   (mode « wait » de stable-worldmodel) : admis si son nombre de pas vaut le dernier pas de premier
   succès plus 1. Le nombre de couches quantifiées doit valoir 0 (fp32), 109 (all), 73 (encoder) ou
   30 (predictor). Le commit de chaque fichier doit exister et descendre de `c5703db`. Le produit en
   float64 est comparé à chaque test au produit en entiers sur les 100 premiers vecteurs de signes.
5. **Protocoles simulés (point 9) : réglages remplacés par (50, 1), (500, 1), (10, 3), (50, 5).** (10, 3)
   est le protocole de Where Bits Matter (10 épisodes × 3 graines, test du signe sur les unités (graine,
   épisode)), à la place de (30, 1) qui ne l'était pas ; (30, 5) est retiré. Un seul flux
   `default_rng(1)`, réglages dans cet ordre ; par tirage : n positions sans remise, puis, si m = 3,
   3 graines sur 5 sans remise côté version et 3 côté réplique. Si m = 1, les graines ne sont pas tirées
   mais énumérées : 5 couples appariés, 25 non appariés, moyennés à poids égaux (le réglage (500, 1)
   devient un calcul exact). Test : McNemar exact bilatéral sur les courses (épisode, graine) si m = 1
   ou 3 ; inversion de signe sur K si m = 5. Chaque comparaison porte deux indicateurs calculés par le
   script : « retenue » (intervalle bootstrap à 95 % du plan complet qui exclut 0 ; seules ces lignes
   portent le message) et « stable » (le signe du plan complet ne change pas quand on retire l'une des
   5 graines).
6. **Part du planificateur** rapportée deux fois : appariée (mêmes graines, borne basse, puisque le
   hasard est partagé) et non appariée (version aux graines 42 à 46 contre la réplique 47 à 51, rang à
   rang).
7. **Recouvrement avec EXP-002 (point 13), précisé** : aucun couple (épisode, pas de départ) commun ;
   9 épisodes du jeu figurent dans les deux échantillons, à d'autres pas de départ.
8. **Tests ajoutés** (`test_analyse_exp003.py`) : T au but au départ, succès au pas 0, paquet arrêté
   tôt, courses divergentes, version identique à fp32 (différence toujours nulle en apparié), part du
   planificateur sur des composantes de variance connues, sens « moins » contre l'énumération exacte ;
   refus d'un nombre de couches faux, d'un commit inconnu, d'un paquet arrêté tôt sans succès complet,
   d'une M3 nulle hors départ au but.
