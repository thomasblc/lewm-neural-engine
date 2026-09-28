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
