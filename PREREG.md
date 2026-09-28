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
