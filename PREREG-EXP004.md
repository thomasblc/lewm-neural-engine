# Pré-enregistrement d'EXP-004 : couches restantes et quantification par groupes (LeWM, PushT)

Rédigé par Claude le 28 septembre 2026 au soir, avant tout lancement d'EXP-004 et avant l'analyse
complète d'EXP-003. Figé par le commit qui contient ce fichier et `analyse_exp004.py` ; les commits de
revue qui précèdent le lancement en font partie (commit final noté dans la fiche EXP-004). Décision de
Thomas : plan d'expériences (`experiences/PLAN-EXPERIENCES.md`, questions Q3 et Q5), accepté le 28 septembre
à 23 h 25 (« ok on fait ça »).

## D'où viennent les questions

- Q3 vient du coup d'œil intermédiaire d'EXP-003 (graine 42 seulement, déclaré dans la fiche EXP-003) :
  tout le modèle à 4 bits perd 13,4 points, l'encodeur seul 5,4, le prédicteur seul en gagne 1,4. Les
  6 autres couches (encodeur d'action, projecteur, projection du prédicteur : 1,73 M poids, 10 % des poids
  quantifiés) sont quantifiées dans « all » et dans aucune des deux autres versions.
- Q5 vient du plan : la quantification par groupes est celle des déploiements réels. Mesuré avant tout
  lancement (`test_quant.py`) : erreur relative des poids de tout le modèle de 10,98 % à 9,54 % à 4 bits,
  de 25,64 % à 22,29 % à 3 bits.
- Déjà vu d'EXP-003 au moment de figer : la graine 42 des 10 versions et fp32 graine 43. Rien d'autre.

## Plan

- Protocole identique à EXP-003 : `run_grid.py` du commit `c5703db`, inchangé ; mêmes 500 épisodes, mêmes
  graines (42 à 46), hasard par épisode, paquets de 50, GPU du Mac. `quant.py` étendu (groupes « autres »
  et « encpred », groupes de 64) ; vérifié avant tout lancement : pour les 9 versions d'EXP-003, les poids
  sont identiques à ceux du `quant.py` de `c5703db`.
- Versions nouvelles (5) : `autres:4`, `encpred:4`, `all:4:g64`, `autres:3`, `all:3:g64`. 5 graines ×
  500 épisodes : 25 lancements, graine par graine, dans cet ordre de versions.
- Références reprises d'EXP-003 sans les recalculer : fp32, `all:4`, `all:3`, `encoder:4`, `predictor:4`
  (graines 42 à 46). La réplique fp32 (47 à 51) n'entre dans aucun test.
- Groupes de 64 poids consécutifs d'une ligne ; deux couches restent quantifiées par canal, faute de
  divisibilité : la projection des images en carrés (588 entrées) et une couche de l'encodeur d'action
  (10 entrées).
- **Sanité bloquante, lancée en premier** : `all:4` graine 42 recalculé avec le code d'EXP-004 doit être
  identique au fichier d'EXP-003 (état, pose du T, position de l'agent, contacts, succès, pas du premier
  succès, nombre de pas). Sinon, rien n'est analysé : les références d'EXP-003 ne seraient plus comparables.

## Hypothèses et tests

Test par inversion de signe de l'amendement 2 d'EXP-003, point 3 (mêmes 10 000 vecteurs de signes tirés
par `default_rng(0)`, p = (1 + nombre de |Σ s x| ≥ |Σ x|) / 10 001), bilatéral, sur x(e) différence des
nombres de succès sur les 5 graines. Correction de Holm sur les quatre, α = 0,05. Soutenue si la p
ajustée < 0,05 et si la somme a le signe attendu ; contredite si elle est significative dans l'autre sens ;
non soutenue sinon.

- **Q3a** : les autres couches seules à 4 bits baissent la réussite : Σ [K(autres:4) - K(fp32)] < 0.
- **Q3b** : à 4 bits, laisser les autres couches en pleine précision relève la réussite :
  Σ [K(encpred:4) - K(all:4)] > 0.
- **Q5a** : les groupes de 64 relèvent la réussite à 4 bits : Σ [K(all:4:g64) - K(all:4)] > 0.
- **Q5b** : de même à 3 bits : Σ [K(all:3:g64) - K(all:3)] > 0.

## Descriptif, sans test

- Réussite par version et écart à fp32, intervalle bootstrap percentile à 95 % (10 000 rééchantillons des
  500 positions, `default_rng(2)`) ; écart de log M3 (plancher 0,05) dans les 328 épisodes où le T doit
  bouger (`default_rng(3)`) ; classes de courses. Mêmes définitions qu'EXP-003 (amendements 2 à 4).
- Additivité à 4 bits : I(e) = [K(all:4) - K(encoder:4) - K(predictor:4) - K(autres:4) + 2 K(fp32)] / 5,
  l'écart de tout le modèle moins la somme des écarts des trois parties ; J(e) = [K(encpred:4) -
  K(encoder:4) - K(predictor:4) + K(fp32)] / 5, la même chose pour l'encodeur et le prédicteur ensemble.
  Moyenne et intervalle bootstrap.
- À 3 bits : écart d'`autres:3` à fp32 et à `all:3`.

## Contrôles bloquants

Ceux d'EXP-003 (amendements 2 à 4) pour chaque course, par la fonction `controler_course` du script
d'EXP-003. Pour chaque fichier d'EXP-004 : 25 fichiers attendus, sans doublon ; arbre git propre au
lancement ; commit existant dont `run_grid.py` est identique à celui de `c5703db` et dont `quant.py` a
l'empreinte SHA-256 `2c450c5b06ebd6753e77ff9c189fffc339b655070efaf32e103f13ad5e29c65d` ; nombre de couches
quantifiées 6 (autres), 103 (encpred) ou 109 (all) ; épisodes, pas de départ, états de départ et buts
identiques à EXP-003. Les fichiers d'EXP-003 repris passent aussi les contrôles d'EXP-003.

## Aveugle

`run_grid.py` écrit le taux de réussite de chaque lancement dans le journal : il n'est pas lu avant
l'analyse ; l'avancement se suit au nombre de fichiers.
