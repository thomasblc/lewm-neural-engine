# Pré-enregistrement d'EXP-005 : à 6 bits, la pose du T se dégrade-t-elle sans que la réussite baisse ?

Rédigé par Claude le 29 septembre 2026 au matin, à la demande de Thomas (« prépare l'exp 5 et je te dirai
quand on la lancera »), avant tout calcul d'EXP-005. Figé par le commit enregistré dans les fichiers
d'EXP-005 : le lanceur vérifie avant chaque étape que l'arbre est propre et que HEAD n'a pas bougé depuis son
démarrage. Tout changement ultérieur est un amendement déclaré.

## D'où vient la question

- C'est l'idée du plan d'origine de Thomas (« Does Compression Break Physics? ») : une mesure de l'état
  exact voit la dégradation avant le taux de réussite. EXP-003 l'a testée à 5 bits (P3'), sans succès : à
  5 bits, la réussite baisse déjà de 3,0 points.
- Exploratoire dans EXP-003 (vu après l'analyse, donc non affirmable) : à 6 bits, dans les 328 épisodes où
  le T doit bouger, la réussite ne bouge pas (+0,2 point [-1,3 ; +1,6]) et la pose du T se dégrade un peu
  (écart de log M3 +0,030 [+0,003 ; +0,059]). EXP-005 teste ce constat sur de nouvelles données.
- Déjà vu : tous les résultats d'EXP-003 ; les 2 premiers fichiers d'EXP-004 n'ont pas été lus. Aucun
  épisode d'EXP-005 n'a été joué.

## Plan

- Code : `run_grid.py` d'EXP-003 plus deux options de tirage (`--graine-episodes`, `--exclure`). Sans elles,
  le calcul est identique à `c5703db` (vérifié par `test_run_grid.py` : fichier de test de 10 épisodes
  sur CPU identique octet pour octet). `quant.py` d'EXP-004, inchangé pour les versions d'EXP-003.
- Épisodes : 1 000 nouveaux épisodes, même protocole d'évaluation (but à 25 pas, budget de 50 pas, CEM 300
  × 30, horizon 5), tirés avec la graine 2026 en écartant les 550 couples (épisode, pas de départ)
  d'EXP-003 et d'EXP-002. Vérifié avant tout calcul : aucun couple en commun, tirage reproductible ;
  653 épisodes où le T doit bouger, 120 où il part au but ; 37 épisodes du jeu figurent deux fois, à des
  pas de départ différents.
- Versions : pleine précision (`fp32`) et tout le modèle à 6 bits (`all:6`, RTN par canal). Graines de la
  CEM : 42 à 46. 10 lancements de 1 000 épisodes, graine par graine (environ 18 min chacun, 3 h au total).
- Sanité de clôture, bloquante : `all:6` graine 42 recalculé à la fin doit redonner le fichier de la
  grille (état, pose, agent, contacts, succès, pas du premier succès, nombre de pas).

## L'hypothèse (une seule, pas de correction multiple)

**H6** : à 6 bits, (a) la réussite est équivalente à la pleine précision à ± 2 points près, sur les
1 000 épisodes, et (b) la pose du T est plus mauvaise, dans les 653 épisodes où le T doit bouger.

- x(e) = K(e, all:6) - K(e, fp32), succès sur les 5 graines. Test par inversion de signe comme EXP-003
  (10 000 vecteurs de signes tirés par `default_rng(0)`, ici de forme 10 000 × 1 000 ; p = (1 + nombre de
  sommes permutées au moins aussi extrêmes, égalité comprise) / 10 001).
- (a) TOST : y = 10 x + 1, p_bas = (1 + nombre de Σ s y ≥ Σ y) / 10 001 ; z = 10 x - 1,
  p_haut = (1 + nombre de Σ s z ≤ Σ z) / 10 001. Contrôle rapporté sans valeur de décision : TOST t apparié et
  intervalle à 90 %.
- (b) w(e) = moyenne sur les graines de log(max(M3, 0,05)) pour all:6, moins la même chose pour fp32, sur
  les 653 épisodes de la strate (653 premières colonnes des signes), unilatéral « plus grand ».
- p(H6) = max(p_bas, p_haut, p_b). H6 établie si p(H6) < 0,05, non établie sinon.
- Puissance : avec l'écart vu dans EXP-003 (+0,030), environ 90 % ; si l'écart réel est moitié moindre,
  ce qui arrive souvent quand on reteste un constat choisi après coup, environ 45 %. Une H6 non établie ne
  prouvera donc pas l'absence d'effet.

## Descriptif, sans test

- Réussite par version ; écarts de réussite, de log M3 et de log M2 avec intervalle bootstrap percentile
  (10 000 rééchantillons, `default_rng(2)` sur les 1 000 épisodes, `default_rng(3)` sur la strate) ;
  classes de courses (réussite, T placé sans réussite, T jamais placé) sur tous les épisodes et dans la
  strate ; plancher de bruit (épisodes discordants par 50 entre deux graines de fp32).
- Comparaison avec EXP-003 : les deux écarts à 6 bits côte à côte, sans les regrouper.

## Contrôles bloquants

Pour chaque course, ceux d'EXP-003 (`controler_course`). 10 fichiers attendus (fp32 et all:6, graines 42 à
46), sans doublon ; arbre git propre au lancement ; commit existant dont `run_grid.py` a l'empreinte
SHA-256 `31e665344fbc5a312a7714d7252bde9eae2c931e65e31cdfc1b2f3c5af513fcc` et `quant.py`
`2c450c5b06ebd6753e77ff9c189fffc339b655070efaf32e103f13ad5e29c65d` ; 0 ou 109 couches quantifiées ;
1 000 épisodes, graine de tirage 2026, liste d'exclusion égale aux couples d'EXP-003 et d'EXP-002, aucun
couple en commun avec eux ; mêmes épisodes, états de départ et buts dans tous les fichiers ; 653 et 120
épisodes dans les strates ; même protocole et même torch pour la grille et la sanité de clôture ; sanité
de clôture identique.

## Aveugle

Le journal du lancement écrit le taux de réussite de chaque fichier : il n'est pas lu avant l'analyse.
