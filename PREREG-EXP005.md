# Pré-enregistrement d'EXP-005 : à 6 bits, la pose du T se dégrade-t-elle alors que la réussite reste à ± 2 points ?

Rédigé par Claude le 29 septembre 2026 au matin, à la demande de Thomas (« prépare l'exp 5 et je te dirai
quand on la lancera »), avant tout calcul d'EXP-005. Taille choisie par Thomas : option C (« la plus
complète ») : 2 000 épisodes et un contrôle à 8 bits. Figé par le commit enregistré dans les fichiers
d'EXP-005 : le lanceur vérifie avant chaque étape que l'arbre est propre et que HEAD n'a pas bougé depuis son
démarrage, et l'analyse vérifie que son propre code et ce texte sont ceux de ce commit. Tout changement
ultérieur est un amendement déclaré. Version relue trois fois par deux relecteurs (première version `ef6ca16`,
corrections `35f312f` puis `a8e8cab`, puis ce texte).

## D'où vient la question

- C'est l'idée du plan d'origine de Thomas (« Does Compression Break Physics? ») : une mesure de l'état
  exact voit la dégradation avant le taux de réussite. EXP-003 l'a testée à 5 bits (P3'), sans succès : à
  5 bits, la réussite baisse déjà de 3,0 points.
- Exploratoire dans EXP-003 (vu après l'analyse, donc non affirmable) : à 6 bits, dans les 328 épisodes où
  le T doit bouger, la réussite ne bouge pas (+0,2 point [-1,3 ; +1,6]) et la pose du T se dégrade un peu
  (écart de log M3 +0,030 [+0,003 ; +0,059], p unilatérale 0,017). Repère de bruit dans la même strate :
  la réplique de la pleine précision (aucune compression, autres graines) donne +0,018 (p 0,095), et 8 bits
  +0,005. L'effet à 6 bits est donc proche du bruit et sans doute surestimé : d'où la taille choisie.
- Déjà vu : tous les résultats d'EXP-003 ; aucun fichier d'EXP-004 n'a été lu. Aucun épisode d'EXP-005 n'a
  été joué.

## Plan

- Code : `run_grid.py` d'EXP-003 plus deux options de tirage (`--graine-episodes`, `--exclure`). Sans elles,
  le calcul est identique à `c5703db` (vérifié par `test_run_grid.py` : un fichier de test de 10 épisodes
  sur CPU identique octet pour octet, protocole compris). `quant.py` d'EXP-004, inchangé pour les versions
  d'EXP-003.
- Épisodes : 2 000 nouveaux, même protocole d'évaluation (but à 25 pas, budget de 50 pas, CEM 300 × 30,
  horizon 5), tirés avec la graine 2026. Règle de tirage : toute trajectoire du jeu utilisée par EXP-003 ou
  EXP-002 est exclue entière (535 trajectoires, 550 couples), et on garde au plus une position par
  trajectoire, dans l'ordre du tirage : les 2 000 positions viennent de 2 000 trajectoires distinctes et
  sont indépendantes. Vérifié avant tout calcul : tirage reproductible ; aucune trajectoire commune ; 1 333
  épisodes où le T doit bouger, 233 où il part au but.
- Versions : pleine précision (`fp32`), tout le modèle à 6 bits (`all:6`) et, en contrôle, à 8 bits
  (`all:8`), RTN par canal comme EXP-003. Graines de la CEM : 52 à 56, neuves (EXP-003 et EXP-004 ont
  utilisé 42 à 51). 15 lancements de 2 000 épisodes, graine par graine (environ 37 min chacun, 9 h 15), plus
  la sanité de clôture (37 min).
- Sanité de clôture, bloquante : `all:6` graine 52 recalculé à la fin doit redonner le fichier de la
  grille (état, pose, agent, contacts, succès, pas du premier succès, nombre de pas).

## L'hypothèse (une seule, pas de correction multiple)

**H6** : à 6 bits, (a) la réussite reste à ± 2 points de la pleine précision, sur les 2 000 épisodes, et
(b) la pose du T est plus mauvaise, dans les 1 333 épisodes où le T doit bouger.

- x(e) = K(e, all:6) - K(e, fp32), succès sur les 5 graines. Test par inversion de signe comme EXP-003
  (10 000 vecteurs de signes tirés par `default_rng(0)`, ici de forme 10 000 × 2 000 ; p = (1 + nombre de
  sommes permutées au moins aussi extrêmes, égalité comprise) / 10 001).
- (a) TOST : y = 10 x + 1, p_bas = (1 + nombre de Σ s y ≥ Σ y) / 10 001 ; z = 10 x - 1,
  p_haut = (1 + nombre de Σ s z ≤ Σ z) / 10 001. Contrôle rapporté sans valeur de décision : TOST t apparié et
  intervalle à 90 %. La marge de ± 2 points reprend celle de P3' (EXP-003) et n'est pas changée.
- (b) w(e) = moyenne sur les graines de log(max(M3, 0,05)) pour all:6, moins la même chose pour fp32, sur
  les 1 333 épisodes de la strate (1 333 premières colonnes des signes), unilatéral « plus grand ».
- p(H6) = max(p_bas, p_haut, p_b). H6 établie si p(H6) < 0,05, non établie sinon.
- **Règle de rédaction.** Chaque intervalle bootstrap à 95 % de l'écart de réussite (sur les 2 000 épisodes
  et dans la strate) est classé : sous 0, contient 0, au-dessus de 0. Si H6 est établie : les deux
  contiennent 0, « la pose du T se dégrade à 6 bits alors que la réussite ne bouge pas » ; au moins un sous 0
  et aucun au-dessus, « la pose et la réussite se dégradent toutes deux, la réussite restant à ± 2 points sur
  l'ensemble des épisodes » ; au moins un au-dessus et aucun sous 0, « la pose se dégrade alors que la réussite
  augmente légèrement, en restant à ± 2 points sur l'ensemble des épisodes » ; un de chaque, « l'écart de
  réussite change de sens entre tous les épisodes et la strate ». La marge de ± 2 points n'est testée que sur
  les 2 000 épisodes ; dans la strate, l'intervalle est rapporté sans test. La phrase se complète par la progression selon les bits (section suivante). Les deux intervalles
  figurent dans le tableau de H6, avec la traduction de l'écart de pose en pixels et en degrés.
- Puissance (écart-type de w estimé sur EXP-003, 0,258 par épisode) : pour (b), environ 99 % si l'écart réel
  vaut celui d'EXP-003 (+0,030), environ 70 % s'il vaut la moitié ; pour (a), presque 1 (erreur type de
  l'écart de réussite d'environ 0,3 point sur 2 000 épisodes). Vérifié par étalonnage sur des
  pseudo-expériences tirées des données d'EXP-003 (`etalonnage_exp005.py`) : fausses alertes 4,6 % pour (b)
  sans effet, 4,95 % et 4,58 % pour les bornes de (a) exactement à ± 2 points ; pour H6 entière avec la
  réussite exactement à -2 points et la pose dégradée de +0,030, 5,6 % sur 1 000 tirages (erreur Monte-Carlo
  0,7 point, compatible avec 5 %) ; puissance 99,5 % et 70,0 %.
  Une H6 non établie ne prouvera pas l'absence d'effet. H6 n'a pas de taille d'effet minimale : on rapporte
  l'écart de pose en unités physiques pour juger de son importance.

## Contrôle à 8 bits (descriptif, sans test d'hypothèse)

Les mêmes statistiques qu'H6 pour all:8 contre fp32 (écarts, intervalles, p rapportées sans valeur de
décision). Progression selon les bits : d(e) = moyenne sur les graines de log(max(M3, 0,05)) à 6 bits moins
la même chose à 8 bits, dans la strate (la pleine précision s'annule) ; intervalle bootstrap apparié
(`default_rng(3)`). La décision se lit sur l'intervalle, classé comme ceux de la réussite : entièrement
au-dessus de 0, progression établie ; il contient 0, non établie ; entièrement sous 0, non établie, pose plus
mauvaise à 8 bits qu'à 6 bits. La p unilatérale par inversion de signe est rapportée sans valeur de décision.
Si H6 est établie, la phrase se termine selon ces trois cas par « la dégradation est plus faible à 8 bits
(progression selon les bits) », « mais l'effet à 6 bits ne se distingue pas du contrôle à 8 bits » ou « mais la
pose est plus mauvaise à 8 bits qu'à 6 bits (pas de progression selon les bits) ». La progression n'est
rapportée (rapport et fichier de résultats) que si H6 est établie. Puissance (écart-type de d estimé sur
EXP-003, 0,261 par épisode ; intervalle bilatéral à 95 %, soit un test unilatéral à 2,5 % ; approximation
normale) : environ 94 % pour une différence réelle de 0,025, environ 29 % pour 0,010.

## Descriptif, sans test

- Réussite par version ; écarts de réussite, de log M3 et de log M2 avec intervalle bootstrap percentile
  (10 000 rééchantillons, `default_rng(2)` sur les 2 000 épisodes, `default_rng(3)` sur la strate) ;
  classes de courses (réussite, T placé sans réussite, T jamais placé) sur tous les épisodes et dans la
  strate ; plancher de bruit (épisodes discordants par 50 entre deux graines de fp32).
- Pas de décomposition de l'écart de pose selon l'issue des courses (réussites, échecs). Deux versions ont été
  écartées avant le lancement : la répartition par paires d'issues à la même graine (SS, SF, FS, FF) dépend de
  l'appariement des graines ; la décomposition de Kitagawa, sans appariement, mélange la précision et la
  sélection des courses qui réussissent. Vérifié sur la pleine précision d'EXP-003 : basculer à l'échec les 16
  réussites les moins précises de la strate (M3 le plus haut, 1 % des 1 640 courses), celles qu'une petite
  perturbation fait basculer d'abord, sans toucher à aucune pose, donne une part « dans les réussites » de
  -0,010 [-0,017 ; -0,004], le tiers de l'effet cherché (16 réussites tirées au hasard : 0,000 en moyenne).
  H6 dit déjà ce que ces lectures visaient : la pose se dégrade alors que la réussite ne bouge pas.
- Traduction physique : écart moyen, dans la strate, de la distance minimale du T à son but (pixels) et de
  l'écart d'angle minimal (degrés), moyennés sur les graines, avec intervalle bootstrap ; minima pris sur les
  mêmes pas que M3 (amendement 4 d'EXP-003 pour les courses divergentes). Repères : la marge du T seul passe
  sous 1 à 20 px et 20 degrés ; le seuil de succès de 20 px porte sur l'agent et le T ensemble.
- Comparaison avec EXP-003 : les écarts à 6 et 8 bits côte à côte, sans les regrouper.

## Contrôles bloquants

Pour chaque course, ceux d'EXP-003 (`controler_course`). 15 fichiers attendus (fp32, all:6 et all:8,
graines 52 à 56), sans doublon, tous du même commit ; arbre git propre au lancement ; commit existant dont
`run_grid.py` et `quant.py` ont les empreintes SHA-256 inscrites dans `analyse_exp005.py` ; code d'analyse et
ce texte identiques à ceux de ce commit ; 0 ou 109 couches quantifiées ; 2 000 épisodes, graine de tirage
2026, liste d'exclusion de 550 couples égale à celle d'EXP-003 et d'EXP-002, règle de tirage inscrite,
aucune trajectoire commune avec elles, 2 000 trajectoires distinctes ; mêmes épisodes, états de départ et buts
dans tous les fichiers ; 1 333 et 233 épisodes dans les strates ; même protocole et même torch pour la
grille et la sanité de clôture ; sanité de clôture identique. Le résultat enregistre le commit de l'analyse.

## Aveugle

Le journal du lancement écrit le taux de réussite de chaque fichier : il n'est pas lu avant l'analyse.
