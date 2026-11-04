# Pré-enregistrement d'EXP-021 : prédicteur ternaire avec 1 % de poids gardés et réglage fin, contre le 3 bits réglé

Rédigé par Claude le 4 novembre 2026 à partir de 11 h 07 (heure du Mac, lue avec `date`), avant tout run d'EXP-021, à la
demande de Thomas (« prépare l'expérience », 4 novembre). Figé par le commit enregistré dans les fichiers. Tâche : PushT.
Existant : K-037 et K-038 (aucun world model ternaire qui planifie ; ingrédients connus : SpQR, PEQA, EfficientQAT).
Leçons appliquées : `experiences/LECONS.md` (références réglées comme la méthode, épisodes neufs, conclusions écrites
d'avance, contrôles bloquants essayés avant, relecture scientifique).

Amendement 1 (4 novembre, 11 h 30, avant tout run de l'expérience ; seul un essai court de 50 épisodes a tourné, au commit
7ec626a, sur des dossiers `essai-exp021-*`). Deux relectures courtes : méthode et statistique (0 P0, 6 P1) ; code et
protocole (1 P0, 2 P1). Il l'emporte sur le texte qui suit, mis à jour en conséquence.
1. P0, garde de batterie : elle ne suivait que les `run_grid.py --out` de son dossier et serait sortie pendant les
   réglages, laissant la grille sans protection. Le lanceur relance une garde avant chaque étape de `run_grid.py`.
2. Variance entre réglages (P1) : avec deux réglages, les conclusions ne valaient que pour ces deux modèles. Quatre
   réglages (graines 52 à 55) sur les 1 000 premières positions des épisodes d'EXP-005, même nombre de couples épisode et
   graine ; chaque lecture dit « pour ces quatre réglages » et indique si l'écart estimé passe le seuil dans chacun.
3. Barre 3 bits (P1) : ajout de B0, le même 3 bits non réglé (`predictor:3:clip:gptq`). Si B0 dépasse B de plus de 2
   points en moyenne, la lecture de H3 le dit, comme un constat descriptif non testé (amendement 2).
4. Mémoire (P1) : format écrit et compté. Bits par poids des couches linéaires du prédicteur (biais en float32, normes et
   plongements exclus, identiques dans tous les bras) ; poids gardé = 16 bits de valeur et 16 bits d'indice de colonne,
   plus un compte de 16 bits par ligne. A et C 2,44 bits, B et B0 3,06, D 2,06, E 3,11 : E a une mémoire voisine de B,
   pas égale. Nombre de poids et de lignes quantifiés enregistré dans chaque fichier et contrôlé.
5. Contrôles (P1) : données de réglage (32 épisodes, graine 0, 8 lignes par appel, PushT, 8 640, 8 640 et 25 920 lignes
   par longueur d'historique, comme le pilote) ; perte de contrôle finie et en baisse pour chaque réglage, exigée par
   `affinage.py` et par l'analyse ; fichier de réglage écrit d'un coup (fichier temporaire puis renommage).
6. Lectures (P1) : E contre B n'est ni testé ni classé ; l'intervalle d'un test est sa borne au niveau de Holm,
   l'intervalle percentile est étiqueté descriptif.

Amendement 2 (4 novembre, 11 h 37, avant tout run de l'expérience). Relecture des corrections : 0 P0, 2 P1.
1. Batterie : la garde ne gèle que `run_grid.py` ; la collecte et les 16 réglages (environ 1 h de GPU) attendent
   désormais une batterie à 60 % au moins avant de démarrer.
2. La phrase sur B0 contre B est descriptive, avec son intervalle, et ne dit plus que le réglage « a dégradé » la barre.
Le lanceur écrit aussi ce qui occupe le GPU quand il attend.

## Pourquoi

Le pilote d'EXP-019 (100 épisodes d'EXP-003, graine 42) donne : pleine précision 82 %, 3 bits RTN non réglé 82 %,
ternaire GPTQ 52 %, ternaire avec 1 % gardé 70 %, la même réglée 79 %. Deux biais à corriger avant toute conclusion :
le 3 bits n'y était pas réglé alors que le ternaire l'était (leçon d'EXP-020), et les 100 épisodes ont servi à décider
de cette expérience. EXP-021 compare des versions toutes réglées de la même façon, sur des épisodes neufs.

## Ce qui est connu avant (divulgation)

- Pilote d'EXP-019 ci-dessus ; EXP-011 (500 épisodes d'EXP-003) : ternaire GPTQ 53,0 % contre 83,5 %.
- Sur les épisodes d'EXP-005 : pleine précision 87,7, 86,5, 87,0 et 86,5 % (1 000 premières positions, graines 52 à 55) ; RTN 8 et 6 bits (EXP-005) ;
  GPTQ 4 et 3 bits sur tout le modèle (EXP-010). Aucune version ternaire, à poids gardés ou réglée n'a tourné sur ces
  épisodes.
- Réglage fin : recette du pilote, fixée sans essai sur l'évaluation (5 000 pas, lots de 256, Adam 1e-3 et cosinus).

## Bras (PushT, 1 000 premières positions des épisodes d'EXP-005, graines 52 à 55, chunk 50)

Tirage : `--n 2000 --chunk 50 --graine-episodes 2026 --exclure resultats/exp003/meta.json resultats/sanite-n50/meta.json`
(celui d'EXP-005), `--premiers 1000`. Prédicteur seul quantifié (30 couches), échelle réduite d'EXP-011, GPTQ calibré sur
`resultats/exp009-calib-pusht/calib.pt` (trajectoires d'EXP-005 exclues). Bits par poids des couches linéaires du
prédicteur (biais, normes et plongements exclus) : entiers sur 2 ou 3 bits, une échelle de 16 bits par ligne ; par
poids gardé, 16 bits de valeur et 16 bits d'indice de colonne, plus un compte de 16 bits par ligne.

| Clé | Version | Bits par poids des couches linéaires du prédicteur | Rôle |
| --- | --- | --- | --- |
| A | `predictor:2:clip:gptq:k1:obs:aff` | 2,44 | la méthode : ternaire, 1 % gardé (107 640 poids), réglé |
| B | `predictor:3:clip:gptq:aff` | 3,06 | la barre : 3 bits GPTQ, réglé de la même façon |
| B0 | `predictor:3:clip:gptq` | 3,06 | contrôle de la barre : le même 3 bits, non réglé |
| C | `predictor:2:clip:gptq:k1:obs` | 2,44 | A sans réglage |
| D | `predictor:2:clip:gptq:aff` | 2,06 | A sans poids gardés |
| E | `predictor:2:clip:gptq:k3.1:obs:aff` | 3,11 | ternaire avec 3,1 % gardé (333 702 poids), réglé : mémoire voisine de B |

Référence descriptive : pleine précision d'EXP-005 (`resultats/exp005/fp32_s52.npz` à `fp32_s55.npz`, 1 000 premières
positions), mêmes épisodes et mêmes graines, donc même hasard par épisode.

Réglage fin (A, B, D, E), identique pour tous : `affinage.py`, entiers et masque figés ; un facteur par ligne sur
l'échelle, les valeurs gardées et les biais sont réglés pour que la sortie du prédicteur puis de pred_proj (dernière
position) reproduise celle du float32 ; 5 000 pas, lots de 256, Adam 1e-3 avec décroissance cosinus, 5 % des lignes
tenues à l'écart. Données : entrées du prédicteur enregistrées pendant une planification en pleine précision sur les 32
premiers épisodes de calibration d'EXP-009 (tirage 9009, CEM graine 0, 8 lignes par appel), collectées par le lanceur
au commit des runs. Un réglage par bras et par graine, graine du réglage = graine du planificateur (52 à 55). Les tests
rééchantillonnent les épisodes, pas les réglages : leurs conclusions valent pour ces quatre réglages ; l'écart par
réglage est rapporté à côté.

Ordre : collecte, 16 réglages, puis graine par graine les six bras dans l'ordre A, B, B0, C, D, E. 24 fichiers de 1 000
épisodes, environ 26 min chacun, plus environ 1 h de réglage et les reproductions : environ 11 h 30 de GPU.

## Tests (Holm sur 3, alpha 5 % ; inversion de signe et bootstrap-t, p retenue la plus grande)

x(e) = K(e, bras) - K(e, référence), K la somme des succès sur les 4 graines ; marge 2 points (celle de l'article).
Tirages : inversion de signe `analyse_exp003.Inversion` (`default_rng(0)`) ; bootstrap-t `analyse_exp012.p_bootstrap_t`
(`default_rng(4)`, 10 000 tirages).
- H1, effet du réglage : A contre C fait mieux de plus de 2 points ; y = 100 x - 8.
- H2, effet des poids gardés après réglage : A contre D fait mieux de plus de 2 points ; y = 100 x - 8.
- H3, la barre : A contre B ne perd pas plus de 2 points ; y = 100 x + 8.
Bornes unilatérales au niveau de Holm par bootstrap-t (borne inférieure de l'écart) ; un verdict n'est écrit que si
borne et verdict concordent, sinon « indéterminé ». Niveaux de Holm à ex aequo rangés dans l'ordre H1 à H3 (tri stable).
Sensibilité aux tirages comme EXP-018 (5 autres paires de graines si la p est à moins de 3 erreurs de Monte-Carlo de son
niveau), descriptive.

Puissance (approximation normale ; écart-type par épisode de la différence moyenne sur 4 graines entre 0,22 et 0,45,
d'après les taux de discordance du pilote, 15 à 25 %) : erreur-type de l'écart 0,7 à 1,1 point, variation entre
réglages non comprise. H1 : presque 1 si
l'écart vrai vaut celui du pilote (+9), 0,6 à 0,85 s'il vaut +4. H2 : 0,8 à 0,99 si l'écart vrai vaut +5. H3 : 0,45 à
0,75 si l'écart vrai est nul (plus si H1 et H2 passent avant dans Holm), 0,2 à 0,4 s'il vaut -1, presque 0 s'il vaut -3.

## Descriptifs fixés (bootstrap des épisodes, `default_rng(2)`, 10 000 tirages)

Réussite de chaque bras, et par graine ; écarts A, B, B0, C, D, E contre la pleine précision ; A contre B, C, D ; B0
contre B ; E contre B (mémoire voisine) ; E contre A ; écart par réglage pour H1 à H3 ; perte de contrôle du réglage avant et après (lignes tenues à l'écart) pour chaque réglage.
Aucun écart descriptif n'est lu comme un test.

## Lecture écrite d'avance

- H1 soutenue : « pour ces quatre réglages, régler les échelles, les valeurs gardées et les biais sur des entrées enregistrées en planification
  améliore le prédicteur ternaire avec 1 % gardé de plus de 2 points ». Non soutenue : « gain de plus de 2 points du
  réglage non établi », avec l'écart et son intervalle.
- H2 soutenue : « pour ces quatre réglages, après réglage, garder 1 % des poids en 16 bits apporte plus de 2 points au prédicteur ternaire ».
  Non soutenue : « apport de plus de 2 points des poids gardés non établi ».
- H3 soutenue : « pour ces quatre réglages, à 2,44 bits par poids des couches linéaires du prédicteur, la version
  ternaire avec 1 % gardé et réglée ne perd pas plus de 2 points face au 3 bits réglé de la même façon (3,06 bits), sur
  des épisodes neufs » ; suivi de « l'écart estimé dépasse -2 points dans chacun des 4 réglages » ou de « l'écart estimé
  varie selon le réglage (de x à y points) » ; et, si B0 dépasse B de plus de 2 points en moyenne, « B0 non réglé dépasse
  B de x [IC] points (descriptif, non testé) : la barre B réglé est peut-être dégradée par le réglage ». Non soutenue : « non-infériorité face
  au 3 bits réglé non établie », avec l'écart et son intervalle ; jamais « perte » sans intervalle entièrement sous 0.
- Face à la pleine précision : écart et intervalle seulement ; jamais « sans perte » ni « gratuit » (aucun test).
- E contre B, mémoire voisine : écart et intervalle, non testé ; aucun classement.

## Contrôles bloquants

- Code : un seul commit pour tous les fichiers d'EXP-021, arbre propre ; fichiers de calcul et d'analyse identiques entre
  ce commit et celui de l'analyse.
- Reproduction de la pleine précision d'EXP-005 sur les 100 premières positions, graine 52 au début et graine 55 à la
  fin : tableaux identiques à ceux d'EXP-005 sur ces positions (le pilote a montré l'identité après macOS 27).
- Pour chaque fichier : mêmes épisodes, départs, états initiaux et buts que la référence ; protocole identique à celui
  d'EXP-005, calibration GPTQ, dossier de réglage et `premiers` (1 000) exceptés ; 30 couches quantifiées, 10 764 288 poids et
  39 936 lignes ; poids gardés attendus (107 640 pour A et C, 333 702 pour E, 0 pour B, B0 et D) ; contrôles de course.
- Réglage : chaque bras réglé charge le fichier de sa version et de sa graine (empreinte dans le fichier de résultats
  égale à celle du fichier de réglage) ; spec, graine et empreinte des données conformes ; format vérifié au chargement
  (mêmes entiers, une échelle float16 par ligne, valeurs gardées float16, même masque).
- Données de réglage : disjointes des épisodes d'évaluation (aucune trajectoire commune) ; 32 épisodes, graine 0, 8
  lignes par appel, PushT, lignes par longueur d'historique égales à celles du pilote ; perte de contrôle finie et en
  baisse pour chaque réglage.
- Ces contrôles sont exercés avant le lancement sur un essai court de bout en bout (`ESSAI=1`), sur la vraie machine.

## Limites écrites d'avance

Une tâche, un checkpoint, une machine ; quantification simulée (poids ramenés en float32 pour le calcul), donc ni
vitesse ni énergie mesurées ; réglage fixé sans recherche ; 4 réglages (réglage et planificateur confondus ; les tests
ne rééchantillonnent pas les réglages) ; une seule
quantité de données de réglage (32 épisodes) ; le ternaire rangé sur 1,6 bit (5 valeurs par octet) donnerait 2,04 bits
pour A, non mesuré ; pas de TwoRoom (expérience séparée si H3 passe).

## Analyse et budget

`analyse_exp021.py`, testé par `test_analyse_exp021.py` sur des fichiers fabriqués à effet connu (dont des p
intermédiaires) avant tout run. Lanceur `lancer_exp021.sh` (garde de batterie, sentinelle, lancement détaché). Environ
11 h 30 de GPU, aucun crédit Claude pendant le calcul.
