# Pré-enregistrement d'EXP-021 : prédicteur ternaire avec 1 % de poids gardés et réglage fin, contre le 3 bits réglé

Rédigé par Claude le 4 novembre 2026 à partir de 11 h 07 (heure du Mac, lue avec `date`), avant tout run d'EXP-021, à la
demande de Thomas (« prépare l'expérience », 4 novembre). Figé par le commit enregistré dans les fichiers. Tâche : PushT.
Existant : K-037 et K-038 (aucun world model ternaire qui planifie ; ingrédients connus : SpQR, PEQA, EfficientQAT).
Leçons appliquées : `experiences/LECONS.md` (références réglées comme la méthode, épisodes neufs, conclusions écrites
d'avance, contrôles bloquants essayés avant, relecture scientifique).

## Pourquoi

Le pilote d'EXP-019 (100 épisodes d'EXP-003, graine 42) donne : pleine précision 82 %, 3 bits RTN non réglé 82 %,
ternaire GPTQ 52 %, ternaire avec 1 % gardé 70 %, la même réglée 79 %. Deux biais à corriger avant toute conclusion :
le 3 bits n'y était pas réglé alors que le ternaire l'était (leçon d'EXP-020), et les 100 épisodes ont servi à décider
de cette expérience. EXP-021 compare des versions toutes réglées de la même façon, sur des épisodes neufs.

## Ce qui est connu avant (divulgation)

- Pilote d'EXP-019 ci-dessus ; EXP-011 (500 épisodes d'EXP-003) : ternaire GPTQ 53,0 % contre 83,5 %.
- Sur les 2 000 épisodes d'EXP-005 : pleine précision 87,0 et 86,6 % (graines 52 et 53) ; RTN 8 et 6 bits (EXP-005) ;
  GPTQ 4 et 3 bits sur tout le modèle (EXP-010). Aucune version ternaire, à poids gardés ou réglée n'a tourné sur ces
  épisodes.
- Réglage fin : recette du pilote, fixée sans essai sur l'évaluation (5 000 pas, lots de 256, Adam 1e-3 et cosinus).

## Bras (PushT, les 2 000 épisodes d'EXP-005, graines 52 et 53, chunk 50)

Tirage : `--n 2000 --chunk 50 --graine-episodes 2026 --exclure resultats/exp003/meta.json resultats/sanite-n50/meta.json`
(celui d'EXP-005). Prédicteur seul quantifié (30 couches), échelle réduite d'EXP-011, GPTQ calibré sur
`resultats/exp009-calib-pusht/calib.pt` (trajectoires d'EXP-005 exclues). Bits par poids du prédicteur : ternaire stocké
sur 2 bits, une échelle de 16 bits par ligne, 32 bits par poids gardé (16 de valeur, 16 d'adresse).

| Clé | Version | Bits par poids du prédicteur | Rôle |
| --- | --- | --- | --- |
| A | `predictor:2:clip:gptq:k1:obs:aff` | 2,38 | la méthode : ternaire, 1 % gardé (107 640 poids), réglé |
| B | `predictor:3:clip:gptq:aff` | 3,06 | la barre : 3 bits GPTQ, réglé de la même façon |
| C | `predictor:2:clip:gptq:k1:obs` | 2,38 | A sans réglage |
| D | `predictor:2:clip:gptq:aff` | 2,06 | A sans poids gardés |
| E | `predictor:2:clip:gptq:k3.1:obs:aff` | 3,05 | ternaire avec 3,1 % gardé (333 702 poids), réglé : même mémoire que B |

Référence : pleine précision d'EXP-005 (`resultats/exp005/fp32_s52.npz` et `fp32_s53.npz`), mêmes épisodes et mêmes
graines, donc même hasard par épisode.

Réglage fin (A, B, D, E), identique pour tous : `affinage.py`, entiers et masque figés ; un facteur par ligne sur
l'échelle, les valeurs gardées et les biais sont réglés pour que la sortie du prédicteur puis de pred_proj (dernière
position) reproduise celle du float32 ; 5 000 pas, lots de 256, Adam 1e-3 avec décroissance cosinus, 5 % des lignes
tenues à l'écart. Données : entrées du prédicteur enregistrées pendant une planification en pleine précision sur les 32
premiers épisodes de calibration d'EXP-009 (tirage 9009, CEM graine 0, 8 lignes par appel), collectées par le lanceur
au commit des runs. Un réglage par bras et par graine, graine du réglage = graine du planificateur (52, 53) : la
variation du réglage fait partie de la variation mesurée.

Ordre : collecte, 8 réglages, puis graine par graine les cinq bras dans l'ordre A, B, C, D, E. 10 fichiers de 2 000
épisodes, environ 50 min chacun, plus 30 min de réglage et les reproductions : environ 9 h 30 de GPU.

## Tests (Holm sur 3, alpha 5 % ; inversion de signe et bootstrap-t, p retenue la plus grande)

x(e) = K(e, bras) - K(e, référence), K la somme des succès sur les 2 graines ; marge 2 points (celle de l'article).
Tirages : inversion de signe `analyse_exp003.Inversion` (`default_rng(0)`) ; bootstrap-t `analyse_exp012.p_bootstrap_t`
(`default_rng(4)`, 10 000 tirages).
- H1, effet du réglage : A contre C fait mieux de plus de 2 points ; y = 100 x - 4.
- H2, effet des poids gardés après réglage : A contre D fait mieux de plus de 2 points ; y = 100 x - 4.
- H3, la barre : A contre B ne perd pas plus de 2 points ; y = 100 x + 4.
Bornes unilatérales au niveau de Holm par bootstrap-t (borne inférieure de l'écart) ; un verdict n'est écrit que si
borne et verdict concordent, sinon « indéterminé ». Niveaux de Holm à ex aequo rangés dans l'ordre H1 à H3 (tri stable).
Sensibilité aux tirages comme EXP-018 (5 autres paires de graines si la p est à moins de 3 erreurs de Monte-Carlo de son
niveau), descriptive.

Puissance (approximation normale ; écart-type par épisode de la différence moyenne sur 2 graines entre 0,31 et 0,45,
d'après les taux de discordance du pilote, 15 à 21 %) : erreur-type de l'écart 0,7 à 1,0 point. H1 : presque 1 si
l'écart vrai vaut celui du pilote (+9), 0,6 à 0,85 s'il vaut +4. H2 : 0,8 à 0,99 si l'écart vrai vaut +5. H3 : 0,45 à
0,75 si l'écart vrai est nul (plus si H1 et H2 passent avant dans Holm), 0,2 à 0,4 s'il vaut -1, presque 0 s'il vaut -3.

## Descriptifs fixés (bootstrap des épisodes, `default_rng(2)`, 10 000 tirages)

Réussite de chaque bras ; écarts A, B, C, D, E contre la pleine précision ; A contre B, C, D ; E contre B (même
mémoire) ; E contre A ; perte de contrôle du réglage avant et après (lignes tenues à l'écart) pour chaque réglage.
Aucun écart descriptif n'est lu comme un test.

## Lecture écrite d'avance

- H1 soutenue : « régler les échelles, les valeurs gardées et les biais sur des entrées enregistrées en planification
  améliore le prédicteur ternaire avec 1 % gardé de plus de 2 points ». Non soutenue : « gain de plus de 2 points du
  réglage non établi », avec l'écart et son intervalle.
- H2 soutenue : « après réglage, garder 1 % des poids en 16 bits apporte plus de 2 points au prédicteur ternaire ».
  Non soutenue : « apport de plus de 2 points des poids gardés non établi ».
- H3 soutenue : « à 2,38 bits par poids, le prédicteur ternaire avec 1 % gardé et réglé ne perd pas plus de 2 points
  face au 3 bits réglé de la même façon (3,06 bits), sur des épisodes neufs ». Non soutenue : « non-infériorité face
  au 3 bits réglé non établie », avec l'écart et son intervalle ; jamais « perte » sans intervalle entièrement sous 0.
- Face à la pleine précision : écart et intervalle seulement ; jamais « sans perte » ni « gratuit » (aucun test).
- E contre B, à mémoire égale : écart et intervalle seulement ; on peut écrire lequel a le meilleur écart moyen, jamais
  « meilleur » sans un test pré-enregistré.

## Contrôles bloquants

- Code : un seul commit pour tous les fichiers d'EXP-021, arbre propre ; fichiers de calcul et d'analyse identiques entre
  ce commit et celui de l'analyse.
- Reproduction de la pleine précision d'EXP-005 sur les 100 premières positions, graine 52 au début et graine 53 à la
  fin : tableaux identiques à ceux d'EXP-005 sur ces positions (le pilote a montré l'identité après macOS 27).
- Pour chaque fichier : mêmes épisodes, départs, états initiaux et buts que la référence ; protocole identique à celui
  d'EXP-005, calibration GPTQ et dossier de réglage exceptés ; 30 couches quantifiées ; poids gardés attendus (107 640
  pour A et C, 333 702 pour E, 0 pour B et D) ; contrôles de course.
- Réglage : chaque bras réglé charge le fichier de sa version et de sa graine (empreinte dans le fichier de résultats
  égale à celle du fichier de réglage) ; spec, graine et empreinte des données conformes ; format vérifié au chargement
  (mêmes entiers, une échelle float16 par ligne, valeurs gardées float16, même masque).
- Données de réglage disjointes des épisodes d'évaluation : aucune trajectoire commune avec les 2 000 épisodes.
- Ces contrôles sont exercés avant le lancement sur un essai court de bout en bout (`ESSAI=1`), sur la vraie machine.

## Limites écrites d'avance

Une tâche, un checkpoint, une machine ; quantification simulée (poids ramenés en float32 pour le calcul), donc ni
vitesse ni énergie mesurées ; réglage fixé sans recherche ; 2 graines (réglage et planificateur confondus) ; une seule
quantité de données de réglage (32 épisodes) ; le ternaire rangé sur 1,6 bit (5 valeurs par octet) donnerait 1,98 bit
pour A, non mesuré ; pas de TwoRoom (expérience séparée si H3 passe).

## Analyse et budget

`analyse_exp021.py`, testé par `test_analyse_exp021.py` sur des fichiers fabriqués à effet connu (dont des p
intermédiaires) avant tout run. Lanceur `lancer_exp021.sh` (garde de batterie, sentinelle, lancement détaché). Environ
9 h 30 de GPU, aucun crédit Claude pendant le calcul.
