# Pré-enregistrement d'EXP-015 : où se loge la perte d'int4 par canal sur le Neural Engine ?

Rédigé par Claude le 2 octobre 2026 à partir de 8 h 31 (heure du Mac, lue avec `date`), avant tout run de réussite
d'EXP-015 et avant l'étape 0 b. Figé par le commit enregistré dans les fichiers. Tâche : PushT seulement.

Amendement 1 (2 octobre, à partir de 8 h 57, avant l'étape 0 b et avant tout run de réussite ; relecture du plan et du
code : 1 P0, 6 P1). Il l'emporte sur le texte qui suit.
1. Conclusion principale (P0) : L1 et L2 ne disent pas que `pred_proj` porte la perte d'int4c (une perte de 5 points de
   PP avec BL près de 0 les soutiendrait toutes deux). Test ajouté, L4 : PP ne fait pas mieux qu'int4c de plus de 2
   points (x = PP - int4c par épisode, y = -20 x + 2). « La perte d'int4c sur le Neural Engine vient de `pred_proj` » est
   soutenue seulement si L1, L2 et L4 le sont (Holm sur 4). La part portée par `pred_proj` ne se dit que par D4 et son
   intervalle.
2. Bras SP : s'il est lancé (R2), L5 est ajouté à la famille : SP ne perd pas plus de 2 points face à ne16 (y = 20 x +
   2), Holm sur 5. Le conseil de déploiement « tout coder sauf `pred_proj` » ne s'appuie que sur L5 si SP est lancé, sur
   L2 sinon (alors à poids décodés égaux, R2), et toujours contre ne16 : contre PyTorch float32, il s'écrit avec les deux
   écarts (BL ou SP contre fp32, descriptif ajouté à D1), sans enchaîner les marges en une affirmation.
3. Lecture des « non soutenue » : un test non soutenu dit « non établi », jamais l'inverse. « Coûte » ne s'écrit que si
   la perte est établie (intervalle à 95 % entièrement sous 0, comme EXP-014), calculé pour BL, PP et SP. Les branches de
   lecture plus bas se lisent ainsi.
4. Puce contre simulation : « la simulation ne prédit pas la puce » pour une portée seulement si l'intervalle D3 de cette
   portée exclut 0. Un désaccord des verdicts (L1 contre L3) seul se dit « non concluant ».
5. Règle R1, précisée : pour chaque codage, R1 est vérifiée lot par lot (e_exec au plus 2 fois celle du 16 bits sur
   chacun des 3 lots). Plancher de validité : si e_exec du 16 bits ou l'écart du 16 bits sur la puce à PyTorch float32
   dépasse 1e-2 sur un lot, l'étape 0 b est invalide (la puce ou le montage calcule autre chose) et EXP-015 ne se lance
   pas. Matérialité rapportée : e_exec / e_quant et la pire erreur par échantillon. Le verdict R1 est écrit par l'étape
   0 b et repris par l'analyse.
6. Référence de l'étape 0 b : le graphe float32 reçoit toutes les constantes arrondies en float16 comme Core ML (biais,
   normes, plongement de position), sauf la première couche de `pred_proj`, que Core ML fusionne avec sa BatchNorm (poids
   décodés divisés par le facteur, BatchNorm gardée en float32). Entrées du Neural Engine passées en mémoire contiguë,
   comme dans les runs.
7. Séries réutilisées : en plus de l'ancre de la graine 42, leurs propres reproductions (graine 42 au début, 46 à la fin,
   d'EXP-012 et d'EXP-014) doivent être identiques à EXP-003 et de leur commit, sinon la série n'est pas utilisée (arrêt).
8. Descriptifs ajoutés : temps par plan médian de chaque bras sur le Neural Engine (D5) ; BL, PP et SP contre fp32 (D1) ;
   si SP est lancé, l'interaction sur la puce se calcule aussi avec SP (plan complet ne16, SP, PP, int4c), qui ne laisse
   pas les matrices 0/1 dans le terme d'interaction. L'analyse vérifie que les modèles de l'étape 0 b sont ceux des runs
   (empreintes) et que l'étape 0 b a les versions des runs (coremltools, macOS).
9. Formulations : `const_7` est à 4,4e-4 du maximum de ligne près de float16 de son attendu, pas égale bit pour bit ; la
   taille d'int4c-sp sera celle mesurée (D5), pas le chiffre de la console des tests ; la correspondance des 30 couches
   est écrite dans la sortie de l'étape 0 (refaite au commit final) ; la perte à décomposer est int4c contre ne16 (environ
   9 points, calcul ad hoc ; contre fp32, 8,6), chiffres officiels par les analyses.
10. Lanceur : batterie revérifiée au départ (après l'attente du GPU) ; garde de batterie toujours lancée pour EXP-015
   (les gardes ne suivent que leur propre préfixe) ; macOS 26.6.2 et coremltools 9.0 exigés avant le départ ; un fichier
   de référence manquant arrête le lanceur au lieu de compter comme « différent » ; la validation de l'énergie et les
   tests des portées comptent comme calculs en cours.
11. Relecture des corrections (9 h 26, aucun P0), appliquée avant le commit final :
   - Divulgation : un essai de l'étape 0 b a tourné à 9 h 15 au commit `70f6e1e` (sortie hors du dossier des résultats),
     ses règles déjà fixées ; il a été vu : étape valide, H2 écartée par R1 pour tous les codages (erreur d'exécution du
     Neural Engine égale à celle du 16 bits à 1 % près), sorties d'int4c-sp et d'int4c-bl non identiques bit pour bit
     (donc bras SP par R2). Aucune règle n'a changé depuis ; la sortie qui compte est celle de l'étape 0 b refaite au
     commit final. Le relecteur proposait de recaler le plancher de validité sur l'exécution de Core ML sur CPU : non
     fait, l'essai ayant été vu ; cette exécution est seulement ajoutée comme repère descriptif (e_cpu).
   - Puissance de L4 (estimation ad hoc de la relecture, pas un résultat) : environ 0,9 si l'erreur-type de PP - int4c
     vaut 0,7 point, 0,45 si elle vaut 1,3 point, au niveau de 5 % (moins au niveau de Holm). Une L4 non soutenue se lit
     « non établie », jamais comme une preuve contre.
   - Une R1 « H2 retenue » ne se lirait qu'avec e_code : si e_code n'est pas petit devant e_quant, la référence est
     suspecte avant la puce.
   - L'étape 0 est refaite au commit final ; l'analyse et le lanceur exigent sa sortie à ce commit, correspondance des
     couches comprise.
   - D5 : les temps des références viennent de leurs sessions d'origine (signalé) ; les temps des ancres refaites dans
     la session sont ajoutés.

## Origine de l'hypothèse (divulgation)

L'hypothèse vient de données déjà vues, pas d'une prédiction faite à l'aveugle :
- EXP-014, résultats intermédiaires de PushT (lus le 2 octobre au matin à la demande de Thomas, par un calcul ad hoc
  hors de l'analyse pré-enregistrée ; les chiffres officiels viendront de `analyse_exp014.py`) : sur le Neural Engine,
  écart à PyTorch float32 de +0,6 [-0,4 ; +1,6] pour int8, -2,4 [-4,4 ; -0,4] pour les palettes, -8,6 [-11,2 ; -6,2]
  pour int4 par canal (int4c).
- En simulé (PyTorch float32, RTN par canal) : `predictor:4`, les 30 mêmes poids que ceux des blocs codés par Core ML
  (étape 0), +0,6 [-0,7 ; +1,9] (EXP-003) ; `autres:4` (encodeur d'action, projecteur, `pred_proj`) -9,4 [-12,1 ; -6,8]
  (EXP-004).
`pred_proj` (2 couches, 786 432 poids) est donc la candidate, par élimination seulement : elle n'a jamais été isolée,
ni en simulé ni sur la puce. Hypothèses ouvertes : H1 les valeurs de `pred_proj` à 4 bits ; H2 le chemin d'exécution
int4 du Neural Engine (un chemin du Neural Engine peut fausser la sortie sans erreur : ticket Apple n° 279, K-028) ;
H3 une interaction entre les blocs et `pred_proj` ; H4 les 2 matrices 0/1 de la réécriture ; H5 les codes tirés des
poids 16 bits plutôt que 32 bits. Existant (K-028) : garder la dernière couche en haute précision est une pratique
standard (LSQ, GPTQ ; VLAQuantBench sur une politique de robot ; jepa.cpp garde `pred.proj` en F32 sur LeWM) ; rien
trouvé qui localise la perte sur la puce pour un world model, ni qui compare la perte simulée et la perte sur la puce.
EXP-015 ne présentera pas la sensibilité de la dernière couche comme nouvelle.

## Étape 0 (faite avant ce texte ; CPU seul ; `etape0_exp015.py`, commit `5b8f249`, sortie `resultats/exp015-etape0/`)

- Correspondance : les 34 constantes codées sont 30 poids des blocs (exactement les 30 couches de `predictor:4` ; poids
  16 bits égaux à float16 des poids float32, lignes q multipliées par 1/8, exact), 2 de `pred_proj` (`const_7` porte la
  BatchNorm fusionnée par le convertisseur, écart au plus 4,4e-4 du maximum de ligne, arrondi float16) et 2 matrices
  0/1.
- int4c de coremltools contre RTN de `quant.py` : 0,09 % des codes diffèrent dans les blocs, 0,07 % dans `pred_proj`,
  d'un pas au plus ; erreur relative de reconstruction égale à 1e-5 près (blocs 0,1217 ; `pred_proj` 0,1331).
  Palettes : 0,099 et 0,096.
- Matrices 0/1 décodées exactement (0 et 1), en int4c comme en palettes.
- Portées : -bl code les 30 poids des blocs, -pp les 2 de `pred_proj`, -sp les 32 hors `pred_proj` ; toute constante non
  codée reste égale bit pour bit au programme 16 bits.
Conséquence : H5 est écartée au niveau des poids ; H4 l'est au niveau des valeurs décodées, pas encore de l'exécution.

## Étape 0 b (après la fin d'EXP-014, avant tout run de réussite ; `etape0b_exp015.py`, sortie `resultats/exp015-etape0b/`)

Sortie du prédicteur sur le Neural Engine (y_ne) contre le même graphe en float32 sur CPU avec exactement les poids
décodés par coremltools (y_dec), sur 3 lots de 300 x 3 entrées tirées comme celles de la CEM. Règles fixées ici :
- R1 (H2) : pour une portée, H2 est écartée numériquement si e_exec = ||y_ne - y_dec|| / ||y_dec|| est au plus 2 fois
  celle du 16 bits, sur chacun des 3 lots. Sinon H2 est retenue pour cette portée et rapportée ; les bras de réussite
  ne changent pas (ils mesurent l'effet sur la réussite, quelle qu'en soit la cause).
- R2 (H4) : si les sorties d'int4c-sp et d'int4c-bl sur le Neural Engine sont identiques bit pour bit sur les 3 lots,
  les matrices 0/1 sont inertes et int4c-bl représente aussi int4c-sp (pas de bras -sp) ; sinon un bras int4c-sp est
  ajouté (graines 42 à 46). Le lanceur lit cette décision dans la sortie de l'étape 0 b, produite au même commit.
- Descriptifs : e_code (coremltools contre `quant.py`, float32 des deux côtés) et e_quant (taille de l'effet simulé).

## Bras (PushT, les 500 épisodes d'EXP-003, graines 42 à 46, chunk 50, même protocole qu'EXP-014)

Sur le Neural Engine (Core ML 16 bits, coremltools 9.0, prédicteur sur le Neural Engine ; encodeur, CEM et coût en
PyTorch sur le GPU, comme EXP-012 et EXP-014) :
- BL : int4c-bl, les 30 poids des blocs codés int4 par canal, `pred_proj` et matrices 0/1 en 16 bits ;
- PP : int4c-pp, `pred_proj` seule codée ;
- SP : int4c-sp, seulement si R2 le demande.
En simulé (PyTorch float32 sur le GPU, moteur torch, comme EXP-003) :
- S-PP : `predproj:4` ;
- S-PR : `predict:4` (prédicteur et `pred_proj` : la portée de Core ML sans les matrices 0/1).
Réutilisés, ancrés par des contrôles : fp32 et `predictor:4` (EXP-003), 16 bits sur le Neural Engine (ne16, EXP-012),
int4c (EXP-014).
À chaque graine, dans cet ordre : BL, PP, (SP), S-PP et S-PR (un seul run_grid pour les deux).

Contrôles :
- Reproduction PyTorch float32, graine 42 au début et 46 à la fin, identique octet pour octet à EXP-003 : bloquante.
- Témoin ne16, graine 42, contre EXP-012 : identique, les fichiers d'EXP-012 servent de référence ne16 ; différent, le
  lanceur refait ne16 sur les graines 43 à 46 dans la session et l'analyse prend la série de la session.
- Réplique int4c, graine 42, contre EXP-014 : même règle (identique, la série d'EXP-014 sert ; sinon int4c refait sur
  les 5 graines dans la session).
- Réplique `predictor:4`, graine 42, contre EXP-003 : même règle (la série d'EXP-003 a été calculée à un autre commit ;
  la réplique vérifie que le chemin de la quantification simulée n'a pas changé).
- Après chaque run sur le Neural Engine : toutes les opérations sur le Neural Engine, sinon arrêt.
- Analyse : mêmes contrôles qu'EXP-014 (fichiers, protocole, commit unique, arbre propre, versions, codage appliqué,
  empreinte constante par bras) ; constantes codées de chaque paquet listées et rangées (BL : 30 blocs ; PP : 2 de
  `pred_proj` ; SP : 30 blocs et 2 matrices) ; `n_couches` 2 pour S-PP et 32 pour S-PR.

## Tests (Holm sur 3, alpha 5 % ; inversion de signe et bootstrap-t concordants, p retenu le plus grand, comme EXP-014)

x = somme sur les 5 graines, par épisode, de (succès du bras - succès de la référence) ; écart en points = 100 x
moyenne(x) / 5.
- L1 (perte de `pred_proj` sur la puce) : PP perd plus de 2 points face à ne16. y = -20 x - 2, moyenne positive si
  l'écart est sous -2 points.
- L2 (blocs non inférieurs sur la puce) : BL ne perd pas plus de 2 points face à ne16. y = 20 x + 2.
- L3 (la simulation localise pareil) : S-PP perd plus de 2 points face à fp32. y = -20 x - 2.
Lecture : « la perte d'int4c sur le Neural Engine vient de `pred_proj` » est soutenue si L1 et L2 le sont ; L3 dit si
la simulation PyTorch le montre aussi. « Non soutenue » veut dire « non établi », pas l'inverse.

## Descriptifs fixés (bootstrap des épisodes, graine 2, 10 000 tirages, comme l'analyse d'EXP-003)

- D1, écarts : BL - ne16, PP - ne16, (SP - ne16), int4c - ne16 ; S-PP - fp32, S-PR - fp32, `predictor:4` - fp32.
- D2, interaction : sur la puce, (int4c - ne16) - (BL - ne16) - (PP - ne16) (matrices 0/1 comprises si R2 n'a pas
  conclu à leur inertie) ; en simulé, (S-PR - fp32) - (`predictor:4` - fp32) - (S-PP - fp32).
- D3, puce contre simulation, par portée, par épisode : (PP - ne16) - (S-PP - fp32) ; (BL - ne16) - (`predictor:4` -
  fp32) ; (int4c - ne16) - (S-PR - fp32). Lecture écrite d'avance : un intervalle qui contient 0 se dit « compatible »,
  sans équivalence affirmée (la puissance ne permet pas une marge de 2 points sur une différence de différences).
- D4, part de la perte d'int4c portée par `pred_proj` sur la puce : (PP - ne16) / (int4c - ne16), intervalle bootstrap
  percentile sur les épisodes ; rapportée seulement si int4c - ne16 a un intervalle entièrement sous 0.
- D5 : réussites, taille du fichier des poids par bras, temps par plan (descriptif, sans comparaison à EXP-013).

## Ce que chaque issue voudrait dire (réécrit par l'amendement 1)

- L1, L2 et L4 soutenues : la perte d'int4c sur le Neural Engine vient de `pred_proj` (à 2 points près) ; pour déployer,
  coder tout sauf `pred_proj`, appuyé sur L5 si SP est lancé, sur L2 sinon.
- L1 et L2 soutenues, L4 non : `pred_proj` seule perd plus de 2 points et les blocs seuls moins de 2, mais la perte de
  `pred_proj` seule n'est pas établie égale à celle d'int4c ; D2 et D4 disent le reste, sans conclusion de plus.
- L1 non soutenue : une perte de plus de 2 points de `pred_proj` seule n'est pas établie (pas « `pred_proj` ne coûte
  rien ») ; « coûte » seulement si la perte de PP est établie (intervalle sous 0).
- L2 non soutenue : la non-infériorité des blocs n'est pas établie ; « les blocs coûtent » seulement si leur perte est
  établie.
- Puce contre simulation : « la simulation ne prédit pas la puce » pour une portée seulement si son intervalle D3 exclut
  0 ; L1 et L3 en désaccord sans cela : « non concluant ». À lire avec R1.

## Limites écrites d'avance

PushT seulement (la perte d'int4c sur TwoRoom sera connue par l'analyse d'EXP-014). Palettes non décomposées. GPTQ non
testé sur la puce. Une machine (M5 Max, macOS 26.6.2), une version de coremltools (9.0). Les entrées de l'étape 0 b sont
tirées, pas prises dans des épisodes.

## Analyse et budget

`analyse_exp015.py` (tests `test_analyse_exp015.py`, refus des fichiers non conformes) ; lanceur `lancer_exp015.sh`,
garde de batterie, départ à au moins 60 % de batterie, un seul lanceur, GPU libre (3 contrôles vides de suite). Durée
estimée d'après EXP-014 (10 min par run sur le Neural Engine, 9 min par run PyTorch) : environ 4 h, 4 h 50 avec SP (plus
40 min par série réutilisée dont l'ancre diffère).
Relecture : une passe sur le plan et le code, une sur les corrections, puis un essai court de bout en bout.
