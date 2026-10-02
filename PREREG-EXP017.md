# Pré-enregistrement d'EXP-017 : les codes de GPTQ sur le Neural Engine rattrapent-ils la perte d'int4 ?

Rédigé par Claude le 2 octobre 2026 à partir de 16 h 07 (heure du Mac, lue avec `date`), avant tout run d'EXP-017.
Figé par le commit enregistré dans les fichiers. Tâche : PushT. Existant : K-030 (Apple propose GPTQ pour Core ML côté
PyTorch ; rien de mesuré pour un planificateur sur le Neural Engine).

Amendement 1 (2 octobre, à partir de 16 h 30, calcul en cours, avant toute lecture d'un résultat d'EXP-017 ; relecture
du plan et du calcul : 1 P0 de procédure, 1 P1, 5 P2). Il l'emporte sur le texte qui suit. Écrit sur une branche à part
(`exp017-amend`, issue de `8d24631`) pour ne pas arrêter le lanceur ; les fichiers de calcul restent ceux de `8d24631`.
1. Contrôle du code (P0) : l'analyse exige que les fichiers de calcul (run_grid.py, quant.py, gptq.py, coreml_pred.py,
   predicteur_ane.py, metriques.py, etape0_exp017.py, etape0_exp015.py, etape0b_exp015.py) soient ceux du commit des
   runs, et que le commit de l'analyse en descende ; les fichiers d'analyse et ce texte sont ceux du commit de l'analyse.
2. Phrase visée, réécrite (P1) : « Sur PushT, prédicteur sur le Neural Engine, les codes de GPTQ à 4 bits par canal
   écrits dans Core ML gardent la réussite à 2 points près de celle du 16 bits sur la puce (G1) et font mieux que
   l'arrondi au plus proche d'int4c de plus de 2 points (G2), pour un fichier des poids de la taille d'int4c ». Contre
   PyTorch float32, seul l'écart G - fp32 et son intervalle se citent, sans enchaîner les marges. G2 seule soutenue :
   « fait mieux que l'arrondi au plus proche ; non-infériorité au 16 bits non établie », rien sur un rattrapage. La
   phrase n'est écrite que si la borne et le verdict concordent pour G1 et G2.
3. Descriptif ajouté : part de la perte d'int4c rattrapée, (G - int4c) / (ne16 - int4c), intervalle bootstrap
   percentile sur les épisodes, rapportée si la perte d'int4c contre ne16 est établie.
4. Étape 0 bloquante (P2) : 32 couches GPTQ ; sur chacun des 3 lots, écart de la sortie de la puce à la sortie GPTQ
   simulée au plus 2 fois le repère 16 bits contre float32 (règle R1 d'EXP-015) ; rapports rapportés. Fichier des poids de
   G égal, à l'octet près, à celui d'int4c.
5. Lecture des descriptifs (P2) : puce contre simulation « compatible » si l'intervalle contient 0, jamais une
   équivalence ; « la simulation ne prédit pas la puce » seulement s'il l'exclut. G - int4c-sp : descriptif, méthodes et
   tailles différentes, « non comparable » si une ancre de la session diffère de sa série.
6. Temps par plan : ceux des ancres ne16 et int4c refaites dans la session (graine 42) sont rapportés à côté de celui de
   G ; aucune comparaison de vitesse n'est affirmée.

## Ce qui est connu avant (divulgation)

- EXP-014 et EXP-015 : int4 par canal (RTN) sur le Neural Engine perd 9,0 points face au 16 bits sur la puce, presque
  toute la perte vient de `pred_proj` ; tout coder sauf `pred_proj` ne perd rien de mesurable (7,1 Mo).
- EXP-009 (simulé, tout le modèle encodeur compris, 4 bits) : GPTQ perd -1,5 [-3,3 ; +0,4] point face à la pleine
  précision, contre -14,1 pour RTN. GPTQ à la portée de Core ML (prédicteur et `pred_proj` seuls) n'a jamais tourné.
- Étape 0 (essai du 2 octobre, 15 h 50 à 16 h 05, sur le commit en préparation) : le quantificateur de coremltools,
  appliqué aux poids GPTQ, change 0,43 % des codes ; l'écriture directe des codes (`coreml_pred.comprimer_gptq`) les
  garde tous (0 code différent sur 11 550 720), valeurs décodées à 9,1e-4 du maximum de ligne près (échelle en float16),
  tout sur le Neural Engine, fichier des poids de 5,9 Mo (comme int4c). L'étape 0 est refaite au commit final par
  `etape0_exp017.py`.

## Question

Écrire dans Core ML les codes int4 par canal de GPTQ (même grille, même taille de fichier qu'int4c) garde-t-il la
réussite du planificateur sur le Neural Engine, et rattrape-t-il la perte d'int4c ?

## Bras (PushT, 500 épisodes d'EXP-003, graines 42 à 46, chunk 50, protocole d'EXP-015)

- G : `predict:4:gptq` (GPTQ sur les 32 couches du prédicteur et de `pred_proj`, calibration d'EXP-009, même fichier) ;
  codes écrits dans Core ML (`--codage int4c-gptq`), prédicteur sur le Neural Engine.
- GS : `predict:4:gptq` en simulé (PyTorch float32), pour relier la puce à la simulation.
Références réutilisées : fp32 (EXP-003), ne16 (EXP-012), int4c (EXP-014), int4c-sp (EXP-015), `predict:4` RTN simulé
(EXP-015). Contrôles : reproduction PyTorch float32, graine 42 au début et 46 à la fin, identique à EXP-003 (bloquante) ;
ancres ne16 et int4c refaites sur la graine 42 (identiques : séries réutilisées ; sinon séries refaites sur les 5 graines
dans la session) ; tout sur le Neural Engine après chaque run (sinon arrêt). À chaque graine : G, puis GS.

## Tests (Holm sur 2, alpha 5 % ; inversion de signe et bootstrap-t concordants, p retenu le plus grand)

x = somme sur les 5 graines, par épisode, de (succès de G - succès de la référence).
- G1 : G ne perd pas plus de 2 points face à ne16 (y = 20 x + 2).
- G2 : G fait mieux qu'int4c de plus de 2 points (y = 20 x - 2).
Phrase visée, si G1 et G2 sont soutenues : « sur le Neural Engine, les codes de GPTQ à 4 bits par canal gardent la
réussite à 2 points près et rattrapent la perte de l'arrondi au plus proche, au même fichier de 5,9 Mo ».
« Non soutenue » : non établi, pas l'inverse ; « perd » seulement si l'intervalle à 95 % est entièrement sous 0.

## Descriptifs fixés (bootstrap des épisodes, graine 2, 10 000 tirages)

G - fp32 ; GS - fp32 ; GS - `predict:4` RTN (GPTQ contre RTN en simulé, à cette portée) ; puce contre simulation
(G - ne16) - (GS - fp32) ; G - int4c-sp (tout en int4 avec GPTQ, 5,9 Mo, contre tout sauf `pred_proj` en RTN, 7,1 Mo) ;
réussites, tailles des fichiers des poids, temps par plan (descriptif).

## Limites

PushT seulement ; une calibration GPTQ (celle d'EXP-009, non séquentielle) ; une machine, coremltools 9.0, macOS 26.6.2.

## Analyse et budget

`analyse_exp017.py` (tests `test_analyse_exp017.py`) ; lanceur `lancer_exp017.sh` (garde de batterie, batterie et GPU
libres au départ, un seul lanceur, versions exigées). Durée estimée : environ 2 h 15. Relecture : une passe sur le plan
et le calcul statistique, puis un essai court.
