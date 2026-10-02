# Pré-enregistrement d'EXP-014 : les codages des poids du prédicteur dans Core ML, sur le Neural Engine, gardent-ils la réussite ?

Rédigé par Claude le 2 octobre 2026 à partir de 2 h 54 (heure du Mac, lue avec `date`), avant tout calcul d'EXP-014.
Figé par le commit enregistré dans les fichiers. Ce qui est connu avant : EXP-012 (Core ML 16 bits sur le Neural Engine,
non-infériorité à 2 points soutenue sur PushT et TwoRoom) et EXP-013 (int8 par canal, int4 par canal et palettes de 16
valeurs ne changent ni le temps ni l'énergie par plan sur le Neural Engine, et divisent le fichier des poids du
prédicteur par 2 à 4 ; leur réussite n'y est mesurée que sur 50 épisodes, sans test). Aucun fichier d'EXP-014 n'existe.

Existant (K-027) : rien trouvé sur la réussite d'un planificateur avec des poids codés par Core ML sur un NPU ; la
quantification elle-même (RTN par canal, palettes par k-moyennes) est faite par les fonctions de coremltools, pas par
nous, et n'est pas nouvelle. Fu et al. (2602.02110) et Where Bits Matter (2602.11882) quantifient DINO-WM en simulé.

Amendement 1 (2 octobre, 03 h 11, avant tout calcul d'EXP-014 ; code au commit `d70f67b`, tests passés : 32 refus). Une
relecture du plan et du code (1 P0, 6 P1) le corrige ainsi ; il l'emporte sur le texte qui suit.
1. Ce qui est codé (P0) : coremltools code toute constante de plus de 2 048 éléments du graphe converti, qui est
   `model.predict` réécrit (prédicteur et projection de sortie `pred_proj`). Listé par script sur les modèles d'EXP-013
   (`analyse_exp014.constantes_codees`) : 34 constantes, 30 dans les 6 blocs du prédicteur (modulation, qkv, sortie,
   fc1, fc2), les 2 couches de `pred_proj` (l'une nommée `const_7` par le convertisseur, de forme 2 048 x 192), et 2
   matrices 0/1 de la réécriture « 1ctb » (somme et étalement des têtes). « Le codage porte sur le seul prédicteur » est
   donc faux : il porte sur le prédicteur et `pred_proj`. Contexte : à 4 bits en simulé, le prédicteur seul ne perdait
   rien de mesurable (+0,6 [-0,7 ; +1,9], EXP-003, `predictor:4`), et les 6 couches hors encodeur et prédicteur, dont
   `pred_proj`, coûtaient 9,4 points (EXP-004, `autres:4`). Un échec d'int4c ne se lit donc pas comme « le prédicteur
   est sensible dans Core ML ». L'analyse liste les constantes codées de chaque modèle d'EXP-014.
2. int4c, formulation corrigée : l'int4 linéaire symétrique par canal de coremltools est la même grille que notre RTN
   par canal (arrondi de w / (max|w| / 7), borné à [-7, 7], par canal de sortie) ; les différences sont les poids et les
   échelles en 16 bits, le calcul en 16 bits sur le Neural Engine et la portée du point 1. pal4 palettise les tenseurs du
   graphe réécrit (dont qkv, où les lignes q portent le facteur 1/8 replié) : son résultat vaut pour ce graphe.
3. Témoin 16 bits : pour chaque tâche, un run 16 bits sur le Neural Engine, graine 42, refait dans EXP-014
   (`exp014-ne16-*`), comparé à celui d'EXP-012. S'il est identique octet pour octet, la colonne « effet du codage seul »
   garde ce nom ; sinon elle se lit « différence avec le 16 bits d'une autre session ». Non bloquant.
4. Contrôles bloquants ajoutés : fichiers 16 bits d'EXP-012 du même moteur, du commit d'EXP-012, d'un seul modèle, au
   même protocole ; versions coremltools 9.0 et macOS d'EXP-012 pour tous les runs d'EXP-014 ; le codage a vraiment été
   appliqué (empreinte différente du 16 bits, fichier des poids plus petit, trois codages différents) ; sur PushT, le
   modèle est celui dont EXP-013 a mesuré le temps (même empreinte) ; le témoin 16 bits a le modèle d'EXP-012. Le
   lanceur s'arrête dès qu'un run codé a une opération hors du Neural Engine.
5. Lecture : « non soutenue » veut dire que la non-infériorité n'est pas établie, pas une perte ; une perte n'est
   affirmée que si l'intervalle à 95 % est entièrement sous 0. Le contrôle de la marge (fp32 contre fp32) est rapporté au
   niveau de Holm de chaque test (dans EXP-012, il passait à 5 % mais pas toujours aux niveaux de Holm sur TwoRoom).
6. Formulations : EXP-013 n'a mesuré le temps que sur PushT, sur une machine et une session, et sa règle disait « non
   constant » (aucune différence constante au-delà de 5 % pour le temps et 10 % pour l'énergie), pas « aucun
   changement ». « Toutes les opérations de calcul » sur le Neural Engine : les opérations de décompression n'ont pas de
   processeur dans le plan de calcul. Le temps par plan d'EXP-014 (sans mise en route, paquets dans le dossier des
   résultats) ne se compare pas à celui d'EXP-013. La taille du 16 bits est calculée depuis le fichier d'EXP-012.
7. Lanceur : départ à au moins 60 % de batterie ; un seul lanceur à la fois. Durée : environ 6 h (2 témoins en plus).

## Question et hypothèses

Les trois codages gratuits en temps sur le Neural Engine (EXP-013) font-ils perdre plus de 2 points de réussite au
planificateur, face à la pleine précision PyTorch ?

- N1 à N3 (PushT) et N4 à N6 (TwoRoom), pour int8, int4c et pal4 dans cet ordre : la réussite du planificateur avec le
  prédicteur codé, exécuté par Core ML sur le Neural Engine, ne perd pas plus de 2 points face à PyTorch float32.
- Pas de prédiction de sens pour int4c et pal4 ; int8 est attendu non inférieur (8 bits par canal ne perdait presque
  rien en simulé, EXP-005). Ces attentes ne sont pas testées.

## Plan

- LeWM publié ; PushT (les 500 épisodes d'EXP-003) et TwoRoom (les 500 d'EXP-006) ; graines de la CEM 42 à 46 ; CEM
  300 x 30, horizon 5 ; mêmes options que EXP-012.
- Moteur `coreml-ne` d'EXP-012 et d'EXP-013 : prédicteur réécrit « 1ctb », converti en Core ML 16 bits (coremltools 9.0,
  mlprogram, macOS 15), puis codé par coremltools : int8 linéaire symétrique par canal ; int4c linéaire symétrique int4
  par canal ; pal4 palettes de 16 valeurs par tenseur (k-moyennes). Encodeur, CEM et coût en PyTorch sur le GPU, comme
  EXP-012. Le codage porte sur le seul prédicteur.
- Par tâche : reproduction bloquante de la pleine précision PyTorch à la graine 42 ; puis, pour chaque graine de 42 à
  46, les trois codages l'un après l'autre (une dérive les touche pareil) ; enfin reproduction finale de la graine 46.
- 30 fichiers de 500 épisodes et 4 reproductions, environ 5 h 30. Garde de batterie de l'expérience (copie d'EXP-013).

## Tests (méthode d'EXP-012)

- Pour chaque hypothèse : différence appariée par épisode, somme sur les 5 graines, y = 20 x + 2 (moyenne positive si
  la perte est de moins de 2 points). Test unilatéral par inversion de signe (10 000 tirages, `default_rng(0)`) et par
  bootstrap-t (10 000 rééchantillonnages, `default_rng(4)`) ; p retenue = la plus grande des deux. Holm sur les 6 tests à
  5 %. Borne inférieure unilatérale du bootstrap-t au niveau de Holm de chaque test, rapportée.
- Contrôle de la marge : deux séries de la pleine précision (graines 47 à 51 contre 42 à 46, écart vrai nul) au même
  test. Sensibilité : référence sur les 10 graines de la pleine précision.
- Descriptifs : réussite de chaque codage ; effet du codage seul contre Core ML 16 bits sur le Neural Engine d'EXP-012
  (mêmes épisodes et graines, commit d'EXP-012), intervalle bootstrap ; temps par plan (médiane des fichiers) ; taille du
  fichier des poids. Pas d'énergie (EXP-013 l'a mesurée).

## Contrôles bloquants

Reproductions PyTorch des graines 42 et 46 identiques octet pour octet à EXP-003 et EXP-006 ; un seul commit, arbre
propre ; mêmes épisodes que la référence ; protocole identique à la référence sauf le moteur ; moteur `coreml-ne`, 16 bits,
codage enregistré égal au dossier ; toutes les opérations du prédicteur sur le Neural Engine (plan de calcul) ; modèle
Core ML identique d'une graine à l'autre (empreintes) ; contrôles de course d'EXP-003 et d'EXP-006 ; les fichiers 16 bits
d'EXP-012 présents et sur les mêmes épisodes.

## Formulations

« Codage des poids par coremltools 9.0 », pas « notre quantification » ; int4c est l'arrondi de coremltools, avec ses
propres échelles, différent de notre RTN simulé (EXP-003) et du calcul en 16 bits de PyTorch : les chiffres ne se
comparent pas directement. Sur un MacBook Pro M5 Max, macOS 26.6.2.
