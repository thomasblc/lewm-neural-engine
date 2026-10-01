# Pré-enregistrement d'EXP-013 : vitesse, énergie et mémoire de la planification de LeWM selon le moteur, le codage des poids et une charge de vision

Brouillon rédigé par Claude le 1er octobre 2026 à partir de 18 h 02 (heure du Mac, lue avec `date`), avant tout calcul
d'EXP-013. Figé par le commit enregistré dans les fichiers. Les mesures d'EXP-012 (temps par plan, 50 épisodes de pilote,
journal d'énergie par composant pendant TwoRoom) sont connues et ont motivé ce plan ; aucun fichier d'EXP-013 n'existe.

## Questions

1. Que fait gagner chaque choix de déploiement, en temps par plan, en énergie par plan et en mémoire des poids : le
   moteur (PyTorch sur le GPU en 32 et 16 bits, Core ML sur le GPU, Core ML sur le Neural Engine) et le codage des poids
   dans Core ML (16 bits, 8 bits par canal, 4 bits par blocs de 32, palettes de 16 valeurs) ?
2. Avec une charge de vision en continu sur le GPU (un robot voit et agit en même temps), le planificateur sur le
   Neural Engine garde-t-il son temps par plan, et la vision ses images par seconde, mieux que le planificateur qui
   partage le GPU ?
3. Conditionnelle à l'essai de faisabilité de l'encodeur (`essais/essai_encodeur_placement.py`) : un planificateur
   entièrement hors du GPU (encodeur et prédicteur sur le Neural Engine, CEM sur le CPU) ; sinon cette partie est retirée
   par un amendement daté, avant tout calcul.

Existant (recherches faites, citées) : K-021 (co-exécution GPU et NPU établie en général : HeteroInfer, HaX-CoNN ;
jamais pour un world model qui planifie), K-023 (aucun world model qui planifie mesuré sur le Neural Engine), K-024
(sebspel : vitesse et réussite de LeWM en FP16, FP8, INT8 sur GPU NVIDIA avec TensorRT, 50 épisodes). Nouveau ici : le
Neural Engine, les codages Core ML, la charge de vision, l'énergie par composant.

## Plan

- Tâche PushT, les 50 épisodes de la sonde (EXP-002), graine de la CEM 42 pour toutes les configurations (mêmes tirages ;
  seuls les calculs changent), CEM 300 x 30, horizon 5, tout le plan exécuté. LeWM publié, pleine précision des poids
  avant codage.
- Configurations sans charge : `torch` (float32, GPU), `torch-fp16` (autocast float16, GPU), `coreml-gpu` et `coreml-ne`
  en 16 bits (prédicteur en Core ML, le reste en PyTorch sur le GPU, comme EXP-012), et `coreml-gpu`, `coreml-ne` avec les
  codages `int8`, `int4b32`, `pal4` : 10 configurations.
- Configurations avec charge de vision (`charge_vision.py` : ResNet-50 à poids aléatoires, 224 x 224, lot de 1, float16,
  en boucle sur le GPU, démarré 30 s avant le calcul) : `torch`, `coreml-gpu`, `coreml-ne` en 16 bits, et la charge seule
  pendant 60 s (sa vitesse de référence, placée au hasard dans l'ordre comme une configuration) : 3 configurations et une
  mesure de référence.
- Trois répétitions ; dans chaque répétition, ordre des configurations tiré au hasard (`default_rng(13)`), pour que
  l'ordre et la chaleur ne se confondent pas avec la configuration. Avant chaque répétition : énergie au repos par
  composant (`repos_puce.py`, 45 s de stabilisation puis 75 s de mesure, aucun calcul en cours ; moins de 3 min sans
  calcul, pour que la garde de batterie ne croie pas le calcul arrêté).
- Environ 40 runs de 50 épisodes (1 min environ chacun), soit environ 1 h avec les mesures au repos.

## Mesures (descriptives, aucun test d'hypothèse)

- Temps par plan : somme des durées des appels de solve sur le nombre de plans, par run (appels allongés par une pause
  de la garde écartés, plus de 3 fois la médiane) ; résumé par configuration : médiane des 3 répétitions, minimum et
  maximum.
- Énergie par plan : énergie de la puce par composant (CPU, GPU, Neural Engine, mémoire ; compteurs IOReport, sans droits
  administrateur) sur la durée du run, moins l'énergie au repos de la répétition sur la même durée, divisée par le nombre
  de plans. Le run comprend la simulation et le rendu de l'environnement, identiques entre configurations. Avec charge,
  l'énergie mesurée est celle de la planification et de la vision ensemble.
- Vision : images par seconde de la charge pendant le run, contre sa vitesse seule.
- Mémoire : taille des poids Core ML de chaque codage (octets de weight.bin) et du modèle PyTorch.
- Réussite des 50 épisodes de chaque run : contrôle seulement (les codages ne sont pas évalués en réussite ici).

Comparaisons annoncées : `coreml-ne` contre `torch` et `coreml-gpu` (temps et énergie, sans charge) ; avec charge,
temps par plan et images par seconde de la vision pour les trois moteurs ; temps, énergie et taille par codage.
Formulations : « sur un MacBook Pro M5 Max, macOS 26.6.2 » ; énergie de la puce, pas de tout le Mac.

## Contrôles bloquants

Au lancement : arbre propre, commit unique ; le run `torch` de chaque répétition et un run `torch` final doivent être
identiques octet pour octet entre eux et au pilote d'EXP-012 (`resultats/pilote-exp012/torch/fp32_s42.npz`, mêmes 50
épisodes, même graine ; seuls les champs git diffèrent). Pour chaque run Core ML : toutes les opérations sur le processeur demandé (plan de calcul). Mesure au repos :
aucun calcul run_grid pendant la mesure. La charge de vision doit écrire au moins une mesure pendant chaque run avec charge.
