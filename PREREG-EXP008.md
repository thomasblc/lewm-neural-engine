# Pré-enregistrement d'EXP-008 : les cases manquantes de la matrice sur TwoRoom (descriptif)

Rédigé par Claude le 30 septembre 2026 vers 16 h 40, à la demande de Thomas (combler les trous de TwoRoom avant
GPTQ), avant tout calcul d'EXP-008. Figé par le commit enregistré dans les fichiers.

- But : compléter la matrice tâche × compression pour comparer les deux tâches partie par partie, sur les cases
  jouées pour PushT (EXP-003, EXP-004) mais pas pour TwoRoom (EXP-006).
- Versions : all:5, encoder:3, predictor:3, autres:3, encpred:4, all:3:g64 (RTN par canal ou groupes de 64,
  `quant.py` d'EXP-004). Graines 42 à 46. Les 500 épisodes d'EXP-006 (même tirage : graine 42, une position par
  trajectoire) ; la pleine précision de référence est celle d'EXP-006. 30 fichiers, environ 5 h.
- Descriptif seulement, pas de test d'hypothèse : réussite par version, écart à fp32 avec intervalle bootstrap
  percentile (`default_rng(2)`), mis en regard des valeurs de PushT. Aucune affirmation de l'article ne reposera
  sur une case sans intervalle.
- Contrôles bloquants : reproduction d'EXP-006 avant la grille (all:4 graine 42 recalculé avec le code actuel,
  identique octet pour octet au fichier d'EXP-006) ; protocole et épisodes identiques à EXP-006 ; couches
  quantifiées ; contrôles de course d'EXP-006 ; un seul commit, dont le code d'analyse et de calcul est celui du
  dépôt. Pas de sanité de clôture : la reproduction d'entrée vérifie déjà le code et la machine.
