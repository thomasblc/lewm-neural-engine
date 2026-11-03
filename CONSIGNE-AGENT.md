# EXP-020 : consigne donnée au sous-agent (texte envoyé tel quel)

Tu es l'optimiseur d'une expérience de recherche. Ton travail : améliorer la réussite de planification d'un petit
world model compressé en ternaire, en modifiant ses poids, à mémoire fixe, sans descente de gradient. Tu travailles
seul, sans poser de question ; réponds en français.

## Le banc d'essai

- Code (lecture seule, ne rien modifier) : `/Users/thomasblanc/Documents/PRO/these-ia-thomas/papier-corl-2026/code-exp020/`
  (`banc.py`, et `gptq.py`, `garder.py` qu'il importe). Lis la docstring de `banc.py` : monde « Pousse » (un disque
  en pousse un autre vers un but), world model MLP 6 -> 96 -> 96 -> 96 -> 4 qui prédit l'état suivant, planification
  CEM, format « compact » des poids compressés.
- Python : `/Users/thomasblanc/Documents/PRO/world-models-demos/le-wm/.venv/bin/python` (torch, numpy). Lance les
  commandes depuis le dossier du code (`cd` puis `python banc.py ...`).
- Données (lecture seule) dans `/Users/thomasblanc/Documents/PRO/these-ia-thomas/papier-corl-2026/resultats/exp020-banc/c96/` :
  - `fp32.pt` : les poids d'origine en float32 (state_dict du `Modele`) ;
  - `gptq2.pt` : ternaire par GPTQ, sans poids gardé ; `gptq2_k1.pt` : le même avec 194 poids gardés en float16
    (choisis par le critère de sensibilité de SpQR) ; deux points de départ possibles ;
  - `references.json` : scores des versions de référence sur les mêmes épisodes que tes évaluations.
- Ton dossier, le seul où tu écris : `/Users/thomasblanc/Documents/PRO/these-ia-thomas/papier-corl-2026/resultats/exp020-banc/c96/essai-opus-1/`.

## Ce que tu peux changer

Tout le contenu d'un fichier compact (dict `{"bits": 2, "couches": [...]}`, voir `banc.valider`) : les codes
ternaires de chaque poids (-1, 0, 1), l'échelle float16 de chaque ligne, quels poids sont gardés hors grille et leur
valeur float16, les biais float16. Contraintes vérifiées à chaque évaluation : `bits` = 2, codes dans {-1, 0, 1},
au plus 194 poids gardés au total (toutes couches confondues), valeurs finies en float16. La mémoire reste donc celle
du ternaire avec 1 % de poids gardés.

## Comment mesurer

- `python banc.py essai <ton dossier> <fichier.pt>` : valide le fichier puis mesure la réussite de planification sur
  100 épisodes de calibration (quelques secondes). Budget : 30 évaluations au total, comptées dans
  `<ton dossier>/evaluations.jsonl` ; au-delà, la commande refuse.
- `python banc.py erreur <fichier.pt>` : erreurs de prédiction à 1 et 8 pas, sans planification ; libre, non comptée.
- Tu peux écrire et lancer du code Python d'analyse : lire et comparer les poids, calculer des statistiques, faire
  passer des données dans le modèle (par exemple `banc.transitions` et `banc.pas`, le vrai simulateur), appliquer une
  formule en forme close que tu as choisie (moindres carrés, projection, etc.).

## Interdits

- Toute descente de gradient : `backward`, `torch.autograd`, `torch.optim`, ou tout optimiseur itératif équivalent.
- Toute recherche automatique : une boucle qui essaie de nombreuses variantes et garde la meilleure selon une mesure
  (recherche aléatoire, gloutonne, évolutionnaire). Comparer à la main jusqu'à 5 variantes avec `erreur` est permis.
- Appeler la planification hors de la commande `essai` (`banc.evaluer`, `banc.mesurer`, `banc.cout`, ou une copie).
- Modifier le code ou les fichiers de `c96/` hors de ton dossier ; lire d'autres dossiers de résultats.

## À rendre

1. `<ton dossier>/final.pt` : ton meilleur fichier compact (il doit passer `python banc.py erreur`).
2. `<ton dossier>/RAPPORT.md` (40 lignes au plus, en français) : ta démarche, les idées essayées, ce qui a marché ou
   non, le nombre d'évaluations utilisées, ton meilleur score de calibration, et l'identifiant exact du modèle que tu
   es (tel qu'indiqué dans tes instructions système).
3. Ta réponse finale : 10 lignes au plus, avec le meilleur score, le chemin de `final.pt` et l'identifiant du modèle.

Vise moins de 60 appels d'outils au total. Arrête-toi quand le budget d'évaluations est épuisé ou quand tu juges
qu'il n'y a plus de gain raisonnable.
