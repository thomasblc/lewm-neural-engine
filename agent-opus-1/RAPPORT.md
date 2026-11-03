# EXP-020, essai-opus-1 : rapport

Modèle : Claude Opus 5.5, identifiant exact `claude-opus-5-5`.

## Résultat
- `final.pt` (copie de `inst_Cp_5151.pt`) : 100/100 sur la calibration (évaluation n° 10), distance finale moyenne 0,0165, erreur 1 pas 0,072, 8 pas 0,368, 194 poids gardés, 2,802 bits par poids.
- Repères (references.json) : gptq2_k1 22/100, gptq3 70/100, gptq2_k1_affine (gradient) 94/100, fp32 100/100.
- Évaluations utilisées : 10 sur 30. Meilleur score de calibration : 100/100 (6 fichiers sur 10 l'atteignent).

## Démarche (forme close, float64, code dans `code/methode.py`)
Couche par couche, dans le réseau déjà compressé, sur 240 000 transitions du simulateur (graines à part) :
1. Moindres carrés ridge des entrées compressées vers la pré-activation du float32 (dernière couche : vers le vrai déplacement). Corrige l'erreur venue de l'amont.
2. Équilibrage par colonnes : la ReLU est positivement homogène, donc W ~ diag(s) C diag(t), et t est absorbé sans mémoire dans l'échelle, le biais et les poids gardés de la ligne de la couche précédente.
3. GPTQ (ordre des activations, colonne de biais libre en dernier), puis réajustement exact par ligne de l'échelle, du biais et des valeurs gardées, codes fixés. Deux passes fixes.
4. Pondération ReLU des couches cachées : un échantillon à cible négative pèse exp(-(y/sigma)^2/2), sigma l'erreur de la ligne après une première passe, car la ReLU coupe cette erreur.
5. Poids gardés : score OBS/SpQR multiplié par la sensibilité de la sortie à chaque ligne (jacobienne du float32 calculée à la main, poids 1 pour le bloc et 0,25 pour l'agent, comme le coût de la CEM), classement global des 194 entre couches. Pour final.pt : 114, 22, 6 et 52 par couche.

## Ce qui a marché (erreur 1 pas sur mes graines de validation, puis essai)
- Moindres carrés et réajustement (v1) : 0,36 -> 0,16.
- Équilibrage par colonnes (v2) : 0,16 -> 0,11, 98/100 au premier essai.
- Pondération ReLU (v4b) : 0,09, 100/100.
- Répartition globale pondérée (recette C) : erreur du bloc quand il bouge 0,18 -> 0,13 en moyenne sur 3 graines ; 7 instances C évaluées, toutes à 99 ou 100/100.

## Ce qui n'a pas marché
- Surpondérer les contacts, ajouter des trajectoires de poussée scriptées à la calibration, plus de données avec une 3e passe : pas de gain.
- Poids gardés en couches 0 et 3 sans pondération (v4a) : erreur à 8 pas plus basse mais 92/100. L'erreur du bloc compte bien plus que celle de l'agent.

## Constats
- Pour une même recette, le résultat varie beaucoup avec la graine des données (erreur bloc 1 pas de 0,038 à 0,074) : la variance des arrondis domine. J'ai comparé les recettes en moyenne sur 3 graines, puis choisi l'instance.
- Choix de final.pt : critère fixé avant de regarder (erreurs bloc + 0,5 x agent, 8 pas en boucle ouverte et le long de trajectoires de poussée, graines neuves). Les deux meilleures instances sont à 1,443 et 1,451 ; j'ai pris celle à 100/100 (l'autre rate l'épisode 97, raté aussi par 3 autres fichiers).
- La calibration est saturée (99 ou 100) : elle ne départage plus les variantes. Seul le test sur épisodes neufs dira si l'avance sur la référence par gradient tient.
- L'erreur de prédiction reste au-dessus de la référence par gradient (1 pas 0,072 contre 0,056), alors que la planification est meilleure.

## Règles
Aucun gradient, autograd ni optimiseur ; pas de boucle de recherche automatique (comparaisons à la main de 2 à 5 variantes) ; planification uniquement par `banc.py essai`. `banc.erreurs` appelé une fois en analyse initiale (gptq2, gptq2_k1, une couche à la fois). Lu dans c96 : fp32, gptq2, gptq2_k1, gptq3 (repère d'erreur) et references.json ; gptq2_k1_affine.pt non lu.
