# EXP-019, pilote : prédicteur ternaire avec quelques poids gardés en 16 bits (exploratoire, non pré-enregistré)

3 novembre 2026. Question : en gardant 0,5 à 2 % des poids du prédicteur hors de la grille ternaire, en 16 bits,
rattrape-t-on la perte du ternaire avec GPTQ sur PushT (-30,5 points, EXP-011), et cela bat-il le prédicteur à 3 bits
à mémoire égale ? Pilote pour décider s'il faut une expérience pré-enregistrée ; aucun chiffre n'ira dans un article.

## Versions (PushT, 200 premières positions du tirage de 500 d'EXP-003 et 011, graines 42 et 43, hasard apparié)

| Version | Rôle | Bits par poids du prédicteur (ternaire stocké sur 2 bits, échelles 16 bits par ligne, poids gardé = 16 bits de valeur + 16 bits d'adresse) |
| --- | --- | --- |
| fp32 | référence ; comparée aux fichiers d'EXP-011 (contrôle de dérive après macOS 27) | 32 |
| predictor:2:clip:gptq | ternaire GPTQ d'EXP-011 | environ 2,0 |
| predictor:3, predictor:3:gptq | prédicteur à 3 bits, la barre à mémoire égale | environ 3,0 |
| predictor:2:clip:gptq:k{0.5,1,2}:{mag,obs} | poids gardés, choisis par la taille ou par la sensibilité de SpQR | environ 2,16 ; 2,32 ; 2,64 |
| predictor:2:clip:k1:obs | contrôle : poids gardés sans GPTQ | environ 2,32 |

## Ce qu'on regarde

Réussite par version, écart apparié à fp32 et au 3 bits, nombre d'épisodes qui basculent. Prometteur si une version à
au plus 2,64 bits par poids perd au plus 7,6 points (rattrape au moins les trois quarts de la perte d'EXP-011) et fait
mieux que le critère de taille au même pourcentage. Sinon, on s'arrête ou on passe au critère guidé par la tâche.

## Limites connues

Mêmes échelles qu'EXP-011 (calculées avec les poids gardés) ; même fraction gardée dans chaque couche ; critère « obs »
calculé sur les poids d'origine, pas sur les poids mis à jour par GPTQ comme dans SpQR ; 400 épisodes par version.

## Changements après la recherche de l'existant (K-037, 3 novembre 2026), avant tout calcul

1. Ajouter un point à 3,1 % de poids gardés (égalité de mémoire avec le 3 bits, convention SpQR de 32 bits par poids
   gardé) et donner aussi les bits avec un ternaire rangé à 1,6 bit.
2. Ajouter deux références au même budget : GPTQ ternaire à groupes plus fins, GPTQ 2 bits à quatre niveaux.
3. Ajouter le critère « spqr » : poids gardés choisis pendant GPTQ, sur les poids mis à jour.
4. Deuxième temps seulement : critère de tâche à la TaCQ (perte de rollout, gradient absolu, épisodes de calibration
   distincts de l'évaluation) et réglage fin des seuls poids gardés.
Rien n'est lancé : l'environnement de LeWM ne tourne plus depuis macOS 27 (voir le journal du 3 novembre).


## Amendement du 4 novembre 2026 : petit format, avec réglage fin (leçons d'EXP-020), avant tout calcul

Décision de Thomas : lancer EXP-019 sans être gourmand en taille. macOS 27.2 a réparé scipy. Plan réduit :

- PushT, 100 premières positions du tirage d'EXP-003, graine 42 seulement.
- 5 versions : pleine précision ; prédicteur 3 bits (RTN, la barre à mémoire plus haute, 3,06 bits) ; ternaire GPTQ
  d'EXP-011 (2,06 bits) ; ternaire GPTQ avec 1 % de poids gardés, critère SpQR (2,38 bits) ; la même avec réglage fin
  (2,38 bits, même mémoire).
- Réglage fin (leçon d'EXP-020) : entiers et masque figés ; on règle un facteur par ligne sur l'échelle, les valeurs
  gardées et les biais, pour que la sortie du prédicteur (puis pred_proj, dernière position) reproduise celle du float32.
  Données : entrées du prédicteur enregistrées pendant une planification en pleine précision sur les 32 premiers
  épisodes de calibration d'EXP-009 (tirage 9009, trajectoires d'évaluation exclues), 8 lignes par appel. Réglage fixé
  sans essai : 5 000 pas, lots de 256, Adam 1e-3 avec décroissance cosinus, graine 0 ; 5 % des lignes tenues à l'écart
  pour une perte de contrôle. Le format est vérifié au chargement (mêmes entiers, une échelle float16 par ligne, valeurs
  gardées en float16, même masque).
- Contrôles de dérive : la pleine précision et le ternaire GPTQ sont comparés épisode par épisode aux fichiers d'avant
  macOS 27 (EXP-003, EXP-011) ; arrêt si la pleine précision s'écarte de plus de 10 points.
- Lecture : prometteur si la version réglée perd au plus 7,6 points contre la pleine précision (critère d'origine) et
  fait mieux que le 1 % gardé sans réglage. 100 épisodes : un écart de moins de 10 points n'est pas établi.
- Lanceur : `lancer_pilote_exp019.sh` ; analyse : `analyse_pilote_exp019.py`.
