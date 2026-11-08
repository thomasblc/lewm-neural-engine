# EXP-005 : analyse pré-enregistrée

Contrôles passés : 15 fichiers (commit 4c53c4e), code d'analyse identique à ce commit (analyse lancée depuis 4c53c4e), sanité de clôture identique, 1333 épisodes où le T doit bouger, 233 où il part au but, courses divergentes : 0.

## H6 : à 6 bits, réussite à ± 2 points et pose du T dégradée

| Partie | Estimation, intervalle à 95 % | p |
| --- | --- | --- |
| (a) réussite, 2000 épisodes | -1.26 [-1.92, -0.61] points | bas 0.0121, haut 0.0001 |
| réussite, 1333 épisodes où le T doit bouger | -1.46 [-2.28, -0.65] points | |
| (b) log M3, mêmes 1333 épisodes | +0.020 [+0.007, +0.034] | 0.0014 |
| H6 (intersection-union) | | 0.0121 : **établie** |

Formulation retenue (règle fixée d'avance) : à 6 bits, la pose du T et la réussite se dégradent toutes deux, la réussite restant à ± 2 points sur l'ensemble des épisodes ; la dégradation est plus faible à 8 bits (progression selon les bits).
En unités physiques, dans la strate : position du T au plus près +0.20 [+0.02, +0.39] px, angle au plus près +0.21 [+0.01, +0.40] degrés (repères : la marge du T seul, M3, passe sous 1 à 20 px et 20 degrés ; le succès mesure les 20 px sur l'agent et le T ensemble).
Contrôle TOST t : p = 0.0139, intervalle à 90 % [-1.81, -0.71] points.

## Contrôle à 8 bits (descriptif, mêmes statistiques)

- Réussite : -0.36 [-0.85, +0.12] points ; log M3 dans la strate : +0.002 [-0.008, +0.012] (p unilatérale 0.3429).
- Progression selon les bits (écart de log M3 entre 6 et 8 bits dans la strate, décidée par l'intervalle) : +0.018 [+0.004, +0.032] : établie (p unilatérale 0.0037, sans valeur de décision).

## Descriptif

- fp32 : réussite 87.1 % ; classes dans la strate 86.2 / 5.8 / 8.1 (réussite / T placé seul / T jamais placé).
- all:6 : réussite 85.8 % ; classes dans la strate 84.7 / 6.5 / 8.8 (réussite / T placé seul / T jamais placé).
- all:8 : réussite 86.7 % ; classes dans la strate 85.7 / 6.2 / 8.1 (réussite / T placé seul / T jamais placé).
- Écart de log M2 à 6 bits : +0.008 [+0.000, +0.016] ; plancher de bruit : 4.75 [4.32, 5.17] épisodes discordants sur 50.

## À côté d'EXP-003 (sans regroupement)

- EXP-003, all:6 : réussite -0.3 [-1.6, +0.9] points ; dans la strate, réussite +0.2 [-1.3, +1.6], log M3 +0.030 [+0.003, +0.059].
- EXP-003, all:8 : réussite +0.3 [-0.8, +1.4] points ; dans la strate, réussite +0.5 [-0.8, +2.0], log M3 +0.005 [-0.016, +0.026].
