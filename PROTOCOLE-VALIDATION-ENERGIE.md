# Validation des compteurs d'énergie IOReport contre la batterie (protocole figé avant la mesure)

Rédigé par Claude le 2 octobre 2026 à 8 h 44 (heure du Mac, lue avec `date`), avant toute mesure de validation.
Exigée par EXP-013 (pré-enregistrement, amendement 3 point 19, amendement 6 point 15, amendement 7 point 7) : aucun
chiffre d'énergie n'entre dans le papier sans elle. Script : `valider_energie.py` (règles recopiées dans sa docstring).

## Conditions

Chargeur débranché du début à la fin (contrôlé chaque seconde) ; Mac au repos sinon (seules l'app Claude et la mesure),
luminosité inchangée, personne ne touche au Mac ; écran gardé allumé (caffeinate -d -i), comme pendant EXP-013 ; macOS
26.6.2 ; PushT, graine 42.

## Segments (180 s mesurées chacun, environ 25 min en tout)

repos, torch (PyTorch float32 sur le GPU : calibration), coreml-gpu, ne-complet, ne-complet avec la vision ResNet-50 à
30 images/s, repos. Mesure 30 s après le premier plan (vision : 20 s après sa ligne « prête »).

## Grandeurs

P_b : puissance tirée de la batterie, - InstantAmperage x Voltage, moyenne des échantillons (1 par seconde) ; recoupée
par la capacité brute. P_p : somme des canaux IOReport PUCE d'EXP-013 (sans l'affichage), énergie / durée. Repos :
moyenne des deux segments de repos. Δ : segment moins repos.

## Règles d'acceptation

- c = ΔP_b(torch) / ΔP_p(torch) : pertes de conversion proportionnelles à la charge (rapporté).
- V1 : pour coreml-gpu, ne-complet et ne-complet-vision30, |c ΔP_p - ΔP_b| / ΔP_b au plus 10 %.
- V2 : pour (coreml-gpu, ne-complet), (torch, ne-complet) et (torch, coreml-gpu), le rapport des ΔP_p et celui des ΔP_b
  diffèrent d'au plus 10 % (indépendant de c ; les affirmations du papier sont des rapports au-dessus du repos).
- Segments valides : chargeur débranché, au moins 95 % des échantillons.
- Validée si tout passe. Sinon, aucun chiffre d'énergie dans le papier (mention « non validée »).

## Ce que la validation ne couvre pas

Le partage par composant (CPU, GPU, Neural Engine, mémoire) ; l'énergie marginale sous charge de vision répartie par
moteur ; une autre machine ou une autre version de macOS. Les niveaux absolus de l'énergie par plan d'EXP-013 restent
des estimations de la puce ; seuls les rapports au-dessus du repos sont validés.
