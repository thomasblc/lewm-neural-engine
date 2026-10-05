# Pré-enregistrement d'EXP-018 : la perte d'int4 et ses deux réparations, sur 2 000 épisodes neufs pour la puce

Rédigé par Claude le 5 octobre 2026 à partir de 9 h 21 (heure du Mac, lue avec `date`), avant tout run d'EXP-018, à la
demande de Thomas (expérience de confirmation accordée le 5 octobre, 5 graines). Figé par le commit enregistré dans les
fichiers. Tâche : PushT. Existant : K-028 et K-030 ; recherche refaite le 5 octobre (aucun planificateur à world model
mesuré sur un NPU) ; EXP-018 est une réplication de nos propres résultats.

## Pourquoi

Les résultats sur la puce d'EXP-014, EXP-015 et EXP-017 portent sur les 500 épisodes d'EXP-003, et l'hypothèse
« `pred_proj` porte la perte d'int4 » a été formée en regardant EXP-014 sur ces mêmes épisodes (audit Codex du 3
octobre, P1). EXP-018 refait les trois versions qui portent les contributions du papier sur les 2 000 épisodes d'EXP-005,
qu'aucune expérience sur la puce n'a vus, ni la calibration de GPTQ.

## Ce qui est connu avant (divulgation)

- Sur les 500 épisodes d'EXP-003, contre PyTorch float32 : int4c -8,6 [-11,2 ; -6,2] (EXP-014) ; tout sauf `pred_proj`
  +0,6 [-0,7 ; +2,0] (EXP-015) ; codes de GPTQ -0,4 [-2,0 ; +1,3] (EXP-017) ; codes de GPTQ contre int4c +8,3 [+5,6 ;
  +11,0].
- Sur les 2 000 épisodes d'EXP-005 : pleine précision 87,1 % (graines 52 à 56) ; RTN 8 bits -0,4, RTN 6 bits -1,3
  (EXP-005) ; GPTQ à 4 bits sur tout le modèle, encodeur compris, -2,6 [-3,5 ; -1,7] (EXP-010). Aucun run Core ML ni
  sur le Neural Engine n'a tourné sur ces épisodes.

## Bras (PushT, les 2 000 épisodes d'EXP-005, graines 52 à 56, chunk 50)

Tirage des épisodes d'EXP-005 : `--graine-episodes 2026 --exclure resultats/exp003/meta.json resultats/sanite-n50/meta.json
--n 2000`. Prédicteur en Core ML sur le Neural Engine (`--moteur coreml-ne`), encodeur, CEM et coût en PyTorch sur le
GPU, comme EXP-014 à EXP-017.
- int4c : `--specs fp32 --codage int4c` (le modèle d'EXP-014) ;
- SP, tout sauf `pred_proj` : `--specs fp32 --codage int4c-sp` (le modèle d'EXP-015) ;
- G, codes de GPTQ : `--specs predict:4:gptq --codage int4c-gptq --gptq-calib resultats/exp009-calib-pusht/calib.pt` (le
  modèle d'EXP-017).
Référence : PyTorch float32 d'EXP-005 (`resultats/exp005/fp32_s52.npz` à `fp32_s56.npz`), mêmes épisodes et mêmes graines,
donc même hasard par épisode. Ordre : graine par graine, les trois bras à la suite. 15 fichiers de 2 000 épisodes,
environ 40 min chacun (temps des runs d'EXP-015), soit environ 10 h, plus les reproductions et les pauses de batterie.

Pas de bras 16 bits sur la puce : pour gagner environ 3 h, la référence est PyTorch float32 ; le 16 bits sur la puce est
à 2 points près de PyTorch float32 sur PushT (EXP-012) mais la contribution du passage au 16 bits n'est pas séparée ici.

## Tests (Holm sur 4, alpha 5 % ; inversion de signe et bootstrap-t concordants, p retenue la plus grande)

x(e) = K(e, bras) - K(e, référence), somme des succès sur les 5 graines. Tirages : `METHODES-NUMERIQUES` de l'audit
(inversion de signe `default_rng(0)` sur 2 000 colonnes ; bootstrap-t `default_rng(4)`, 10 000 tirages).
- T1, perte d'int4c : int4c contre float32 perd plus de 2 points ; y = -20 x - 2.
- T2, recette : SP contre float32 ne perd pas plus de 2 points ; y = 20 x + 2.
- T3, GPTQ face à la pleine précision : G contre float32 ne perd pas plus de 2 points ; y = 20 x + 2.
- T4, GPTQ face à l'arrondi simple : G contre int4c fait mieux de plus de 2 points ; y = 20 x - 2.
Bornes unilatérales au niveau de Holm par bootstrap-t (borne supérieure de l'écart pour T1, inférieure pour les autres) ;
un verdict n'est écrit que si borne et verdict concordent.

Puissance (approximation normale, écart-type par épisode de 15 à 20 points vu sur les 500 épisodes et sur EXP-010) : T1
et T4 presque 1 si les écarts valent ceux des 500 épisodes ; T2 presque 1 pour un écart vrai de 0 à +0,6 ; T3 0,96 si
l'écart vrai vaut -0,4 (500 épisodes), 0,60 s'il vaut -1,0 et 0,40 s'il vaut -1,2 (simulé, EXP-017), au niveau de Holm
le plus strict (0,0125).

## Descriptifs fixés (bootstrap des épisodes, `default_rng(2)`, 10 000 tirages)

Réussite de chaque bras ; écarts int4c, SP et G contre float32, G contre int4c, G contre SP ; part de la perte d'int4c
rattrapée par G et par SP, (bras - int4c) / (float32 - int4c), rapportée si la perte d'int4c est établie ; à côté, les
mêmes écarts sur les 500 épisodes d'EXP-014, EXP-015 et EXP-017 (sans test de leur différence) ; temps par plan de chaque
bras (descriptif, sans comparaison affirmée).

## Lecture écrite d'avance

- T1 et T2 soutenues : la perte d'int4 et sa réparation par `pred_proj` en 16 bits se retrouvent sur des épisodes neufs.
  T1 seule : int4c perd encore, la recette n'est pas établie à 2 points sur ces épisodes. T1 non soutenue : la perte
  d'int4c ne dépasse pas 2 points sur ces épisodes, et rien ne se lit sur sa localisation ici.
- T4 soutenue : les codes de GPTQ font mieux que l'arrondi simple de plus de 2 points sur des épisodes neufs. T3
  soutenue : ils ne perdent pas plus de 2 points face à PyTorch float32 ; non soutenue : non établi, jamais « perte »
  sans intervalle entièrement sous 0. Le résultat de G1 d'EXP-017 (contre le 16 bits sur la puce, 500 épisodes) reste
  rapporté tel quel à côté : deux tests différents, pas un remplacement.

## Contrôles bloquants

- Code : un seul commit pour tous les fichiers, arbre propre ; fichiers de calcul identiques à ceux du commit des runs
  d'EXP-017 (`8d24631`), dont les modèles ont passé les étapes 0 et 0 b.
- Reproduction de la pleine précision d'EXP-005, graine 52, sur les 2 000 épisodes, identique tableau pour tableau, au
  début ; reproduction d'EXP-003, graine 46, à la fin.
- Mêmes épisodes, départs et buts que la référence ; protocole identique à celui d'EXP-005, moteur et calibration
  exceptés ; calibration identique à celle d'EXP-010 (dont la disjonction avec ces épisodes est contrôlée par son
  analyse) ; couches quantifiées attendues (0 pour int4c et SP, 32 pour G).
- Pour chaque bras : tout sur le Neural Engine (placement), codage annoncé, une seule empreinte de modèle sur les 5
  graines, et cette empreinte égale à celle du même codage dans EXP-014, EXP-015 ou EXP-017 (même modèle).
- Contrôles de course et succès cohérents avec l'état (comme toutes les analyses).

## Limites écrites d'avance

Une tâche, un checkpoint ; une machine ; référence PyTorch float32, pas le 16 bits sur la puce ; les 2 000 épisodes ont
servi à EXP-005 et EXP-010 (versions PyTorch), pas à la puce ; pas de contrôle de la marge (pas de seconde série de la
pleine précision sur ces épisodes).

## Analyse et budget

`analyse_exp018.py`, testé par `test_analyse_exp018.py` sur des fichiers fabriqués à effet connu avant tout run.
Environ 10 h 30 de calcul.
