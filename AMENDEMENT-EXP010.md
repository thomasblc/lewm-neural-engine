## Amendement d'analyse (1er octobre 2026, 13 h 27, avant toute analyse et sans lecture des résultats)

Relecture statistique d'EXP-011 et d'EXP-012 (même test) : le test de non-infériorité par inversion de signe de la
statistique décalée n'est pas exact ; à la marge, les nombreux écarts nuls deviennent positifs et la queue des
pertes est longue, si bien que le taux d'erreur réel vaut 4,7 à 6,3 % pour 5 % nominal (simulations sur les données
d'EXP-003 et d'EXP-006), toujours dans le sens de « non inférieur ». Règle ajoutée pour T1 à T4 : une étape de la
séquence n'est « établie » que si la p par inversion de signe (pré-enregistrée) et la p d'un bootstrap-t
(B = 10 000, graine fixée) sont toutes deux sous 0,05. Le script pré-enregistré `analyse_exp010.py` reste inchangé et
tourne tel quel ; le bootstrap-t est calculé par un script complémentaire, commité avant son exécution. Les deux
verdicts seront rapportés.

