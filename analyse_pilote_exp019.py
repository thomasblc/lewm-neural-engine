"""EXP-019 (pilote, petit format) : réussite par version sur les 100 premières positions du tirage d'EXP-003 (graine 42),
écarts appariés, et contrôles de dérive contre les fichiers d'avant macOS 27 (mêmes positions, même graine).

Usage :
    python analyse_pilote_exp019.py
"""
import json
import math
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent.parent / "resultats"
K = 100
N_POIDS, LIGNES, GARDES = 10_764_288, 39_936, 107_640       # prédicteur de LeWM : 30 couches
VERSIONS = [("fp32", "exp019-pilote", 32.0),
            ("predictor:3", "exp019-pilote", 3 + 16 * LIGNES / N_POIDS),
            ("predictor:2:clip:gptq", "exp019-pilote", 2 + 16 * LIGNES / N_POIDS),
            ("predictor:2:clip:gptq:k1:obs", "exp019-pilote", 2 + (16 * LIGNES + 32 * GARDES) / N_POIDS),
            ("predictor:2:clip:gptq:k1:obs:aff", "exp019-pilote-aff", 2 + (16 * LIGNES + 32 * GARDES) / N_POIDS)]
ANCIENS = {"fp32": R / "exp003/fp32_s42.npz", "predictor:3": R / "exp003/predictor-3_s42.npz",
           "predictor:2:clip:gptq": R / "exp011-pusht/predictor-2-clip-gptq_s42.npz"}


def charger(spec, dossier):
    f = R / dossier / f"{spec.replace(':', '-')}_s42.npz"
    return np.load(f)["success"][:K].astype(bool) if f.exists() else None


def mcnemar(a, b):
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    n = n10 + n01
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(n10, n01) + 1)) / 2 ** n)
    return n10, n01, p


def main():
    res = {}
    for spec, dossier, bits in VERSIONS:
        s = charger(spec, dossier)
        if s is None:
            print(f"{spec:34s} absent")
            continue
        res[spec] = s
        ligne = f"{spec:34s} {100 * s.mean():5.1f} %  {bits:5.2f} bits/poids du prédicteur"
        if spec in ANCIENS and ANCIENS[spec].exists():
            ancien = np.load(ANCIENS[spec])["success"][:K].astype(bool)
            ligne += (f"  | avant macOS 27 : {100 * ancien.mean():5.1f} %, "
                      + ("identique épisode par épisode" if np.array_equal(ancien, s)
                         else f"{int((ancien != s).sum())} épisodes différents"))
        print(ligne)
    sortie = {"positions": K, "graine": 42, "reussite": {k: float(v.mean()) for k, v in res.items()}, "comparaisons": {}}
    for ref in ("fp32", "predictor:3", "predictor:2:clip:gptq"):
        for spec in res:
            if spec == ref or ref not in res:
                continue
            n10, n01, p = mcnemar(res[spec], res[ref])
            sortie["comparaisons"][f"{spec} contre {ref}"] = {"seul_reussi": n10, "ref_seule": n01, "p": p}
            print(f"  {spec} contre {ref} : {n10} réussis par lui seul, {n01} par la référence seule (p = {p:.2g})")
    (R / "exp019-pilote" / "analyse.json").write_text(json.dumps(sortie, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
