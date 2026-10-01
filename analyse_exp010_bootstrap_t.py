"""Complément d'EXP-010, prévu par l'amendement d'analyse du 1er octobre 2026 à 13 h 27 (`experiences/EXP-010-gptq-gratuit.md`,
horodaté par le commit 273f5eb de la branche exp012), écrit avant toute analyse : une étape de la séquence T1 à T4
n'est « établie » que si la p par inversion de signe (analyse_exp010.py, inchangée) et la p d'un bootstrap-t
(épisodes rééchantillonnés, B = 10 000, default_rng(4)) sont toutes deux sous 0,05. Lit les mêmes fichiers, avec les
mêmes contrôles, par analyse_exp010.charger.

Usage :
    python analyse_exp010_bootstrap_t.py --racine ../resultats --sortie ../resultats/analyse-exp010
"""
import argparse
import json
from pathlib import Path

import numpy as np

import analyse_exp003 as A
import analyse_exp005 as A5
import analyse_exp010 as A10


def p_bootstrap_t(y, b=10_000, graine=4):
    """p unilatérale (moyenne de y positive) d'un bootstrap-t : épisodes rééchantillonnés, statistique studentisée."""
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    m, se = y.mean(), y.std(ddof=1) / np.sqrt(n)
    t = m / se if se > 0 else (np.inf if m > 0 else -np.inf)
    yb = y[np.random.default_rng(graine).integers(0, n, size=(b, n))]
    seb = yb.std(axis=1, ddof=1) / np.sqrt(n)
    tb = (yb.mean(axis=1) - m) / np.where(seb > 0, seb, np.inf)
    return float((1 + int((tb >= t).sum())) / (b + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--racine", required=True)
    ap.add_argument("--sortie", required=True)
    a = ap.parse_args()
    res, sha = A10.charger(a.racine)
    T = {v: A5.table(res, v) for v in ("fp32",) + A10.VERSIONS}
    inv = A.Inversion(n=A10.N)
    sortie, ouvert, L = {}, True, []
    for nom, v, m in A10.SEQUENCE:
        x = T[v]["K"] - T["fp32"]["K"]
        y = 100 // (5 * m) * x + 1
        ps, pb = inv.p(y, "plus"), p_bootstrap_t(y)
        conclu = ouvert and ps < A.ALPHA and pb < A.ALPHA
        sortie[nom] = {"version": v, "marge_points": m, "p_signe": ps, "p_bootstrap_t": pb, "teste": ouvert,
                       "verdict": "établie" if conclu else ("non établie" if ouvert else "non testée (séquence arrêtée)")}
        L.append(f"| {nom} | {v} | {m} | {ps:.4f} | {pb:.4f} | {sortie[nom]['verdict']} |")
        ouvert = conclu
    out = Path(a.sortie)
    out.mkdir(parents=True, exist_ok=True)
    (out / "bootstrap_t.json").write_text(json.dumps({"commit_donnees": sha, "tests": sortie}, indent=1, ensure_ascii=False))
    txt = ("# EXP-010 : complément bootstrap-t (amendement d'analyse)\n\n| Test | Version | Marge (points) | p signe | p bootstrap-t | "
           "Verdict (les deux sous 0,05) |\n| --- | --- | --- | --- | --- | --- |\n" + "\n".join(L) + "\n")
    (out / "bootstrap_t.md").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
