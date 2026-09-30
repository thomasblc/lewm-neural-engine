"""Analyse d'EXP-008 (PREREG-EXP008.md) : les cases manquantes de la matrice sur TwoRoom, descriptives, appariées
aux fichiers d'EXP-006 (mêmes épisodes, mêmes graines). Contrôles bloquants, écarts à fp32 avec intervalles.

Usage :
    python analyse_exp008.py --racine ../resultats --sortie ../resultats/analyse-exp008
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

import analyse_exp003 as A
import analyse_exp006 as A6

VERSIONS8 = ("all:5", "encoder:3", "predictor:3", "autres:3", "encpred:4", "all:3:g64")
COUCHES = {**A6.COUCHES, "encpred": 103}
FICHIERS_ANALYSE = ("analyse_exp008.py", "analyse_exp006.py", "analyse_exp003.py", "metriques.py", "PREREG-EXP008.md",
                    "run_grid.py", "quant.py")
ICI = Path(__file__).resolve().parent


def git(*args):
    return subprocess.run(["git", "-C", str(ICI), *args], capture_output=True, text=True)


def verifier_code(sha):
    A.exiger(git("cat-file", "-e", f"{sha}^{{commit}}").returncode == 0, f"commit {sha[:7]} inconnu du dépôt")
    A.exiger(git("diff", "--quiet", sha, "--", *FICHIERS_ANALYSE).returncode == 0,
             f"code d'analyse ou de calcul différent de celui du commit {sha[:7]}")


def identiques(fa, fb, nom):
    a, b = dict(np.load(fa)), dict(np.load(fb))
    A.exiger(set(a) == set(b), f"{nom} : clés différentes")
    for k in sorted(set(a) - {"git", "git_sale"}):
        A.exiger(np.array_equal(a[k], b[k], equal_nan=a[k].dtype.kind == "f"), f"{nom} : {k} diffère")


def charger(racine, verifier=True):
    r = Path(racine)
    m6, m8 = (json.loads((r / d / "meta.json").read_text())["protocole"] for d in ("exp006", "exp008"))
    A.exiger(m8.get("tache") == "tworoom" and m8["n"] == A6.N6, "exp008 : pas le protocole TwoRoom de 500 épisodes")
    A.exiger(m8 == m6, "exp008 : protocole différent de celui d'EXP-006 (épisodes, tirage, réglages ou appareil)")
    ref = dict(np.load(r / "exp006" / "fp32_s42.npz"))
    res, shas = {}, set()
    for v in ("fp32",) + VERSIONS8:
        for g in A6.GA:
            f = r / ("exp006" if v == "fp32" else "exp008") / f"{v.replace(':', '-')}_s{g}.npz"
            A.exiger(f.exists(), f"{f.parent.name}/{f.name} manquant")
            x = dict(np.load(f))
            A.exiger((str(x["spec"]), int(x["seed"])) == (v, g), f"{f.name} : mauvaise version ou graine")
            A.exiger(not bool(x["git_sale"]), f"{f.name} : arbre git modifié")
            A.exiger(int(x["n_couches"]) == COUCHES[v.split(":")[0]], f"{f.name} : couches quantifiées")
            for k in ("episodes_idx", "start_steps", "init_state", "goal_state"):
                A.exiger(np.array_equal(x[k], ref[k]), f"{v} graine {g} : {k} différent d'EXP-006")
            if v != "fp32":
                shas.add(str(x["git"]))
            res[(v, g)] = x
    A.exiger(len(shas) == 1, f"EXP-008 : fichiers de plusieurs commits {sorted(shas)}")
    sha = shas.pop()
    if verifier:
        verifier_code(sha)
    identiques(r / "exp008-repro" / "all-4_s42.npz", r / "exp006" / "all-4_s42.npz", "reproduction d'EXP-006")
    return res, sha


def analyser(racine, verifier=True):
    res, sha = charger(racine, verifier)
    K, lim = {}, 0
    for v in ("fp32",) + VERSIONS8:
        S = []
        for g in A6.GA:
            lim += A6.controler(res[(v, g)], f"{v} graine {g}")[1]
            S.append(res[(v, g)]["success"].astype(np.int64))
        K[v] = np.stack(S).sum(axis=0)
    boot = A.Bootstrap(n=A6.N6)
    desc = {v: {"reussite": float(K[v].mean() / 5), "ecart_points": boot.ic((K[v] - K["fp32"]) / 5)} for v in VERSIONS8}
    return {"controles": {"commit": sha, "fichiers": len(res), "cas_limites_arrondi": lim},
            "reussite_fp32": float(K["fp32"].mean() / 5), "descriptifs": desc}


def rapport(r):
    L = ["# EXP-008 : cases manquantes de la matrice sur TwoRoom (descriptif)", "",
         f"Contrôles passés : {r['controles']['fichiers']} fichiers, commit {r['controles']['commit'][:7]}, "
         f"reproduction d'EXP-006 identique. Pleine précision (EXP-006) : {100 * r['reussite_fp32']:.1f} %.", "",
         "| Version | Réussite | Écart à fp32 (points, IC 95 %) |", "| --- | --- | --- |"]
    for v, d in r["descriptifs"].items():
        L.append(f"| {v} | {100 * d['reussite']:.1f} % | {A.ic_txt(d['ecart_points'], 100, 1)} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--racine", required=True)
    ap.add_argument("--sortie", required=True)
    a = ap.parse_args()
    r = analyser(a.racine)
    out = Path(a.sortie)
    out.mkdir(parents=True, exist_ok=True)
    (out / "resultats.json").write_text(json.dumps(A.pur(r), indent=1, ensure_ascii=False))
    (out / "rapport.md").write_text(rapport(r))
    print((out / "rapport.md").read_text())


if __name__ == "__main__":
    main()
