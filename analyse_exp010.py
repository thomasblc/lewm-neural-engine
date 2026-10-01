"""Analyse pré-enregistrée d'EXP-010 (PREREG-EXP010.md) : GPTQ à 4 et 3 bits est-il gratuit ? PushT, les 2 000
épisodes et la pleine précision d'EXP-005, calibration PushT d'EXP-009. Quatre tests de non-infériorité en séquence
fixée (arrêt au premier échec).

Usage :
    python analyse_exp010.py --racine ../resultats --sortie ../resultats/analyse-exp010
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

import analyse_exp003 as A
import analyse_exp005 as A5

VERSIONS, GRAINES, N = ("all:4:gptq", "all:3:gptq"), A5.GRAINES, A5.N5
SEQUENCE = (("T1", "all:4:gptq", 2), ("T2", "all:4:gptq", 1), ("T3", "all:3:gptq", 2), ("T4", "all:3:gptq", 1))
CALIB = "exp009-calib-pusht/calib.pt"
FICHIERS_ANALYSE = ("analyse_exp010.py", "analyse_exp005.py", "analyse_exp003.py", "metriques.py", "run_grid.py", "quant.py",
                    "gptq.py", "PREREG-EXP010.md")
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
    m5, m10 = (json.loads((r / d / "meta.json").read_text())["protocole"] for d in ("exp005", "exp010"))
    cal = m10.pop("gptq_calib", None)
    A.exiger(m10 == m5, "exp010 : protocole différent de celui d'EXP-005")
    A.exiger(cal is not None and cal["sha256"] == hashlib.sha256((r / CALIB).read_bytes()).hexdigest(),
             "exp010 : calibration différente de celle d'EXP-009")
    import torch
    c = torch.load(r / CALIB, map_location="cpu")
    evals = set(m5["episodes_idx"]) | {a for a, _ in m5["exclus"]}
    A.exiger(c["tache"] == "pusht" and not (set(c["episodes_idx"]) & evals),
             "exp010 : calibration et évaluations partagent des trajectoires")
    identiques(r / "exp010-repro" / "fp32_s52.npz", r / "exp005" / "fp32_s52.npz", "reproduction d'EXP-005")
    ref = dict(np.load(r / "exp005" / "fp32_s52.npz"))
    res, shas = {}, set()
    for g in GRAINES:
        res[("fp32", g)] = dict(np.load(r / "exp005" / f"fp32_s{g}.npz"))
        for v in VERSIONS:
            f = r / "exp010" / f"{v.replace(':', '-')}_s{g}.npz"
            A.exiger(f.exists(), f"exp010/{f.name} manquant")
            x = dict(np.load(f))
            A.exiger((str(x["spec"]), int(x["seed"])) == (v, g), f"{f.name} : version ou graine")
            A.exiger(not bool(x["git_sale"]), f"{f.name} : arbre git modifié")
            A.exiger(int(x["n_couches"]) == 109, f"{f.name} : {int(x['n_couches'])} couches quantifiées au lieu de 109")
            for k in ("episodes_idx", "start_steps", "init_state", "goal_state"):
                A.exiger(np.array_equal(x[k], ref[k]), f"{f.name} : {k} différent d'EXP-005")
            shas.add(str(x["git"]))
            res[(v, g)] = x
    A.exiger(len(shas) == 1, f"EXP-010 : fichiers de plusieurs commits {sorted(shas)}")
    sha = shas.pop()
    if verifier:
        verifier_code(sha)
    return res, sha


def analyser(racine, verifier=True, exp005=None):
    res, sha = charger(racine, verifier)
    T = {v: A5.table(res, v) for v in ("fp32",) + VERSIONS}
    dep = A.depart({("fp32", A.GA[0]): res[("fp32", GRAINES[0])]})
    bouge = dep["bouge"]
    A.exiger((int(bouge.sum()), int(dep["au_but"].sum())) == A5.EFFECTIFS5, "strates de départ inattendues")
    inv, boot, boot_s = A.Inversion(n=N), A.Bootstrap(n=N), A.Bootstrap(n=int(bouge.sum()), graine=3)
    tests, ouvert = {}, True
    for nom, v, m in SEQUENCE:                   # séquence fixée : un test n'est conclu que si les précédents l'ont été
        x = T[v]["K"] - T["fp32"]["K"]
        p = inv.p(100 // (5 * m) * x + 1, "plus")  # y = 10 x + 1 (m = 2 points) ou 20 x + 1 (m = 1 point)
        conclu = ouvert and p < A.ALPHA
        tests[nom] = {"version": v, "marge_points": m, "p": p, "teste": ouvert,
                      "verdict": "établie" if conclu else ("non établie" if ouvert else "non testée (séquence arrêtée)")}
        ouvert = conclu
    desc = {"reussite_fp32": float(T["fp32"]["S"].mean())}
    for v in VERSIONS:
        x = T[v]["K"] - T["fp32"]["K"]
        d3 = A.lg(T[v]["M3"]).mean(axis=0) - A.lg(T["fp32"]["M3"]).mean(axis=0)
        desc[v] = {"reussite": float(T[v]["S"].mean()), "ecart_reussite": boot.ic(x / 5), "ecart_log_M3_bouge": boot_s.ic(d3[bouge]),
                   "dT_min_px": boot_s.ic((T[v]["dT_min"].mean(axis=0) - T["fp32"]["dT_min"].mean(axis=0))[bouge]),
                   "dth_min_deg": boot_s.ic((T[v]["dth_min"].mean(axis=0) - T["fp32"]["dth_min"].mean(axis=0))[bouge])}
    comparaison = None
    if exp005:
        r5 = json.loads(Path(exp005).read_text())
        comparaison = {"all:6": r5["H6"]["ecarts"]["reussite"], "all:8": r5["controle_8_bits"]["ecarts"]["reussite"]}
    return {"controles": {"commit": sha, "divergences": {v: T[v]["divergences"] for v in VERSIONS}},
            "tests": tests, "descriptifs": desc, "comparaison_exp005": comparaison}


def rapport(r):
    t = lambda e, f=100, d=2: A.ic_txt(e, f, d)
    d = r["descriptifs"]
    L = ["# EXP-010 : GPTQ à 4 et 3 bits est-il gratuit ? (PushT, 2 000 épisodes neufs)", "",
         f"Contrôles passés, commit {r['controles']['commit'][:7]}, reproduction d'EXP-005 identique, calibration d'EXP-009 "
         f"disjointe des évaluations. Pleine précision : {100 * d['reussite_fp32']:.1f} %.", "",
         "| Test (séquence fixée) | Version | Perte de moins de | p | Verdict |", "| --- | --- | --- | --- | --- |"]
    for n, x in r["tests"].items():
        L.append(f"| {n} | {x['version']} | {x['marge_points']} point{'s' if x['marge_points'] > 1 else ''} | {x['p']:.4f} | {x['verdict']} |")
    L.append("")
    for v in VERSIONS:
        e = d[v]
        L.append(f"- {v} : réussite {100 * e['reussite']:.1f} %, écart {t(e['ecart_reussite'])} points ; pose (log M3, strate) "
                 f"{t(e['ecart_log_M3_bouge'], 1, 3)} ; position {t(e['dT_min_px'], 1, 2)} px, angle {t(e['dth_min_deg'], 1, 2)} degrés.")
    if r["comparaison_exp005"]:
        c = r["comparaison_exp005"]
        L.append(f"- À côté (EXP-005, RTN, tout le modèle) : 8 bits {t(c['all:8'])} points, 6 bits {t(c['all:6'])} points.")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--racine", required=True)
    ap.add_argument("--exp005")
    ap.add_argument("--sortie", required=True)
    a = ap.parse_args()
    r = analyser(a.racine, exp005=a.exp005)
    out = Path(a.sortie)
    out.mkdir(parents=True, exist_ok=True)
    (out / "resultats.json").write_text(json.dumps(A.pur(r), indent=1, ensure_ascii=False))
    (out / "rapport.md").write_text(rapport(r))
    print((out / "rapport.md").read_text())


if __name__ == "__main__":
    main()
