"""Analyse pré-enregistrée d'EXP-011 (PREREG-EXP011.md) : le prédicteur à 2 bits, GPTQ contre RTN et contre la
pleine précision, sur PushT et TwoRoom. Contrôles bloquants, 4 tests appariés par inversion de signe (Holm),
descriptifs.

Usage :
    python analyse_exp011.py --racine ../resultats --sortie ../resultats/analyse-exp011
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

import analyse_exp003 as A
import analyse_exp006 as A6
import analyse_exp009 as A9

GRAINES = (42, 43, 44, 45, 46)
VERSIONS = ("predictor:2", "predictor:2:gptq")
COUCHES = 30
MARGE = 3                                           # points, test de non-infériorité
TACHES = {  # dossier, référence fp32, calibration d'EXP-009, évaluations à exclure de la calibration
    "pusht": {"dossier": "exp011-pusht", "fp32": "exp003", "calib": "exp009-calib-pusht", "evals": ("exp003", "exp005", "sanite-n50")},
    "tworoom": {"dossier": "exp011-tworoom", "fp32": "exp006", "calib": "exp009-calib-tworoom", "evals": ("exp006",)},
}
TESTS = (("G1", "pusht", "gptq_rtn"), ("G2", "tworoom", "gptq_rtn"), ("N1", "pusht", "gptq_fp32"), ("N2", "tworoom", "gptq_fp32"))
FICHIERS_ANALYSE = ("analyse_exp011.py", "analyse_exp009.py", "analyse_exp006.py", "analyse_exp003.py", "metriques.py",
                    "gptq.py", "quant.py", "run_grid.py", "verifier_gptq.py", "PREREG-EXP011.md")
ICI = Path(__file__).resolve().parent


def git(*args):
    return subprocess.run(["git", "-C", str(ICI), *args], capture_output=True, text=True)


def verifier_code(sha):
    A.exiger(git("cat-file", "-e", f"{sha}^{{commit}}").returncode == 0, f"commit {sha[:7]} inconnu du dépôt")
    A.exiger(git("diff", "--quiet", sha, "--", *FICHIERS_ANALYSE).returncode == 0,
             f"code d'analyse ou de calcul différent de celui du commit {sha[:7]}")


def charger(racine, verifier=True, calib_torch=True):
    r = Path(racine)
    K, shas = {}, set()
    for t, c in TACHES.items():
        controler = A6.controler if t == "tworoom" else (lambda d, nom: A.controler_course(d, nom))
        ref = dict(np.load(r / c["fp32"] / "fp32_s42.npz"))
        A9.identiques(r / f"exp011-repro-{t}" / "fp32_s42.npz", r / c["fp32"] / "fp32_s42.npz", f"reproduction {t}")
        proto = json.loads((r / c["dossier"] / "meta.json").read_text())["protocole"]
        fcal = r / c["calib"] / "calib.pt"
        A.exiger(proto.get("gptq_calib", {}).get("sha256") == hashlib.sha256(fcal.read_bytes()).hexdigest(),
                 f"{t} : calibration différente de celle du protocole")
        A.exiger(proto.get("tache", "pusht") == t and proto["n"] == 500, f"{t} : protocole")
        if calib_torch:
            import torch
            cal = torch.load(fcal, map_location="cpu")
            vus = set()
            for e in c["evals"]:
                m = json.loads((r / e / "meta.json").read_text())
                vus |= set(m.get("protocole", m)["episodes_idx"])
            A.exiger(not (set(cal["episodes_idx"]) & vus), f"{t} : calibration et évaluation partagent des trajectoires")
            A.exiger(cal["tache"] == t, f"{t} : calibration d'une autre tâche")

        def lire(dossier, v, g):
            f = r / dossier / f"{v.replace(':', '-')}_s{g}.npz"
            A.exiger(f.exists(), f"{dossier}/{f.name} manquant")
            x = dict(np.load(f))
            A.exiger((str(x["spec"]), int(x["seed"])) == (v, g), f"{dossier}/{f.name} : version ou graine")
            A.exiger(not bool(x["git_sale"]), f"{dossier}/{f.name} : arbre git modifié")
            for k in ("episodes_idx", "start_steps", "init_state", "goal_state"):
                A.exiger(np.array_equal(x[k], ref[k]), f"{dossier}/{f.name} : {k} différent de la référence")
            controler(x, f"{dossier}/{f.name}")
            return x
        K[(t, "fp32")] = sum(lire(c["fp32"], "fp32", g)["success"].astype(np.int64) for g in GRAINES)
        if t == "pusht":
            K[(t, "predictor:3")] = sum(lire("exp003", "predictor:3", g)["success"].astype(np.int64) for g in GRAINES)
        for v in VERSIONS:
            S = 0
            for g in GRAINES:
                x = lire(c["dossier"], v, g)
                A.exiger(int(x["n_couches"]) == COUCHES, f"{t} {v} graine {g} : {int(x['n_couches'])} couches au lieu de {COUCHES}")
                shas.add(str(x["git"]))
                S = S + x["success"].astype(np.int64)
            K[(t, v)] = S
    A.exiger(len(shas) == 1, f"EXP-011 : fichiers de plusieurs commits {sorted(shas)}")
    sha = shas.pop()
    if verifier:
        verifier_code(sha)
    return K, sha


def analyser(racine, verifier=True, calib_torch=True):
    K, sha = charger(racine, verifier, calib_torch)
    inv, boot = A.Inversion(n=500), A.Bootstrap(n=500)
    tests = {}
    for nom, t, quoi in TESTS:
        g = K[(t, "predictor:2:gptq")]
        if quoi == "gptq_rtn":
            x = g - K[(t, "predictor:2")]
            p = inv.p(x, "plus")
        else:
            x = g - K[(t, "fp32")]
            p = inv.p(20 * x + MARGE, "plus")       # y = 20 x + 3 : moyenne positive ssi écart de réussite > -3 points
        tests[nom] = {"tache": t, "comparaison": quoi, "somme": int(x.sum()), "ecart_points": boot.ic(x / 5), "p": p}
    for (nom, *_), q in zip(TESTS, A.holm([x["p"] for x in tests.values()])):
        tests[nom]["p_holm"] = float(q)
        tests[nom]["verdict"] = "soutenue" if q < A.ALPHA else "non soutenue"
    desc = {}
    for t in TACHES:
        g, r_, f = K[(t, "predictor:2:gptq")], K[(t, "predictor:2")], K[(t, "fp32")]
        den = np.array([f[i].sum() - r_[i].sum() for i in boot.idx], dtype=np.float64)
        num = np.array([g[i].sum() - r_[i].sum() for i in boot.idx], dtype=np.float64)
        parts = num[den > 0] / den[den > 0]
        sens = f.sum() - r_.sum() >= 0.02 * 5 * len(f)  # part rattrapée sans objet si RTN perd moins de 2 points
        desc[t] = {"reussite_fp32": float(f.mean() / 5), "reussite_rtn": float(r_.mean() / 5), "reussite_gptq": float(g.mean() / 5),
                   "gptq_moins_rtn": boot.ic((g - r_) / 5), "gptq_moins_fp32": boot.ic((g - f) / 5),
                   "part_rattrapee": None if not sens else
                   {"moyenne": float((g.sum() - r_.sum()) / (f.sum() - r_.sum())),
                    "bas": float(np.percentile(parts, 2.5)), "haut": float(np.percentile(parts, 97.5))}}
    desc["pusht"]["gptq2_moins_rtn3"] = boot.ic((K[("pusht", "predictor:2:gptq")] - K[("pusht", "predictor:3")]) / 5)
    desc["pusht"]["reussite_rtn3"] = float(K[("pusht", "predictor:3")].mean() / 5)
    return {"controles": {"commit": sha}, "tests": tests, "descriptifs": desc}


def rapport(r):
    t = lambda e, f=100: A.ic_txt(e, f, 1)
    L = ["# EXP-011 : analyse pré-enregistrée (le prédicteur à 2 bits, GPTQ)", "",
         f"Contrôles passés, commit {r['controles']['commit'][:7]}.", "",
         "| Test | Tâche | Comparaison | Écart (points) | p | p Holm | Verdict |", "| --- | --- | --- | --- | --- | --- | --- |"]
    noms = {"gptq_rtn": "GPTQ plus haut que RTN", "gptq_fp32": f"GPTQ perd moins de {MARGE} points"}
    for n, x in r["tests"].items():
        L.append(f"| {n} | {x['tache']} | {noms[x['comparaison']]} | {t(x['ecart_points'])} | {x['p']:.4f} | {x['p_holm']:.4f} | {x['verdict']} |")
    L += ["", "| Tâche | fp32 | RTN 2 bits | GPTQ 2 bits | GPTQ moins RTN | GPTQ moins fp32 | Part rattrapée |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for k, d in r["descriptifs"].items():
        L.append(f"| {k} | {100 * d['reussite_fp32']:.1f} % | {100 * d['reussite_rtn']:.1f} % | {100 * d['reussite_gptq']:.1f} % | "
                 f"{t(d['gptq_moins_rtn'])} | {t(d['gptq_moins_fp32'])} | "
                 f"{t(d['part_rattrapee']) + ' %' if d['part_rattrapee'] else 'sans objet'} |")
    d = r["descriptifs"]["pusht"]
    L += ["", f"PushT, GPTQ à 2 bits moins RTN à 3 bits (EXP-003, {100 * d['reussite_rtn3']:.1f} %) : {t(d['gptq2_moins_rtn3'])} points."]
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
