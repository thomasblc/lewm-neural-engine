"""Analyse pré-enregistrée d'EXP-009 (PREREG-EXP009.md) : GPTQ contre RTN, à mémoire égale, sur PushT et TwoRoom.
Contrôles bloquants, 4 tests appariés par inversion de signe (Holm), descriptifs.

Usage :
    python analyse_exp009.py --racine ../resultats --sortie ../resultats/analyse-exp009
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

import analyse_exp003 as A
import analyse_exp006 as A6

GRAINES = (42, 43, 44, 45, 46)
VERSIONS = ("all:4", "all:3", "encoder:4", "autres:4")
COUCHES = {"all": 109, "encoder": 73, "autres": 6}
TACHES = {  # dossier GPTQ, référence fp32, dossier RTN de chaque version, évaluations à exclure de la calibration
    "pusht": {"gptq": "exp009-pusht", "fp32": "exp003", "rtn": {"all:4": "exp003", "all:3": "exp003",
              "encoder:4": "exp003", "autres:4": "exp004"}, "evals": ("exp003", "exp005", "sanite-n50")},
    "tworoom": {"gptq": "exp009-tworoom", "fp32": "exp006", "rtn": {v: "exp006" for v in VERSIONS}, "evals": ("exp006",)},
}
TESTS = (("G1", "pusht", "all:4"), ("G2", "tworoom", "all:4"), ("G3", "pusht", "all:3"), ("G4", "tworoom", "all:3"))
FICHIERS_ANALYSE = ("analyse_exp009.py", "analyse_exp006.py", "analyse_exp003.py", "metriques.py", "gptq.py", "quant.py",
                    "run_grid.py", "verifier_gptq.py", "PREREG-EXP009.md")
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


def charger(racine, verifier=True, calib_torch=True):
    r = Path(racine)
    K, shas = {}, set()
    for t, c in TACHES.items():
        controler = A6.controler if t == "tworoom" else (lambda d, nom: A.controler_course(d, nom))
        ref = dict(np.load(r / c["fp32"] / "fp32_s42.npz"))
        identiques(r / f"exp009-repro-{t}" / "fp32_s42.npz", r / c["fp32"] / "fp32_s42.npz", f"reproduction {t}")
        proto = json.loads((r / c["gptq"] / "meta.json").read_text())["protocole"]
        fcal = r / f"exp009-calib-{t}" / "calib.pt"
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
        for v in VERSIONS:
            K[(t, v, "rtn")] = sum(lire(c["rtn"][v], v, g)["success"].astype(np.int64) for g in GRAINES)
            S = 0
            for g in GRAINES:
                x = lire(c["gptq"], f"{v}:gptq", g)
                A.exiger(int(x["n_couches"]) == COUCHES[v.split(":")[0]], f"{v}:gptq graine {g} : couches")
                shas.add(str(x["git"]))
                S = S + x["success"].astype(np.int64)
            K[(t, v, "gptq")] = S
    A.exiger(len(shas) == 1, f"EXP-009 : fichiers de plusieurs commits {sorted(shas)}")
    sha = shas.pop()
    if verifier:
        verifier_code(sha)
    return K, sha


def analyser(racine, verifier=True, calib_torch=True):
    K, sha = charger(racine, verifier, calib_torch)
    inv, boot = A.Inversion(n=500), A.Bootstrap(n=500)
    tests = {}
    for nom, t, v in TESTS:
        x = K[(t, v, "gptq")] - K[(t, v, "rtn")]
        tests[nom] = {"tache": t, "version": v, "somme": int(x.sum()), "ecart_points": boot.ic(x / 5), "p": inv.p(x, "plus")}
    for (nom, *_), q in zip(TESTS, A.holm([x["p"] for x in tests.values()])):
        tests[nom]["p_holm"] = float(q)
        tests[nom]["verdict"] = "soutenue" if q < A.ALPHA else "non soutenue"
    desc = {}
    for t in TACHES:
        for v in VERSIONS:
            g, r_, f = K[(t, v, "gptq")], K[(t, v, "rtn")], K[(t, "fp32")]
            part = lambda i: (g[i].sum() - r_[i].sum()) / max(f[i].sum() - r_[i].sum(), 1)
            parts = np.array([part(i) for i in boot.idx])
            desc[f"{t} {v}"] = {"reussite_gptq": float(g.mean() / 5), "reussite_rtn": float(r_.mean() / 5),
                                "reussite_fp32": float(f.mean() / 5), "gptq_moins_rtn": boot.ic((g - r_) / 5),
                                "part_rattrapee": {"moyenne": float((g.sum() - r_.sum()) / max(f.sum() - r_.sum(), 1)),
                                                   "bas": float(np.percentile(parts, 2.5)),
                                                   "haut": float(np.percentile(parts, 97.5))}}
    return {"controles": {"commit": sha}, "tests": tests, "descriptifs": desc}


def rapport(r):
    t = lambda e, f=100: A.ic_txt(e, f, 1)
    L = ["# EXP-009 : analyse pré-enregistrée (GPTQ contre RTN, à mémoire égale)", "",
         f"Contrôles passés, commit {r['controles']['commit'][:7]}.", "",
         "| Test | Tâche, version | GPTQ moins RTN (points) | p | p Holm | Verdict |", "| --- | --- | --- | --- | --- | --- |"]
    for n, x in r["tests"].items():
        L.append(f"| {n} | {x['tache']} {x['version']} | {t(x['ecart_points'])} | {x['p']:.4f} | {x['p_holm']:.4f} | {x['verdict']} |")
    L += ["", "| Tâche, version | fp32 | RTN | GPTQ | GPTQ moins RTN | Part de la perte rattrapée |", "| --- | --- | --- | --- | --- | --- |"]
    for k, d in r["descriptifs"].items():
        L.append(f"| {k} | {100 * d['reussite_fp32']:.1f} % | {100 * d['reussite_rtn']:.1f} % | {100 * d['reussite_gptq']:.1f} % | "
                 f"{t(d['gptq_moins_rtn'])} | {t(d['part_rattrapee'])} % |")
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
