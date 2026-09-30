"""Analyse pré-enregistrée d'EXP-007 (PREREG-EXP007.md) : replanifier plus souvent rattrape-t-il la compression ?
PushT, fp32 et all:4, graines 42 à 44 ; la condition de référence (replanifier tous les 25 pas) est reprise
d'EXP-003. Contrôles bloquants, 3 tests appariés par inversion de signe (Holm), descriptifs.

Usage :
    python analyse_exp007.py --racine ../resultats --sanite-fin ../resultats/exp007-sanite-fin \
        --sortie ../resultats/analyse-exp007
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

import analyse_exp003 as A

N7, GRAINES, VERSIONS = 500, (42, 43, 44), ("fp32", "all:4")
CONDITIONS = {                                  # nom : (dossier, blocs de 5 pas avant de replanifier, itérations CEM)
    "r25": ("exp003", 5, 30),                   # référence d'EXP-003 : 2 plans par épisode
    "r5c6": ("exp007-r5c6", 1, 6),              # replanifier tous les 5 pas, CEM 5 fois plus courte : calcul égal
    "r5": ("exp007-r5", 1, 30),                 # replanifier tous les 5 pas, CEM complète : 5 fois plus de calcul
    "r10": ("exp007-r10", 2, 30),               # tous les 10 pas, CEM complète (descriptif)
}
COUCHES = {"fp32": 0, "all:4": 109}
FICHIERS_ANALYSE = ("analyse_exp007.py", "analyse_exp003.py", "PREREG-EXP007.md", "run_grid.py", "quant.py")
ICI = Path(__file__).resolve().parent


def git(*args):
    return subprocess.run(["git", "-C", str(ICI), *args], capture_output=True, text=True)


def verifier_code(sha):
    A.exiger(git("cat-file", "-e", f"{sha}^{{commit}}").returncode == 0, f"commit {sha[:7]} inconnu du dépôt")
    A.exiger(git("diff", "--quiet", sha, "--", *FICHIERS_ANALYSE).returncode == 0,
             f"code d'analyse ou de calcul différent de celui du commit {sha[:7]}")


def charger(racine, verifier=True):
    """{(condition, version, graine): fichier}, après les contrôles ; renvoie aussi le commit d'EXP-007."""
    res, shas = {}, set()
    for c, (dossier, rh, it) in CONDITIONS.items():
        d = Path(racine) / dossier
        proto = json.loads((d / "meta.json").read_text())["protocole"]
        A.exiger(proto["n"] == N7 and "tache" not in proto, f"{dossier} : pas le protocole PushT de 500 épisodes")
        pc = proto["config"]["plan_config"]
        A.exiger((pc["receding_horizon"], proto["config"]["solver"]["n_steps"]) == (rh, it),
                 f"{dossier} : planification {pc['receding_horizon']} blocs, {proto['config']['solver']['n_steps']} itérations")
        if c != "r25":
            A.exiger(proto.get("planification") == {"receding_horizon": rh, "cem_n_steps": it}, f"{dossier} : protocole")
        for v in VERSIONS:
            for g in GRAINES:
                f = d / f"{v.replace(':', '-')}_s{g}.npz"
                A.exiger(f.exists(), f"{dossier} : {f.name} manquant")
                x = dict(np.load(f))
                A.exiger((str(x["spec"]), int(x["seed"])) == (v, g), f"{dossier}/{f.name} : mauvaise version ou graine")
                A.exiger(not bool(x["git_sale"]), f"{dossier}/{f.name} : arbre git modifié")
                A.exiger(int(x["n_couches"]) == COUCHES[v], f"{dossier}/{f.name} : couches quantifiées")
                if c != "r25":
                    shas.add(str(x["git"]))
                res[(c, v, g)] = x
    ref = res[("r25", "fp32", 42)]
    for (c, v, g), x in res.items():
        for k in ("episodes_idx", "start_steps", "init_state", "goal_state"):
            A.exiger(np.array_equal(x[k], ref[k]), f"{c} {v} graine {g} : {k} différent d'EXP-003")
    A.exiger(len(shas) == 1, f"EXP-007 : fichiers de plusieurs commits {sorted(shas)}")
    sha = shas.pop()
    if verifier:
        verifier_code(sha)
    return res, sha


def verifier_sanite(racine, fin):
    a, b = dict(np.load(Path(fin) / "all-4_s42.npz")), dict(np.load(Path(racine) / "exp007-r5c6" / "all-4_s42.npz"))
    A.exiger(set(a) == set(b), "sanité de clôture : clés différentes")
    for k in sorted(set(a) - {"git", "git_sale"}):
        A.exiger(np.array_equal(a[k], b[k], equal_nan=a[k].dtype.kind == "f"), f"sanité de clôture : {k} diffère")


def analyser(racine, fin, verifier=True):
    res, sha = charger(racine, verifier)
    verifier_sanite(racine, fin)
    K, lim, div = {}, 0, 0
    for (c, v, g), x in res.items():
        _, _, l, dv = A.controler_course(x, f"{c} {v} graine {g}")
        lim, div = lim + l, div + dv
    for c in CONDITIONS:
        for v in VERSIONS:
            K[(c, v)] = np.stack([res[(c, v, g)]["success"].astype(np.int64) for g in GRAINES]).sum(axis=0)
    inv, boot = A.Inversion(n=N7), A.Bootstrap(n=N7)
    gain = lambda v, c: K[(c, v)] - K[("r25", v)]
    x_tests = {"Q4a": gain("all:4", "r5"),                          # replanifier tous les 5 pas aide à 4 bits
               "Q4b": gain("all:4", "r5") - gain("fp32", "r5"),     # et aide plus à 4 bits qu'en pleine précision
               "Q4c": gain("all:4", "r5c6")}                        # et aide à 4 bits à calcul égal
    tests = {n: {"somme": int(x.sum()), "ecart_points": boot.ic(x / 3), "p": inv.p(x, "plus")} for n, x in x_tests.items()}
    for n, q in zip(tests, A.holm([t["p"] for t in tests.values()])):
        tests[n]["p_holm"] = float(q)
        tests[n]["verdict"] = "soutenue" if q < A.ALPHA else "non soutenue"
    desc = {f"{c} {v}": {"reussite": float(K[(c, v)].mean() / 3),
                         "gain_sur_r25": None if c == "r25" else boot.ic(gain(v, c) / 3)}
            for c in CONDITIONS for v in VERSIONS}
    desc["ecart_4_bits_par_condition"] = {c: boot.ic((K[(c, "all:4")] - K[(c, "fp32")]) / 3) for c in CONDITIONS}
    return {"controles": {"commit": sha, "fichiers": len(res), "cas_limites_arrondi": lim, "divergences": div},
            "tests": tests, "descriptifs": desc}


def rapport(r):
    t = lambda e: A.ic_txt(e, 100, 1)
    L = ["# EXP-007 : analyse pré-enregistrée (replanification et compression, PushT)", "",
         f"Contrôles passés : {r['controles']['fichiers']} fichiers, commit {r['controles']['commit'][:7]}, "
         f"courses divergentes {r['controles']['divergences']}.", "",
         "| Test | Écart (points, 3 graines) | p | p Holm | Verdict |", "| --- | --- | --- | --- | --- |"]
    for n, x in r["tests"].items():
        L.append(f"| {n} | {t(x['ecart_points'])} | {x['p']:.4f} | {x['p_holm']:.4f} | {x['verdict']} |")
    L += ["", "| Condition, version | Réussite | Gain sur r25 (points) |", "| --- | --- | --- |"]
    for k, d in r["descriptifs"].items():
        if k != "ecart_4_bits_par_condition":
            L.append(f"| {k} | {100 * d['reussite']:.1f} % | {t(d['gain_sur_r25']) if d['gain_sur_r25'] else ''} |")
    L += ["", "Écart de 4 bits à fp32 par condition : " + " ; ".join(
        f"{c} {t(e)}" for c, e in r["descriptifs"]["ecart_4_bits_par_condition"].items()) + "."]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--racine", required=True)
    ap.add_argument("--sanite-fin", required=True)
    ap.add_argument("--sortie", required=True)
    a = ap.parse_args()
    r = analyser(a.racine, a.sanite_fin)
    out = Path(a.sortie)
    out.mkdir(parents=True, exist_ok=True)
    (out / "resultats.json").write_text(json.dumps(A.pur(r), indent=1, ensure_ascii=False))
    (out / "rapport.md").write_text(rapport(r))
    print((out / "rapport.md").read_text())


if __name__ == "__main__":
    main()
