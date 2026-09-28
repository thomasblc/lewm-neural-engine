"""EXP-004 : analyse pré-enregistrée (PREREG-EXP004.md), figée avec lui.

Reprend les références d'EXP-003 (fp32, all:4, all:3, encoder:4, predictor:4, graines 42 à 46) après
leurs propres contrôles bloquants, contrôle les 25 fichiers d'EXP-004 et la sanité (all:4 graine 42
recalculé, identique à EXP-003), puis calcule Q3a, Q3b, Q5a, Q5b (inversion de signe, Holm) et les
descriptifs. Les outils statistiques et les contrôles par course sont ceux d'analyse_exp003.py.

Usage :
    python analyse_exp004.py --grille3 ../resultats/exp003 --grille4 ../resultats/exp004 \\
        --sanite ../resultats/exp004-sanite --sortie ../resultats/analyse-exp004
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

import analyse_exp003 as A

NOUVELLES = ("autres:4", "encpred:4", "all:4:g64", "autres:3", "all:3:g64")
REFERENCES = ("fp32", "all:4", "all:3", "encoder:4", "predictor:4")
GRAINES = A.GA
QUANT_SHA256 = "2c450c5b06ebd6753e77ff9c189fffc339b655070efaf32e103f13ad5e29c65d"
COUCHES = {"autres": 6, "encpred": 103, "all": 109}
COMPARAISONS = (                    # nom, version, référence, signe attendu de la somme
    ("Q3a", "autres:4", "fp32", -1),
    ("Q3b", "encpred:4", "all:4", +1),
    ("Q5a", "all:4:g64", "all:4", +1),
    ("Q5b", "all:3:g64", "all:3", +1),
)
CLES_SANITE = ("state", "block_pose", "agent_pos", "contacts", "success", "first_success", "n_steps")


def git(*args, texte=True):
    return subprocess.run(["git", "-C", str(A.ICI), *args], capture_output=True, text=texte)


def verifier_code4(sha):
    """Le commit existe, garde run_grid.py de c5703db et porte le quant.py d'EXP-004."""
    A.exiger(git("cat-file", "-e", f"{sha}^{{commit}}").returncode == 0, f"commit {sha[:7]} inconnu du dépôt")
    A.exiger(git("diff", "--quiet", A.COMMIT_CODE, sha, "--", "run_grid.py").returncode == 0,
             f"le commit {sha[:7]} n'a pas le run_grid.py de {A.COMMIT_CODE}")
    src = git("show", f"{sha}:quant.py", texte=False)
    A.exiger(src.returncode == 0 and hashlib.sha256(src.stdout).hexdigest() == QUANT_SHA256,
             f"le commit {sha[:7]} n'a pas le quant.py d'EXP-004")


def charger4(dossier, ref):
    """Les 25 fichiers d'EXP-004, contrôlés contre le protocole d'EXP-003 (ref : fichier fp32 graine 42)."""
    dossier = Path(dossier)
    proto = json.loads((dossier / "meta.json").read_text())["protocole"]
    A.exiger(proto["n"] == A.N, f"meta.json d'EXP-004 : {proto['n']} épisodes")
    res = {}
    for f in sorted(dossier.glob("*.npz")):
        if ".tmp." in f.name:
            continue
        d = dict(np.load(f))
        cle = (str(d["spec"]), int(d["seed"]))
        A.exiger(cle not in res, f"{f.name} : doublon de {cle}")
        res[cle] = d
    attendus = {(v, g) for v in NOUVELLES for g in GRAINES}
    A.exiger(set(res) == attendus, f"EXP-004 : fichiers manquants {sorted(attendus - set(res))}, "
                                   f"en trop {sorted(set(res) - attendus)}")
    shas = set()
    for (v, g), d in res.items():
        nom = f"{v} graine {g}"
        A.exiger(not bool(d["git_sale"]), f"{nom} : arbre git modifié au lancement")
        shas.add(str(d["git"]))
        A.exiger(np.array_equal(d["positions"], np.arange(A.N)), f"{nom} : positions")
        for k in ("episodes_idx", "start_steps", "init_state", "goal_state"):
            A.exiger(np.array_equal(d[k], ref[k]), f"{nom} : {k} différent d'EXP-003")
        A.exiger(int(d["chunk"]) == int(ref["chunk"]) == proto["chunk"], f"{nom} : taille des paquets")
        A.exiger(int(d["n_couches"]) == COUCHES[v.split(":")[0]], f"{nom} : {int(d['n_couches'])} couches quantifiées")
        A.exiger(d["state"].shape == (A.T, A.N, 7), f"{nom} : forme de l'état")
        ch = int(d["chunk"])
        for c, pas in enumerate(np.asarray(d["n_steps"]).tolist()):
            sl = slice(c * ch, (c + 1) * ch)
            A.exiger(pas == A.T or (d["success"][sl].all() and int(d["first_success"][sl].max()) == pas - 1),
                     f"{nom} : paquet {c} arrêté à {pas} pas")
    for sha in sorted(shas):
        verifier_code4(sha)
    return res, sorted(shas)


def verifier_sanite(dossier_sanite, f_ref):
    """all:4 graine 42 recalculé avec le code d'EXP-004 : identique au fichier d'EXP-003."""
    a, b = dict(np.load(Path(dossier_sanite) / "all-4_s42.npz")), dict(np.load(f_ref))
    A.exiger(not bool(a["git_sale"]), "sanité : arbre git modifié au lancement")
    verifier_code4(str(a["git"]))
    for k in CLES_SANITE:
        A.exiger(np.array_equal(a[k], b[k], equal_nan=a[k].dtype.kind == "f"), f"sanité : {k} diffère d'EXP-003")
    return str(a["git"])


def table(res, v):
    """Même construction que analyse_exp003.tables, pour une version et les graines 42 à 46."""
    S, M2, M3, P, divergences, limites = [], [], [], [], 0, 0
    for g in GRAINES:
        d = res[(v, g)]
        m, mT, lim, div = A.controler_course(d, f"{v} graine {g}")
        limites += lim
        divergences += div
        S.append(d["success"].astype(np.int64))
        M2.append(m)
        M3.append(mT)
        P.append(d["first_success"].astype(np.int64))
    S, M3 = np.stack(S), np.stack(M3)
    return {"S": S, "K": S.sum(axis=0), "M2": np.stack(M2), "M3": M3,
            "classe": np.where(S == 1, 0, np.where(M3 < 1, 1, 2)), "premier": np.stack(P),
            "divergences": divergences, "limites": limites}


def analyser(grille3, grille4, sanite, effectifs=A.EFFECTIFS, verifier_exp003=True):
    res3, shas3 = A.charger(grille3) if verifier_exp003 else (None, [])
    if res3 is not None:
        D3, _, _ = A.tables(res3)
        ref = res3[("fp32", GRAINES[0])]
    else:                                                      # grilles simulées des tests
        res3 = {}
        for f in Path(grille3).glob("*.npz"):
            d = dict(np.load(f))
            res3[(str(d["spec"]), int(d["seed"]))] = d
        ref = res3[("fp32", GRAINES[0])]
        D3 = None
    sha_sanite = verifier_sanite(sanite, Path(grille3) / "all-4_s42.npz")
    res4, shas4 = charger4(grille4, ref)
    dep = A.depart(res3)
    if effectifs is not None:
        A.exiger((int(dep["bouge"].sum()), int(dep["au_but"].sum())) == effectifs, "strates de départ inattendues")
    D = {v: (D3[(v, "A")] if D3 is not None else table(res3, v)) for v in REFERENCES}
    D.update({v: table(res4, v) for v in NOUVELLES})

    inv, boot = A.Inversion(), A.Bootstrap()
    boot_strate = A.Bootstrap(n=int(dep["bouge"].sum()), graine=3)
    bouge = dep["bouge"]
    p = []
    for nom, v, r, _ in COMPARAISONS:
        p.append(inv.p(D[v]["K"] - D[r]["K"], "bilateral"))
    adj = A.holm(p)
    tests = {}
    for (nom, v, r, sens), pv, pa in zip(COMPARAISONS, p, adj):
        x = D[v]["K"] - D[r]["K"]
        tests[nom] = {"version": v, "reference": r, "somme": int(x.sum()), "delta_reussite": float(x.mean() / 5),
                      "p": pv, "p_holm": float(pa), "verdict": A.verdict(pa, x.sum(), sens)}

    f = D["fp32"]
    desc = {}
    for v in REFERENCES + NOUVELLES:
        d = D[v]
        e = {"reussite": float(d["S"].mean()), "classes": A.parts(d["classe"]),
             "classes_bouge": A.parts(d["classe"][:, bouge])}
        if v != "fp32":
            d3 = A.lg(d["M3"]).mean(axis=0) - A.lg(f["M3"]).mean(axis=0)
            e["delta_reussite"] = boot.ic((d["K"] - f["K"]) / 5)
            e["delta_log_M3_bouge"] = boot_strate.ic(d3[bouge])
        desc[v] = e
    K = {v: D[v]["K"].astype(np.float64) for v in D}
    additivite = {
        "I : tout le modèle moins la somme des trois parties (4 bits)":
            boot.ic((K["all:4"] - K["encoder:4"] - K["predictor:4"] - K["autres:4"] + 2 * K["fp32"]) / 5),
        "J : encodeur et prédicteur ensemble moins la somme des deux (4 bits)":
            boot.ic((K["encpred:4"] - K["encoder:4"] - K["predictor:4"] + K["fp32"]) / 5),
        "autres:3 contre fp32": boot.ic((K["autres:3"] - K["fp32"]) / 5),
        "autres:3 contre all:3": boot.ic((K["autres:3"] - K["all:3"]) / 5),
    }
    return {"controles": {"commits_exp003": shas3, "commits_exp004": shas4, "commit_sanite": sha_sanite,
                          "divergences": {v: D[v]["divergences"] for v in NOUVELLES},
                          "cas_limites_arrondi": {v: D[v]["limites"] for v in NOUVELLES}},
            "tests": tests, "descriptifs": desc, "additivite": additivite}


def rapport(r):
    L = ["# EXP-004 : analyse pré-enregistrée", "",
         f"Contrôles passés : EXP-003 ({', '.join(s[:7] for s in r['controles']['commits_exp003']) or 'simulé'}), "
         f"EXP-004 ({', '.join(s[:7] for s in r['controles']['commits_exp004'])}), sanité identique à EXP-003 "
         f"({r['controles']['commit_sanite'][:7]}). Courses divergentes : "
         + (", ".join(f"{v} : {n}" for v, n in r["controles"]["divergences"].items() if n) or "aucune") + ".", "",
         "## Tests (Holm sur 4)", "", "| Test | Comparaison | Écart (points) | p | p Holm | Verdict |",
         "| --- | --- | --- | --- | --- | --- |"]
    for nom, t in r["tests"].items():
        L.append(f"| {nom} | {t['version']} contre {t['reference']} | {100 * t['delta_reussite']:+.2f} | {t['p']:.4f} | "
                 f"{t['p_holm']:.4f} | {t['verdict']} |")
    L += ["", "## Par version (5 graines, 500 épisodes ; écarts à fp32, intervalle bootstrap à 95 %)", "",
          "| Version | Réussite | Écart (points) | Écart log M3, strate où le T doit bouger |", "| --- | --- | --- | --- |"]
    for v, e in r["descriptifs"].items():
        if v == "fp32":
            L.append(f"| {v} | {100 * e['reussite']:.1f} % | | |")
        else:
            L.append(f"| {v} | {100 * e['reussite']:.1f} % | {A.ic_txt(e['delta_reussite'])} | "
                     f"{A.ic_txt(e['delta_log_M3_bouge'], 1, 3)} |")
    L += ["", "## Additivité et 3 bits (points, intervalle bootstrap à 95 %)", ""]
    for nom, e in r["additivite"].items():
        L.append(f"- {nom} : {A.ic_txt(e)}")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grille3", required=True)
    ap.add_argument("--grille4", required=True)
    ap.add_argument("--sanite", required=True)
    ap.add_argument("--sortie", required=True)
    a = ap.parse_args()
    r = analyser(a.grille3, a.grille4, a.sanite)
    out = Path(a.sortie)
    out.mkdir(parents=True, exist_ok=True)
    (out / "resultats.json").write_text(json.dumps(A.pur(r), indent=1, ensure_ascii=False))
    (out / "rapport.md").write_text(rapport(r))
    print((out / "rapport.md").read_text())


if __name__ == "__main__":
    main()
