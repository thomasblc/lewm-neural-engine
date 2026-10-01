"""Analyse pré-enregistrée d'EXP-011 (PREREG-EXP011.md, amendement 2) : le prédicteur ternaire (stocké sur 2 bits),
RTN et GPTQ à la même échelle réduite, contre la pleine précision, sur PushT et TwoRoom. Contrôles bloquants (dont deux
reproductions octet pour octet), 4 tests appariés (G par inversion de signe ; N par inversion de signe et bootstrap-t
concordants), Holm, borne inférieure au niveau de Holm, contrôle négatif, sensibilité à 10 graines, descriptifs.

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
REPLIQUE = (47, 48, 49, 50, 51)                     # deuxième série de la pleine précision, déjà calculée
RTN, GPTQ = "predictor:2:clip", "predictor:2:clip:gptq"
VERSIONS = (RTN, GPTQ)
COUCHES = 30
MARGE = 2                                           # points, test de non-infériorité
B_BOOT, GRAINE_BOOT = 10_000, 4
TACHES = {  # dossier, référence fp32, RTN à 3 bits du prédicteur (descriptif), calibration, évaluations à exclure
    "pusht": {"dossier": "exp011-pusht", "fp32": "exp003", "rtn3": "exp003", "calib": "exp009-calib-pusht",
              "evals": ("exp003", "exp005", "sanite-n50")},
    "tworoom": {"dossier": "exp011-tworoom", "fp32": "exp006", "rtn3": "exp008", "calib": "exp009-calib-tworoom",
                "evals": ("exp006",)},
}
TESTS = (("G1", "pusht", "gptq_rtn"), ("G2", "tworoom", "gptq_rtn"), ("N1", "pusht", "gptq_fp32"), ("N2", "tworoom", "gptq_fp32"))
FICHIERS_ANALYSE = ("analyse_exp011.py", "erreur_reconstruction.py", "analyse_exp009.py", "analyse_exp006.py", "analyse_exp003.py",
                    "metriques.py", "gptq.py", "quant.py", "run_grid.py", "verifier_gptq.py", "PREREG-EXP011.md")
ICI = Path(__file__).resolve().parent


def git(*args):
    return subprocess.run(["git", "-C", str(ICI), *args], capture_output=True, text=True)


def verifier_code(sha):
    A.exiger(git("cat-file", "-e", f"{sha}^{{commit}}").returncode == 0, f"commit {sha[:7]} inconnu du dépôt")
    A.exiger(git("diff", "--quiet", sha, "--", *FICHIERS_ANALYSE).returncode == 0,
             f"code d'analyse ou de calcul différent de celui du commit {sha[:7]}")


def p_bootstrap_t(y, b=B_BOOT, graine=GRAINE_BOOT):
    """p unilatérale (moyenne de y positive) d'un bootstrap-t : épisodes rééchantillonnés, statistique studentisée."""
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    m, se = y.mean(), y.std(ddof=1) / np.sqrt(n)
    t = m / se if se > 0 else (np.inf if m > 0 else -np.inf)
    yb = y[np.random.default_rng(graine).integers(0, n, size=(b, n))]
    seb = yb.std(axis=1, ddof=1) / np.sqrt(n)
    tb = (yb.mean(axis=1) - m) / np.where(seb > 0, seb, np.inf)
    return float((1 + int((tb >= t).sum())) / (b + 1))


def charger(racine, verifier=True, calib_torch=True):
    r = Path(racine)
    K, shas = {}, set()
    for t, c in TACHES.items():
        controler = A6.controler if t == "tworoom" else (lambda d, nom: A.controler_course(d, nom))
        ref = dict(np.load(r / c["fp32"] / "fp32_s42.npz"))
        for d, g in ((f"exp011-repro-{t}", 42), (f"exp011-fin-{t}", 46)):
            A9.identiques(r / d / f"fp32_s{g}.npz", r / c["fp32"] / f"fp32_s{g}.npz", f"reproduction {t} graine {g}")
        proto = json.loads((r / c["dossier"] / "meta.json").read_text())["protocole"]
        fcal = r / c["calib"] / "calib.pt"
        sha_cal = hashlib.sha256(fcal.read_bytes()).hexdigest()
        A.exiger(proto.get("gptq_calib", {}).get("sha256") == sha_cal, f"{t} : calibration différente de celle du protocole")
        p9 = json.loads((r / f"exp009-{t}" / "meta.json").read_text())["protocole"]
        A.exiger(p9.get("gptq_calib", {}).get("sha256") == sha_cal, f"{t} : calibration différente de celle d'EXP-009")
        reste = {k: v for k, v in proto.items() if k != "gptq_calib"}
        A.exiger(reste == json.loads((r / c["fp32"] / "meta.json").read_text())["protocole"],
                 f"{t} : protocole différent de celui de la référence {c['fp32']}")
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
        K[(t, "replique")] = sum(lire(c["fp32"], "fp32", g)["success"].astype(np.int64) for g in REPLIQUE)
        K[(t, "rtn3")] = sum(lire(c["rtn3"], "predictor:3", g)["success"].astype(np.int64) for g in GRAINES)
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
    for t in TACHES:                                 # les reproductions viennent du même commit, arbre propre
        for d, g in ((f"exp011-repro-{t}", 42), (f"exp011-fin-{t}", 46)):
            x = np.load(r / d / f"fp32_s{g}.npz")
            A.exiger(str(x["git"]) == sha and not bool(x["git_sale"]), f"reproduction {t} graine {g} : autre commit ou arbre modifié")
    if verifier:
        verifier_code(sha)
    return K, sha


def analyser(racine, verifier=True, calib_torch=True):
    K, sha = charger(racine, verifier, calib_torch)
    inv, boot = A.Inversion(n=500), A.Bootstrap(n=500)

    def test_ni(x):
        y = 20 * x + MARGE                            # moyenne de y positive ssi écart de réussite > -2 points
        ps, pb = inv.p(y, "plus"), p_bootstrap_t(y)
        return {"somme": float(x.sum()), "ecart_points": boot.ic(x / 5), "p_signe": ps, "p_bootstrap_t": pb,
                "erreur_mc_p_signe": float(np.sqrt(ps * (1 - ps) / A.B)), "p": max(ps, pb)}
    tests, differences = {}, {}
    for nom, t, quoi in TESTS:
        g = K[(t, GPTQ)]
        if quoi == "gptq_rtn":
            x = g - K[(t, RTN)]
            p = inv.p(x, "plus")
            tests[nom] = {"tache": t, "comparaison": quoi, "somme": int(x.sum()), "ecart_points": boot.ic(x / 5), "p_signe": p,
                          "erreur_mc_p_signe": float(np.sqrt(p * (1 - p) / A.B)), "p": p}
        else:
            x = g - K[(t, "fp32")]
            tests[nom] = {"tache": t, "comparaison": quoi, **test_ni(x)}
        differences[nom] = x
    for (nom, *_), q in zip(TESTS, A.holm([tests[n]["p"] for n, *_ in TESTS])):
        tests[nom]["p_holm"] = float(q)
        tests[nom]["verdict"] = "soutenue" if q < A.ALPHA else "non soutenue"
    for rang, i in enumerate(np.argsort([tests[n]["p"] for n, *_ in TESTS])):   # borne inférieure au niveau de Holm
        nom = TESTS[i][0]
        niveau = A.ALPHA / (len(TESTS) - rang)
        x = differences[nom] / 5
        tests[nom]["niveau_holm"] = float(niveau)
        tests[nom]["borne_inferieure_points"] = float(100 * np.percentile(np.array([x[j].mean() for j in boot.idx]), 100 * niveau))
    controle = {t: test_ni(K[(t, "replique")] - K[(t, "fp32")]) for t in TACHES}
    sensibilite = {n: test_ni(K[(t, GPTQ)] - (K[(t, "fp32")] + K[(t, "replique")]) / 2) for n, t, q in TESTS if q == "gptq_fp32"}
    desc = {}
    for t in TACHES:
        g, r_, f = K[(t, GPTQ)], K[(t, RTN)], K[(t, "fp32")]
        den = np.array([f[i].sum() - r_[i].sum() for i in boot.idx], dtype=np.float64)
        num = np.array([g[i].sum() - r_[i].sum() for i in boot.idx], dtype=np.float64)
        parts = num[den > 0] / den[den > 0]
        sens = f.sum() - r_.sum() >= 0.02 * 5 * len(f)  # part rattrapée sans objet si RTN perd moins de 2 points
        desc[t] = {"reussite_fp32": float(f.mean() / 5), "reussite_replique": float(K[(t, "replique")].mean() / 5),
                   "reussite_rtn": float(r_.mean() / 5), "reussite_gptq": float(g.mean() / 5),
                   "reussite_rtn3": float(K[(t, "rtn3")].mean() / 5),
                   "gptq_moins_fp32": boot.ic((g - f) / 5), "rtn_moins_fp32": boot.ic((r_ - f) / 5),
                   "gptq_moins_rtn": boot.ic((g - r_) / 5), "gptq_moins_rtn3": boot.ic((g - K[(t, "rtn3")]) / 5),
                   "part_rattrapee": None if not sens else
                   {"moyenne": float((g.sum() - r_.sum()) / (f.sum() - r_.sum())),
                    "bas": float(np.percentile(parts, 2.5)), "haut": float(np.percentile(parts, 97.5))}}
    return {"controles": {"commit": sha}, "tests": tests, "controle_negatif": controle, "sensibilite_10_graines": sensibilite,
            "descriptifs": desc}


def rapport(r):
    t = lambda e, f=100: A.ic_txt(e, f, 1)
    L = ["# EXP-011 : analyse pré-enregistrée (le prédicteur ternaire, RTN et GPTQ à échelle réduite)", "",
         f"Contrôles passés, commit {r['controles']['commit'][:7]} ; reproductions des graines 42 et 46 identiques.", "",
         "| Tâche | fp32 | RTN réduit | GPTQ réduit | GPTQ moins fp32 | RTN moins fp32 | GPTQ moins RTN | RTN 3 bits | GPTQ ternaire moins RTN 3 bits |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for k, d in r["descriptifs"].items():
        L.append(f"| {k} | {100 * d['reussite_fp32']:.1f} % | {100 * d['reussite_rtn']:.1f} % | {100 * d['reussite_gptq']:.1f} % | "
                 f"{t(d['gptq_moins_fp32'])} | {t(d['rtn_moins_fp32'])} | {t(d['gptq_moins_rtn'])} | {100 * d['reussite_rtn3']:.1f} % | "
                 f"{t(d['gptq_moins_rtn3'])} |")
    noms = {"gptq_rtn": "GPTQ plus haut que RTN", "gptq_fp32": f"GPTQ perd moins de {MARGE} points"}
    L += ["", "| Test | Tâche | Hypothèse | Écart (points) | Borne inférieure au niveau de Holm | p signe | p bootstrap-t | p Holm | Verdict |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for n, x in r["tests"].items():
        pb = f"{x['p_bootstrap_t']:.4f}" if "p_bootstrap_t" in x else "sans objet"
        L.append(f"| {n} | {x['tache']} | {noms[x['comparaison']]} | {t(x['ecart_points'])} | {x['borne_inferieure_points']:.1f} "
                 f"(niveau {x['niveau_holm']:.4f}) | {x['p_signe']:.4f} | {pb} | {x['p_holm']:.4f} | {x['verdict']} |")
    L += ["", "Contrôle négatif (deuxième série de la pleine précision, graines 47 à 51, contre 42 à 46, test N) :"]
    for k, c in r["controle_negatif"].items():
        L.append(f"- {k} : écart {t(c['ecart_points'])} points, p signe {c['p_signe']:.4f}, p bootstrap-t {c['p_bootstrap_t']:.4f}")
    L += ["", "Sensibilité, référence sur les 10 graines de la pleine précision :"]
    for n, c in r["sensibilite_10_graines"].items():
        L.append(f"- {n} : écart {t(c['ecart_points'])} points, p {c['p']:.4f}")
    L += ["", "Part de la perte de RTN rattrapée par GPTQ (en second : elle dépend de la faiblesse de RTN) :"]
    for k, d in r["descriptifs"].items():
        L.append(f"- {k} : {t(d['part_rattrapee']) + ' %' if d['part_rattrapee'] else 'sans objet'}")
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
