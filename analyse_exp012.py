"""Analyse pré-enregistrée d'EXP-012 (PREREG-EXP012.md) : le prédicteur en Core ML 16 bits, sur le GPU et sur le
Neural Engine, contre la pleine précision PyTorch, sur PushT et TwoRoom. Contrôles bloquants, 4 tests de
non-infériorité appariés par inversion de signe (Holm), descriptifs dont le temps par plan et la puissance.

Usage :
    python analyse_exp012.py --racine ../resultats --sortie ../resultats/analyse-exp012
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

import analyse_exp003 as A
import analyse_exp006 as A6
import analyse_exp009 as A9

GRAINES = (42, 43, 44, 45, 46)
MOTEURS = {"ne": "coreml-ne", "gpu": "coreml-gpu"}
MARGE = 2                                           # points
TACHES = {"pusht": "exp003", "tworoom": "exp006"}  # référence fp32 PyTorch
TESTS = (("N1", "pusht", "ne"), ("N2", "tworoom", "ne"), ("N3", "pusht", "gpu"), ("N4", "tworoom", "gpu"))
FICHIERS_ANALYSE = ("analyse_exp012.py", "analyse_exp009.py", "analyse_exp006.py", "analyse_exp003.py", "metriques.py",
                    "quant.py", "run_grid.py", "coreml_pred.py", "predicteur_ane.py", "PREREG-EXP012.md")
ICI = Path(__file__).resolve().parent


def git(*args):
    return subprocess.run(["git", "-C", str(ICI), *args], capture_output=True, text=True)


def verifier_code(sha):
    A.exiger(git("cat-file", "-e", f"{sha}^{{commit}}").returncode == 0, f"commit {sha[:7]} inconnu du dépôt")
    A.exiger(git("diff", "--quiet", sha, "--", *FICHIERS_ANALYSE).returncode == 0,
             f"code d'analyse ou de calcul différent de celui du commit {sha[:7]}")


def dossier(t, m):
    return f"exp012-{m}-{t}"


def charger(racine, verifier=True):
    r = Path(racine)
    K, temps, shas = {}, {}, set()
    for t, ref_d in TACHES.items():
        controler = A6.controler if t == "tworoom" else (lambda d, nom: A.controler_course(d, nom))
        ref = dict(np.load(r / ref_d / "fp32_s42.npz"))
        proto_ref = json.loads((r / ref_d / "meta.json").read_text())["protocole"]
        A9.identiques(r / f"exp012-repro-{t}" / "fp32_s42.npz", r / ref_d / "fp32_s42.npz", f"reproduction {t}")

        def lire(d, g, moteur=None):
            f = r / d / f"fp32_s{g}.npz"
            A.exiger(f.exists(), f"{d}/{f.name} manquant")
            x = dict(np.load(f))
            A.exiger((str(x["spec"]), int(x["seed"])) == ("fp32", g), f"{d}/{f.name} : version ou graine")
            A.exiger(not bool(x["git_sale"]), f"{d}/{f.name} : arbre git modifié")
            A.exiger(int(x["n_couches"]) == 0, f"{d}/{f.name} : {int(x['n_couches'])} couches quantifiées")
            if moteur is not None:
                A.exiger(str(x.get("moteur", "")) == moteur, f"{d}/{f.name} : moteur {x.get('moteur')} au lieu de {moteur}")
            for k in ("episodes_idx", "start_steps", "init_state", "goal_state"):
                A.exiger(np.array_equal(x[k], ref[k]), f"{d}/{f.name} : {k} différent de la référence")
            controler(x, f"{d}/{f.name}")
            return x
        K[(t, "fp32")] = sum(lire(ref_d, g)["success"].astype(np.int64) for g in GRAINES)
        for m, moteur in MOTEURS.items():
            d = dossier(t, m)
            proto = json.loads((r / d / "meta.json").read_text())["protocole"]
            A.exiger({k: v for k, v in proto.items() if k != "moteur"} == proto_ref,
                     f"{d} : protocole différent de celui de la référence {ref_d}")
            mo = proto.get("moteur", {})
            A.exiger(mo.get("nom") == moteur and mo.get("precision") == "float16", f"{d} : moteur {mo}")
            S, empreintes, tp, pw = 0, set(), [], []
            for g in GRAINES:
                x = lire(d, g, moteur)
                empreintes.add(str(x["coreml_sha256"]))
                shas.add(str(x["git"]))
                S = S + x["success"].astype(np.int64)
                j = json.loads((r / d / f"fp32_s{g}.temps.json").read_text())
                A.exiger(j["moteur"] == moteur and j["git"] == str(x["git"]), f"{d} graine {g} : fichier de temps")
                tp += j["temps_par_plan_s"]
                pw += j["puissance_w"]
            A.exiger(len(empreintes) == 1, f"{d} : poids Core ML différents selon les graines")
            K[(t, m)] = S
            temps[(t, m)] = (np.array(tp), np.array(pw))
        j = json.loads((r / f"exp012-repro-{t}" / "fp32_s42.temps.json").read_text())
        A.exiger(j["moteur"] == "torch", f"reproduction {t} : moteur {j['moteur']}")
        temps[(t, "torch")] = (np.array(j["temps_par_plan_s"]), np.array(j["puissance_w"]))
    A.exiger(len(shas) == 1, f"EXP-012 : fichiers de plusieurs commits {sorted(shas)}")
    sha = shas.pop()
    for t in TACHES:
        x = np.load(r / f"exp012-repro-{t}" / "fp32_s42.npz")
        A.exiger(str(x["git"]) == sha and not bool(x["git_sale"]), f"reproduction {t} : autre commit ou arbre modifié")
    if verifier:
        verifier_code(sha)
    return K, temps, sha


def resume_temps(tp, pw):
    return {"plans": int(len(tp)), "mediane_ms": float(1000 * np.median(tp)), "q1_ms": float(1000 * np.percentile(tp, 25)),
            "q3_ms": float(1000 * np.percentile(tp, 75)), "puissance_mediane_w": float(np.median(pw)) if len(pw) else None,
            "energie_par_plan_j": float(np.median(pw) * np.median(tp)) if len(pw) else None}


def analyser(racine, verifier=True):
    K, temps, sha = charger(racine, verifier)
    inv, boot = A.Inversion(n=500), A.Bootstrap(n=500)
    tests = {}
    for nom, t, m in TESTS:
        x = K[(t, m)] - K[(t, "fp32")]
        tests[nom] = {"tache": t, "moteur": MOTEURS[m], "somme": int(x.sum()), "ecart_points": boot.ic(x / 5),
                      "p": inv.p(20 * x + MARGE, "plus")}   # moyenne de y positive ssi écart de réussite > -2 points
    for (nom, *_), q in zip(TESTS, A.holm([x["p"] for x in tests.values()])):
        tests[nom]["p_holm"] = float(q)
        tests[nom]["verdict"] = "soutenue" if q < A.ALPHA else "non soutenue"
    desc = {}
    for t in TACHES:
        desc[t] = {"reussite_fp32": float(K[(t, "fp32")].mean() / 5),
                   **{f"reussite_{m}": float(K[(t, m)].mean() / 5) for m in MOTEURS},
                   "ne_moins_gpu": boot.ic((K[(t, "ne")] - K[(t, "gpu")]) / 5),
                   "temps": {m: resume_temps(*temps[(t, m)]) for m in ("torch", *MOTEURS)}}
    return {"controles": {"commit": sha}, "tests": tests, "descriptifs": desc}


def rapport(r):
    t = lambda e, f=100: A.ic_txt(e, f, 1)
    L = ["# EXP-012 : analyse pré-enregistrée (le prédicteur en Core ML 16 bits, GPU et Neural Engine)", "",
         f"Contrôles passés, commit {r['controles']['commit'][:7]}.", "",
         f"| Test | Tâche | Moteur | Écart à la pleine précision (points) | p | p Holm | Perte de moins de {MARGE} points |",
         "| --- | --- | --- | --- | --- | --- | --- |"]
    for n, x in r["tests"].items():
        L.append(f"| {n} | {x['tache']} | {x['moteur']} | {t(x['ecart_points'])} | {x['p']:.4f} | {x['p_holm']:.4f} | {x['verdict']} |")
    L += ["", "| Tâche | Moteur | Réussite | Temps par plan (ms, médiane et quartiles) | Puissance (W) | Énergie par plan (J) |",
          "| --- | --- | --- | --- | --- | --- |"]
    for k, d in r["descriptifs"].items():
        for m, nom in (("torch", "PyTorch float32 (graine 42)"), ("gpu", "Core ML GPU"), ("ne", "Core ML Neural Engine")):
            s = d["temps"][m]
            reu = d["reussite_fp32"] if m == "torch" else d[f"reussite_{m}"]
            pw = f"{s['puissance_mediane_w']:.1f}" if s["puissance_mediane_w"] is not None else "?"
            en = f"{s['energie_par_plan_j']:.1f}" if s["energie_par_plan_j"] is not None else "?"
            L.append(f"| {k} | {nom} | {100 * reu:.1f} % | {s['mediane_ms']:.0f} ({s['q1_ms']:.0f} à {s['q3_ms']:.0f}) | {pw} | {en} |")
        L.append(f"| {k} | Neural Engine moins GPU | {t(d['ne_moins_gpu'])} points | | | |")
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
