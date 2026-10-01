"""Analyse pré-enregistrée d'EXP-014 (PREREG-EXP014.md) : le prédicteur en Core ML sur le Neural Engine avec ses poids
codés (int8 par canal, int4 par canal, palettes de 16 valeurs), contre la pleine précision PyTorch, sur PushT et
TwoRoom. Reprend la méthode d'EXP-012 : contrôles bloquants (reproductions PyTorch octet pour octet), 6 tests de
non-infériorité à 2 points (inversion de signe et bootstrap-t concordants, Holm), contrôle de la marge (deuxième série
de la pleine précision), sensibilité à 10 graines de référence ; descriptifs : effet du codage seul contre le 16 bits
sur le Neural Engine d'EXP-012 (mêmes épisodes, mêmes graines), temps par plan.

Usage :
    python analyse_exp014.py --racine ../resultats --sortie ../resultats/analyse-exp014
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

import analyse_exp003 as A
import analyse_exp006 as A6
import analyse_exp009 as A9
from analyse_exp012 import borne_bootstrap_t, niveaux_holm, p_bootstrap_t

GRAINES = (42, 43, 44, 45, 46)
REPLIQUE = (47, 48, 49, 50, 51)                     # deuxième série de la pleine précision, déjà calculée
CODAGES = ("int8", "int4c", "pal4")
MOTEUR = "coreml-ne"
MARGE = 2                                           # points
TACHES = {"pusht": "exp003", "tworoom": "exp006"}  # référence fp32 PyTorch
TESTS = tuple((f"N{i + 1}", t, c) for i, (t, c) in enumerate((t, c) for t in TACHES for c in CODAGES))
FICHIERS_ANALYSE = ("analyse_exp014.py", "analyse_exp012.py", "analyse_exp009.py", "analyse_exp006.py", "analyse_exp003.py",
                    "metriques.py", "quant.py", "run_grid.py", "coreml_pred.py", "predicteur_ane.py", "PREREG-EXP014.md")
ICI = Path(__file__).resolve().parent


def git(*args):
    return subprocess.run(["git", "-C", str(ICI), *args], capture_output=True, text=True)


def verifier_code(sha):
    A.exiger(git("cat-file", "-e", f"{sha}^{{commit}}").returncode == 0, f"commit {sha[:7]} inconnu du dépôt")
    A.exiger(git("diff", "--quiet", sha, "--", *FICHIERS_ANALYSE).returncode == 0,
             f"code d'analyse ou de calcul différent de celui du commit {sha[:7]}")


def dossier(t, c):
    return f"exp014-{c}-{t}"


def lire_temps(f, sha):
    j = json.loads(f.read_text())
    A.exiger(j["moteur"] == MOTEUR and j["git"] == sha, f"{f.parent.name}/{f.name} : fichier de temps")
    d = np.array([x[0] for x in j["appels_solve"]], dtype=np.float64)
    n = np.array([x[1] for x in j["appels_solve"]], dtype=np.float64)
    pp = d / np.maximum(n, 1)
    garde = pp <= 3 * np.median(pp)                   # un appel allongé par une pause de batterie est écarté (descriptif)
    return {"par_plan_s": float(d[garde].sum() / n[garde].sum()), "appels_ecartes": int((~garde).sum()), "coreml": j["coreml"]}


def charger(racine, verifier=True):
    r = Path(racine)
    K, temps, shas = {}, {}, set()
    for t, ref_d in TACHES.items():
        controler = A6.controler if t == "tworoom" else (lambda d, nom: A.controler_course(d, nom))
        ref = dict(np.load(r / ref_d / "fp32_s42.npz"))
        proto_ref = json.loads((r / ref_d / "meta.json").read_text())["protocole"]
        for d, g in ((f"exp014-repro-{t}", 42), (f"exp014-fin-{t}", 46)):
            A9.identiques(r / d / f"fp32_s{g}.npz", r / ref_d / f"fp32_s{g}.npz", f"reproduction {t} graine {g}")

        def lire(d, g, codage=None):
            f = r / d / f"fp32_s{g}.npz"
            A.exiger(f.exists(), f"{d}/{f.name} manquant")
            x = dict(np.load(f))
            A.exiger((str(x["spec"]), int(x["seed"])) == ("fp32", g), f"{d}/{f.name} : version ou graine")
            A.exiger(not bool(x["git_sale"]), f"{d}/{f.name} : arbre git modifié")
            A.exiger(int(x["n_couches"]) == 0, f"{d}/{f.name} : {int(x['n_couches'])} couches quantifiées en PyTorch")
            if codage is not None:
                A.exiger(str(x.get("moteur", "")) == MOTEUR, f"{d}/{f.name} : moteur {x.get('moteur')} au lieu de {MOTEUR}")
            for k in ("episodes_idx", "start_steps", "init_state", "goal_state"):
                A.exiger(np.array_equal(x[k], ref[k]), f"{d}/{f.name} : {k} différent de la référence")
            controler(x, f"{d}/{f.name}")
            return x
        K[(t, "fp32")] = sum(lire(ref_d, g)["success"].astype(np.int64) for g in GRAINES)
        K[(t, "replique")] = sum(lire(ref_d, g)["success"].astype(np.int64) for g in REPLIQUE)
        K[(t, "ne16")] = sum(lire(f"exp012-ne-{t}", g)["success"].astype(np.int64) for g in GRAINES)   # EXP-012, 16 bits
        for c in CODAGES:
            d = dossier(t, c)
            proto = json.loads((r / d / "meta.json").read_text())["protocole"]
            A.exiger({k: v for k, v in proto.items() if k != "moteur"} == proto_ref,
                     f"{d} : protocole différent de celui de la référence {ref_d}")
            mo = proto.get("moteur", {})
            A.exiger(mo.get("nom") == MOTEUR and mo.get("precision") == "float16" and mo.get("codage") == c, f"{d} : moteur {mo}")
            S, empreintes, fichiers = 0, set(), []
            for g in GRAINES:
                x = lire(d, g, c)
                shas.add(str(x["git"]))
                S = S + x["success"].astype(np.int64)
                tf = lire_temps(r / d / f"fp32_s{g}.temps.json", str(x["git"]))
                A.exiger(set(tf["coreml"].get("placement", {})) == {"NeuralEngine"},
                         f"{d} graine {g} : opérations hors du Neural Engine {tf['coreml'].get('placement')}")
                A.exiger(tf["coreml"].get("codage") == c, f"{d} graine {g} : codage {tf['coreml'].get('codage')} au lieu de {c}")
                A.exiger(tf["coreml"].get("sha256_poids") == str(x["coreml_sha256"]), f"{d} graine {g} : empreintes")
                empreintes.add((str(x["coreml_sha256"]), tf["coreml"].get("sha256_modele")))
                fichiers.append(tf)
            A.exiger(len(empreintes) == 1, f"{d} : modèle Core ML différent selon les graines")
            K[(t, c)] = S
            temps[(t, c)] = fichiers
    A.exiger(len(shas) == 1, f"EXP-014 : fichiers de plusieurs commits {sorted(shas)}")
    sha = shas.pop()
    for t in TACHES:
        for d, g in ((f"exp014-repro-{t}", 42), (f"exp014-fin-{t}", 46)):
            x = np.load(r / d / f"fp32_s{g}.npz")
            A.exiger(str(x["git"]) == sha and not bool(x["git_sale"]), f"reproduction {t} graine {g} : autre commit ou arbre modifié")
    if verifier:
        verifier_code(sha)
    return K, temps, sha


def analyser(racine, verifier=True):
    K, temps, sha = charger(racine, verifier)
    inv, boot = A.Inversion(n=500), A.Bootstrap(n=500)

    def test_ni(x):
        y = 20 * x + MARGE                            # moyenne de y positive ssi écart de réussite > -2 points
        ps, pb = inv.p(y, "plus"), p_bootstrap_t(y)
        return {"somme": float(x.sum()), "ecart_points": boot.ic(x / 5), "p_signe": ps, "p_bootstrap_t": pb,
                "erreur_mc_p_signe": float(np.sqrt(ps * (1 - ps) / A.B)),
                "erreur_mc_p_bootstrap_t": float(np.sqrt(pb * (1 - pb) / 10_000)), "p": max(ps, pb)}
    tests = {}
    for nom, t, c in TESTS:
        tests[nom] = {"tache": t, "codage": c, **test_ni(K[(t, c)] - K[(t, "fp32")])}
    for (nom, *_), q in zip(TESTS, A.holm([tests[n]["p"] for n, *_ in TESTS])):
        tests[nom]["p_holm"] = float(q)
        tests[nom]["verdict"] = "soutenue" if q < A.ALPHA else "non soutenue"
    for (nom, t, c), niveau in zip(TESTS, niveaux_holm([tests[n]["p"] for n, *_ in TESTS])):
        x = (K[(t, c)] - K[(t, "fp32")]) / 5          # borne inférieure unilatérale, bootstrap-t, au niveau de Holm
        tests[nom]["niveau_holm"] = float(niveau)
        tests[nom]["borne_inferieure_points"] = 100 * borne_bootstrap_t(x, niveau)
        tests[nom]["borne_et_verdict_concordent"] = (tests[nom]["borne_inferieure_points"] > -MARGE) == (tests[nom]["verdict"] == "soutenue")
    controle = {t: test_ni(K[(t, "replique")] - K[(t, "fp32")]) for t in TACHES}
    sensibilite = {nom: test_ni(K[(t, c)] - (K[(t, "fp32")] + K[(t, "replique")]) / 2) for nom, t, c in TESTS}
    desc = {}
    for t in TACHES:
        desc[t] = {"reussite_fp32": float(K[(t, "fp32")].mean() / 5), "reussite_ne16": float(K[(t, "ne16")].mean() / 5),
                   **{f"reussite_{c}": float(K[(t, c)].mean() / 5) for c in CODAGES},
                   **{f"{c}_moins_ne16": boot.ic((K[(t, c)] - K[(t, "ne16")]) / 5) for c in CODAGES},
                   "temps_par_plan_ms": {c: float(1000 * np.median([f["par_plan_s"] for f in temps[(t, c)]])) for c in CODAGES},
                   "octets_poids": {c: temps[(t, c)][0]["coreml"].get("octets_poids") for c in CODAGES}}
    return {"controles": {"commit": sha}, "tests": tests, "controle_negatif": controle, "sensibilite_10_graines": sensibilite,
            "descriptifs": desc}


def rapport(r):
    t = lambda e, f=100: A.ic_txt(e, f, 1)
    L = ["# EXP-014 : analyse pré-enregistrée (codages des poids du prédicteur dans Core ML, sur le Neural Engine)", "",
         f"Contrôles passés, commit {r['controles']['commit'][:7]} ; reproductions PyTorch des graines 42 et 46 identiques.", "",
         f"| Test | Tâche | Codage | Écart à PyTorch float32 (points, IC 95 %) | Borne inférieure au niveau de Holm | p signe | p bootstrap-t | p Holm | Perte de moins de {MARGE} points |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for n, x in r["tests"].items():
        L.append(f"| {n} | {x['tache']} | {x['codage']} | {t(x['ecart_points'])} | {x['borne_inferieure_points']:.1f} "
                 f"(niveau {x['niveau_holm']:.4f}) | {x['p_signe']:.4f} | {x['p_bootstrap_t']:.4f} | {x['p_holm']:.4f} | {x['verdict']} |")
    for n, x in r["tests"].items():
        if not x["borne_et_verdict_concordent"]:
            L.append(f"- Attention, {n} : la borne inférieure et le verdict ne concordent pas (l'inversion de signe décide).")
    L += ["", "Contrôle de la marge (deux séries de la pleine précision, graines 47 à 51 contre 42 à 46, écart vrai nul) :"]
    for k, c in r["controle_negatif"].items():
        L.append(f"- {k} : écart {t(c['ecart_points'])} points, p signe {c['p_signe']:.4f}, p bootstrap-t {c['p_bootstrap_t']:.4f}")
    L += ["", "Sensibilité, référence sur les 10 graines de la pleine précision :"]
    for n, c in r["sensibilite_10_graines"].items():
        L.append(f"- {n} : écart {t(c['ecart_points'])} points, p {c['p']:.4f}")
    L += ["", "| Tâche | Version | Réussite | Effet du codage seul (contre Core ML 16 bits sur le Neural Engine, EXP-012 ; points) | Temps par plan (ms) | Fichier des poids (Mo) |",
          "| --- | --- | --- | --- | --- | --- |"]
    for k, d in r["descriptifs"].items():
        L.append(f"| {k} | PyTorch float32 | {100 * d['reussite_fp32']:.1f} % | | | |")
        L.append(f"| {k} | Core ML 16 bits, Neural Engine (EXP-012) | {100 * d['reussite_ne16']:.1f} % | | | 23.2 |")
        for c in CODAGES:
            o = d["octets_poids"][c]
            L.append(f"| {k} | {c} | {100 * d[f'reussite_{c}']:.1f} % | {t(d[f'{c}_moins_ne16'])} | {d['temps_par_plan_ms'][c]:.0f} | "
                     f"{'' if o is None else f'{o / 1e6:.1f}'} |")
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
