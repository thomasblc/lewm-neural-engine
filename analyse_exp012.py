"""Analyse pré-enregistrée d'EXP-012 (PREREG-EXP012.md, amendement 1) : le prédicteur en Core ML 16 bits, sur le GPU
et sur le Neural Engine, contre la pleine précision PyTorch, sur PushT et TwoRoom. Contrôles bloquants (dont trois
reproductions PyTorch octet pour octet), 4 tests de non-infériorité (inversion de signe et bootstrap-t concordants,
Holm), contrôle négatif (deuxième série de la pleine précision), sensibilité à 10 graines de référence, temps par
plan, puissance et énergie par plan au-dessus du repos.

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
REPLIQUE = (47, 48, 49, 50, 51)                     # deuxième série de la pleine précision, déjà calculée
MOTEURS = {"ne": "coreml-ne", "gpu": "coreml-gpu"}
PLACEMENT = {"ne": "NeuralEngine", "gpu": "GPU"}
MARGE = 2                                           # points
B_BOOT, GRAINE_BOOT = 10_000, 4
TACHES = {"pusht": "exp003", "tworoom": "exp006"}  # référence fp32 PyTorch
TESTS = (("N1", "pusht", "ne"), ("N2", "tworoom", "ne"), ("N3", "pusht", "gpu"), ("N4", "tworoom", "gpu"))
FICHIERS_ANALYSE = ("analyse_exp012.py", "analyse_exp009.py", "analyse_exp006.py", "analyse_exp003.py", "metriques.py",
                    "quant.py", "run_grid.py", "coreml_pred.py", "predicteur_ane.py", "mesurer_repos.py", "PREREG-EXP012.md")
ICI = Path(__file__).resolve().parent


def git(*args):
    return subprocess.run(["git", "-C", str(ICI), *args], capture_output=True, text=True)


def verifier_code(sha):
    A.exiger(git("cat-file", "-e", f"{sha}^{{commit}}").returncode == 0, f"commit {sha[:7]} inconnu du dépôt")
    A.exiger(git("diff", "--quiet", sha, "--", *FICHIERS_ANALYSE).returncode == 0,
             f"code d'analyse ou de calcul différent de celui du commit {sha[:7]}")


def dossier(t, m):
    return f"exp012-{m}-{t}"


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


def borne_bootstrap_t(x, niveau, b=B_BOOT, graine=GRAINE_BOOT):
    """Borne inférieure unilatérale de la moyenne de x au niveau donné, par bootstrap-t (même rééchantillonnage que la
    p du bootstrap-t) : moyenne - quantile (1 - niveau) de t* x erreur-type."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(n)
    if se == 0:
        return float(m)
    xb = x[np.random.default_rng(graine).integers(0, n, size=(b, n))]
    seb = xb.std(axis=1, ddof=1) / np.sqrt(n)
    tb = (xb.mean(axis=1) - m) / np.where(seb > 0, seb, np.inf)
    return float(m - np.quantile(tb, 1 - niveau) * se)


def niveaux_holm(ps, alpha=A.ALPHA):
    """Niveau de chaque test dans la procédure de Holm : alpha / (m - rang) tant qu'on rejette ; après le premier test
    gardé, tous les suivants au niveau du premier gardé, alpha / (m - k), k le nombre de rejets."""
    m = len(ps)
    niv = [None] * m
    k, arrete = 0, False
    for rang, i in enumerate(np.argsort(ps)):
        if not arrete and ps[i] <= alpha / (m - rang):
            niv[i] = alpha / (m - rang)
            k += 1
        else:
            arrete = True
            niv[i] = alpha / (m - k)
    return niv


def repos_avant(r, t, debut, sha):
    """Dernière mesure de la puissance au repos de la tâche faite avant debut : même commit, au moins 50 s couvertes,
    aucun calcul run_grid pendant la mesure."""
    cand = []
    for f in sorted(r.glob(f"exp012-repos-{t}-*.json")):
        j = json.loads(f.read_text())
        if j["fin"] <= debut:
            cand.append((j["fin"], f, j))
    A.exiger(cand, f"{t} : aucune mesure de la puissance au repos avant un fichier (début {debut:.0f})")
    _, f, j = max(cand, key=lambda c: c[0])
    A.exiger(j["git"] == sha and j["secondes_couvertes"] >= 50 and j["puissance_moyenne_w"] is not None
             and max(x.get("processus_run_grid", 0) for x in j["lectures"]) == 0, f"{f.name} : mesure au repos invalide")
    return j["puissance_moyenne_w"]


def lire_temps(f, moteur, sha):
    j = json.loads(f.read_text())
    A.exiger(j["moteur"] == moteur and j["git"] == sha, f"{f.parent.name}/{f.name} : fichier de temps")
    d = np.array([x[0] for x in j["appels_solve"]], dtype=np.float64)
    n = np.array([x[1] for x in j["appels_solve"]], dtype=np.float64)
    pp = d / np.maximum(n, 1)
    garde = pp <= 3 * np.median(pp)                   # un appel allongé par une pause de batterie est écarté
    return {"par_plan_s": float(d[garde].sum() / n[garde].sum()), "plans": int(n.sum()), "appels_ecartes": int((~garde).sum()),
            "puissance_w": j["puissance"]["puissance_moyenne_w"], "coreml": j["coreml"],
            "debut": float(j["lectures"][0]["t"]) if j.get("lectures") else None}


def charger(racine, verifier=True):
    r = Path(racine)
    K, temps, repos, shas = {}, {}, {}, set()
    for t, ref_d in TACHES.items():
        controler = A6.controler if t == "tworoom" else (lambda d, nom: A.controler_course(d, nom))
        ref = dict(np.load(r / ref_d / "fp32_s42.npz"))
        proto_ref = json.loads((r / ref_d / "meta.json").read_text())["protocole"]
        for d, g in ((f"exp012-repro-{t}", 42), (f"exp012-torch-{t}", 43), (f"exp012-fin-{t}", 46)):
            A9.identiques(r / d / f"fp32_s{g}.npz", r / ref_d / f"fp32_s{g}.npz", f"reproduction {t} graine {g}")

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
        K[(t, "replique")] = sum(lire(ref_d, g)["success"].astype(np.int64) for g in REPLIQUE)
        for m, moteur in MOTEURS.items():
            d = dossier(t, m)
            proto = json.loads((r / d / "meta.json").read_text())["protocole"]
            A.exiger({k: v for k, v in proto.items() if k != "moteur"} == proto_ref,
                     f"{d} : protocole différent de celui de la référence {ref_d}")
            mo = proto.get("moteur", {})
            A.exiger(mo.get("nom") == moteur and mo.get("precision") == "float16", f"{d} : moteur {mo}")
            S, empreintes, fichiers = 0, set(), []
            for g in GRAINES:
                x = lire(d, g, moteur)
                shas.add(str(x["git"]))
                S = S + x["success"].astype(np.int64)
                tf = lire_temps(r / d / f"fp32_s{g}.temps.json", moteur, str(x["git"]))
                A.exiger(set(tf["coreml"].get("placement", {})) == {PLACEMENT[m]},
                         f"{d} graine {g} : opérations hors du processeur demandé {tf['coreml'].get('placement')}")
                A.exiger(tf["coreml"].get("sha256_poids") == str(x["coreml_sha256"]), f"{d} graine {g} : empreintes")
                empreintes.add((str(x["coreml_sha256"]), tf["coreml"].get("sha256_modele")))
                fichiers.append(tf)
            A.exiger(len(empreintes) == 1, f"{d} : modèle Core ML différent selon les graines")
            K[(t, m)] = S
            temps[(t, m)] = fichiers
    A.exiger(len(shas) == 1, f"EXP-012 : fichiers de plusieurs commits {sorted(shas)}")
    sha = shas.pop()
    for t in TACHES:
        for d, g in ((f"exp012-repro-{t}", 42), (f"exp012-torch-{t}", 43), (f"exp012-fin-{t}", 46)):
            x = np.load(r / d / f"fp32_s{g}.npz")
            A.exiger(str(x["git"]) == sha and not bool(x["git_sale"]), f"reproduction {t} graine {g} : autre commit ou arbre modifié")
        temps[(t, "torch")] = [lire_temps(r / f"exp012-torch-{t}" / "fp32_s43.temps.json", "torch", sha)]
        for m in ("torch", *MOTEURS):                # à chaque fichier, la dernière mesure au repos faite avant lui
            for f in temps[(t, m)]:
                A.exiger(f["debut"] is not None, f"{t} {m} : fichier de temps sans lectures")
                f["repos_w"] = repos_avant(r, t, f["debut"], sha)
    if verifier:
        verifier_code(sha)
    return K, temps, sha


def resume_temps(fichiers):
    """Par fichier : somme des durées des appels de solve sur le nombre de plans (appels allongés par une pause
    écartés), puissance moyenne pendant le calcul (pauses exclues), énergie par plan au-dessus de la dernière mesure au
    repos faite avant le fichier ; résumés par la médiane des fichiers, avec le minimum et le maximum."""
    pp = np.array([f["par_plan_s"] for f in fichiers])
    pw = np.array([f["puissance_w"] if f["puissance_w"] is not None else np.nan for f in fichiers])
    rp = np.array([f["repos_w"] for f in fichiers])
    en = (pw - rp) * pp
    p_repos = float(np.median(rp))
    med = lambda v: None if np.all(np.isnan(v)) else float(np.nanmedian(v))
    return {"fichiers": len(fichiers), "plans": int(sum(f["plans"] for f in fichiers)),
            "appels_ecartes": int(sum(f["appels_ecartes"] for f in fichiers)),
            "par_plan_ms": float(1000 * np.median(pp)), "par_plan_ms_min": float(1000 * pp.min()), "par_plan_ms_max": float(1000 * pp.max()),
            "puissance_w": med(pw), "puissance_repos_w": p_repos, "energie_par_plan_j": med(en)}


def analyser(racine, verifier=True):
    K, temps, sha = charger(racine, verifier)
    inv, boot = A.Inversion(n=500), A.Bootstrap(n=500)

    def test_ni(x):
        y = 20 * x + MARGE                            # moyenne de y positive ssi écart de réussite > -2 points
        ps, pb = inv.p(y, "plus"), p_bootstrap_t(y)
        return {"somme": float(x.sum()), "ecart_points": boot.ic(x / 5), "p_signe": ps, "p_bootstrap_t": pb,
                "erreur_mc_p_signe": float(np.sqrt(ps * (1 - ps) / A.B)),
                "erreur_mc_p_bootstrap_t": float(np.sqrt(pb * (1 - pb) / B_BOOT)), "p": max(ps, pb)}
    tests = {}
    for nom, t, m in TESTS:
        tests[nom] = {"tache": t, "moteur": MOTEURS[m], **test_ni(K[(t, m)] - K[(t, "fp32")])}
    for (nom, *_), q in zip(TESTS, A.holm([tests[n]["p"] for n, *_ in TESTS])):
        tests[nom]["p_holm"] = float(q)
        tests[nom]["verdict"] = "soutenue" if q < A.ALPHA else "non soutenue"
    niv = niveaux_holm([tests[n]["p"] for n, *_ in TESTS])
    for (nom, t, m), niveau in zip(TESTS, niv):     # borne inférieure unilatérale, bootstrap-t, au niveau de Holm
        x = (K[(t, m)] - K[(t, "fp32")]) / 5
        tests[nom]["niveau_holm"] = float(niveau)
        tests[nom]["borne_inferieure_points"] = 100 * borne_bootstrap_t(x, niveau)
        tests[nom]["borne_et_verdict_concordent"] = (tests[nom]["borne_inferieure_points"] > -MARGE) == (tests[nom]["verdict"] == "soutenue")
    controle = {t: test_ni(K[(t, "replique")] - K[(t, "fp32")]) for t in TACHES}
    sensibilite = {}
    for nom, t, m in TESTS:                           # référence : moyenne des 10 graines de la pleine précision
        x = K[(t, m)] - (K[(t, "fp32")] + K[(t, "replique")]) / 2
        sensibilite[nom] = test_ni(x)
    desc = {}
    for t in TACHES:
        desc[t] = {"reussite_fp32": float(K[(t, "fp32")].mean() / 5), "reussite_replique": float(K[(t, "replique")].mean() / 5),
                   **{f"reussite_{m}": float(K[(t, m)].mean() / 5) for m in MOTEURS},
                   "ne_moins_gpu": boot.ic((K[(t, "ne")] - K[(t, "gpu")]) / 5),
                   "temps": {m: resume_temps(temps[(t, m)]) for m in ("torch", *MOTEURS)}}
    return {"controles": {"commit": sha}, "tests": tests, "controle_negatif": controle, "sensibilite_10_graines": sensibilite,
            "descriptifs": desc}


def rapport(r):
    t = lambda e, f=100: A.ic_txt(e, f, 1)
    L = ["# EXP-012 : analyse pré-enregistrée (le prédicteur en Core ML 16 bits, GPU et Neural Engine)", "",
         f"Contrôles passés, commit {r['controles']['commit'][:7]} ; reproductions PyTorch des graines 42, 43 et 46 identiques.", "",
         f"| Test | Tâche | Moteur | Écart (points, IC 95 %) | Borne inférieure au niveau de Holm | p signe | p bootstrap-t | p Holm | Perte de moins de {MARGE} points |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for n, x in r["tests"].items():
        L.append(f"| {n} | {x['tache']} | {x['moteur']} | {t(x['ecart_points'])} | {x['borne_inferieure_points']:.1f} "
                 f"(niveau {x['niveau_holm']:.4f}) | {x['p_signe']:.4f} | {x['p_bootstrap_t']:.4f} | {x['p_holm']:.4f} | {x['verdict']} |")
    for n, x in r["tests"].items():
        if not x["borne_et_verdict_concordent"]:
            L.append(f"- Attention, {n} : la borne inférieure et le verdict ne concordent pas (l'inversion de signe décide).")
    L.append("- Erreur de Monte-Carlo des p : " + " ; ".join(
        f"{n} {x['erreur_mc_p_signe']:.4f} (signe), {x['erreur_mc_p_bootstrap_t']:.4f} (bootstrap-t)" for n, x in r["tests"].items()))
    L += ["", "Contrôle de la marge (deux séries de la pleine précision, graines 47 à 51 contre 42 à 46, écart vrai nul) :"]
    for k, c in r["controle_negatif"].items():
        L.append(f"- {k} : écart {t(c['ecart_points'])} points, p signe {c['p_signe']:.4f}, p bootstrap-t {c['p_bootstrap_t']:.4f}")
    L += ["", "Sensibilité, référence sur les 10 graines de la pleine précision :"]
    for n, c in r["sensibilite_10_graines"].items():
        L.append(f"- {n} : écart {t(c['ecart_points'])} points, p {c['p']:.4f}")
    L += ["", "| Tâche | Moteur | Réussite | Temps par plan (ms, médiane des fichiers, min à max) | Puissance (W) | Repos (W) | Énergie par plan au-dessus du repos (J) |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for k, d in r["descriptifs"].items():
        for m, nom in (("torch", "PyTorch float32 (réussite : 5 graines ; temps : graine 43)"), ("gpu", "Core ML GPU"),
                       ("ne", "Core ML Neural Engine")):
            s = d["temps"][m]
            reu = d["reussite_fp32"] if m == "torch" else d[f"reussite_{m}"]
            f = lambda v, n=1: "?" if v is None else f"{v:.{n}f}"
            L.append(f"| {k} | {nom} | {100 * reu:.1f} % | {s['par_plan_ms']:.0f} ({s['par_plan_ms_min']:.0f} à {s['par_plan_ms_max']:.0f}) | "
                     f"{f(s['puissance_w'])} | {f(s['puissance_repos_w'])} | {f(s['energie_par_plan_j'], 2)} |")
        L.append(f"| {k} | Neural Engine moins GPU | {t(d['ne_moins_gpu'])} points | | | | |")
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
