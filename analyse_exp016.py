"""Analyse pré-enregistrée d'EXP-016 (PREREG-EXP016.md) : la compression casse-t-elle des épisodes précis ? Relit les
fichiers existants, aucun nouveau run.

Par tâche : k_R(e) = réussites de la pleine précision sur les graines 47 à 51 (référence), k_V(e) = réussites d'une
version sur les graines 42 à 46. T1 = épisodes avec k_R = 5 et k_V = 0. Modèle nul : difficulté p(e) a posteriori
(bêta-binomiale ajustée sur les k_R), version q(e) = min(1, r p(e)) (amincissement, principal) ou logit q = logit p +
delta (sensibilité), r ou delta réglé à chaque tirage sur la réussite moyenne de la version. p unilatérale sur 10 000
tirages. Holm sur la famille principale ; contrôles négatifs ; descriptifs E1 à E4.

Usage :
    python analyse_exp016.py --racine ../resultats --sortie ../resultats/analyse-exp016
"""
import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln, expit, logit

import analyse_exp003 as A
import analyse_exp006 as A6

G_V = (42, 43, 44, 45, 46)
G_R = (47, 48, 49, 50, 51)
B = 10_000
GRAINE = 16
REF = {"pusht": "exp003", "tworoom": "exp006"}
# (tâche, nom, dossier, fichier sans graine)
PRINCIPALES = (("pusht", "RTN 4 bits", "exp003", "all-4"), ("pusht", "RTN 3 bits", "exp003", "all-3"),
               ("pusht", "GPTQ 3 bits", "exp009-pusht", "all-3-gptq"), ("pusht", "int4 par canal, Neural Engine", "exp014-int4c-pusht", "fp32"),
               ("pusht", "palettes, Neural Engine", "exp014-pal4-pusht", "fp32"),
               ("tworoom", "RTN 4 bits", "exp006", "all-4"), ("tworoom", "RTN 3 bits", "exp006", "all-3"))
NEGATIFS = (("pusht", "pleine précision, graines 42 à 46", "exp003", "fp32"), ("pusht", "RTN 8 bits", "exp003", "all-8"),
            ("tworoom", "pleine précision, graines 42 à 46", "exp006", "fp32"), ("tworoom", "RTN 8 bits", "exp006", "all-8"))
SECONDAIRES = (("pusht", "GPTQ 4 bits", "exp009-pusht", "all-4-gptq"), ("tworoom", "GPTQ 4 bits", "exp009-tworoom", "all-4-gptq"),
               ("tworoom", "GPTQ 3 bits", "exp009-tworoom", "all-3-gptq"),
               ("pusht", "encodeur 4 bits", "exp003", "encoder-4"), ("pusht", "prédicteur 4 bits", "exp003", "predictor-4"),
               ("pusht", "autres 4 bits", "exp004", "autres-4"), ("pusht", "encodeur et prédicteur 4 bits", "exp004", "encpred-4"),
               ("tworoom", "encodeur 4 bits", "exp006", "encoder-4"), ("tworoom", "prédicteur 4 bits", "exp006", "predictor-4"),
               ("tworoom", "autres 4 bits", "exp006", "autres-4"),
               ("pusht", "prédicteur ternaire RTN", "exp011-pusht", "predictor-2-clip"),
               ("pusht", "prédicteur ternaire GPTQ", "exp011-pusht", "predictor-2-clip-gptq"),
               ("tworoom", "prédicteur ternaire RTN", "exp011-tworoom", "predictor-2-clip"),
               ("tworoom", "prédicteur ternaire GPTQ", "exp011-tworoom", "predictor-2-clip-gptq"),
               ("pusht", "int8, Neural Engine", "exp014-int8-pusht", "fp32"), ("tworoom", "int8, Neural Engine", "exp014-int8-tworoom", "fp32"),
               ("tworoom", "int4 par canal, Neural Engine", "exp014-int4c-tworoom", "fp32"),
               ("tworoom", "palettes, Neural Engine", "exp014-pal4-tworoom", "fp32"))
PAIRES = ((("pusht", "RTN 4 bits"), ("pusht", "RTN 3 bits")), (("pusht", "RTN 3 bits"), ("pusht", "GPTQ 3 bits")),
          (("pusht", "RTN 4 bits"), ("pusht", "int4 par canal, Neural Engine")),
          (("pusht", "int4 par canal, Neural Engine"), ("pusht", "palettes, Neural Engine")))
FICHIERS_ANALYSE = ("analyse_exp016.py", "analyse_exp003.py", "analyse_exp006.py", "PREREG-EXP016.md")
ICI = Path(__file__).resolve().parent


def git(*args):
    return subprocess.run(["git", "-C", str(ICI), *args], capture_output=True, text=True)


def verifier_code(sha):
    A.exiger(git("cat-file", "-e", f"{sha}^{{commit}}").returncode == 0, f"commit {sha[:7]} inconnu du dépôt")
    A.exiger(git("diff", "--quiet", sha, "--", *FICHIERS_ANALYSE).returncode == 0, f"code d'analyse différent du commit {sha[:7]}")


def comptes(r, tache, d, v, graines):
    """Réussites par épisode (somme sur les graines), fichiers contrôlés et appariés à la référence de la tâche."""
    ref = dict(np.load(r / REF[tache] / "fp32_s42.npz"))
    controler = A6.controler if tache == "tworoom" else (lambda x, nom: A.controler_course(x, nom))
    S = np.zeros(len(ref["episodes_idx"]), dtype=np.int64)
    for g in graines:
        f = r / d / f"{v}_s{g}.npz"
        A.exiger(f.exists(), f"{d}/{f.name} manquant")
        x = dict(np.load(f))
        A.exiger(int(x["seed"]) == g and str(x["spec"]).replace(":", "-") == v, f"{d}/{f.name} : version ou graine")
        A.exiger(not bool(x["git_sale"]), f"{d}/{f.name} : arbre git modifié")
        for k in ("episodes_idx", "start_steps", "init_state", "goal_state"):
            A.exiger(np.array_equal(x[k], ref[k]), f"{d}/{f.name} : {k} différent de la référence")
        controler(x, f"{d}/{f.name}")
        S += x["success"].astype(np.int64)
    return S


def ajuster_beta(k, n=5):
    """Loi a priori bêta (alpha, beta) des difficultés, maximum de vraisemblance bêta-binomial sur les comptes k."""
    k = np.asarray(k)
    nll = lambda t: -np.sum(betaln(k + np.exp(t[0]), n - k + np.exp(t[1])) - betaln(np.exp(t[0]), np.exp(t[1])))
    res = minimize(nll, x0=np.log([0.5, 0.5]), method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 4000})
    A.exiger(res.success, f"ajustement bêta-binomial : {res.message}")
    return float(np.exp(res.x[0])), float(np.exp(res.x[1]))


def regler(p, cible, nul, iterations=80):
    """r (amincissement) ou delta (logistique) par dichotomie, un par tirage, pour que la moyenne des q égale la cible."""
    if nul == "amincissement":
        bas, haut = np.zeros(len(p)), np.full(len(p), 1e6)   # r > 1 possible (version meilleure que la référence)
        f = lambda c: np.minimum(1.0, c[:, None] * p)
    else:
        lp = logit(np.clip(p, 1e-12, 1 - 1e-12))
        bas, haut = np.full(len(p), -40.0), np.full(len(p), 40.0)
        f = lambda c: expit(lp + c[:, None])
    for _ in range(iterations):
        mil = (bas + haut) / 2
        trop = f(mil).mean(axis=1) > cible
        haut, bas = np.where(trop, mil, haut), np.where(trop, bas, mil)
    return f((bas + haut) / 2)


def tirages_nuls(kR, kV, alpha, beta, nul, b=B, graine=GRAINE, n=5):
    """T1 simulés sous le nul : p(e) a posteriori, q(e) réglé sur la réussite moyenne de la version, k(e) binomial."""
    rng = np.random.default_rng(graine)
    p = rng.beta(alpha + kR, beta + n - kR, size=(b, len(kR)))
    q = regler(p, kV.mean() / n, nul)
    k = rng.binomial(n, q)
    return ((kR == n) & (k == 0)).sum(axis=1), k


def t1(kR, kV, n=5):
    return int(((kR == n) & (kV == 0)).sum())


def etudier(kR, kV, alpha, beta, kV42=None):
    obs = t1(kR, kV)
    res = {"T1": obs, "toujours_reussis_reference": int((kR == 5).sum()), "part_cassee": obs / max(1, int((kR == 5).sum())),
           "reussite_version": float(kV.mean() / 5), "reussite_reference": float(kR.mean() / 5)}
    for nul in ("amincissement", "logistique"):
        sim, k = tirages_nuls(kR, kV, alpha, beta, nul)
        res[nul] = {"p": float((1 + (sim >= obs).sum()) / (len(sim) + 1)), "T1_attendu": float(sim.mean()),
                    "rapport_observe_attendu": float(obs / sim.mean()) if sim.mean() > 0 else None}
        if nul == "amincissement":                      # calibration : comptes (k_R = 5, k_V = j) observés et attendus
            res["calibration_kR5"] = {"observe": [int(((kR == 5) & (kV == j)).sum()) for j in range(6)],
                                      "attendu": [float(((kR == 5) & (k == j)).sum(axis=1).mean()) for j in range(6)]}
    jamais = kV == 0                                     # E2 : nouveaux ou amplifiés
    res["E2_jamais_reussis_par_kR"] = [int((jamais & (kR == j)).sum()) for j in range(6)]
    nj = max(1, int(jamais.sum()))
    res["E2_part_nouveaux"] = float((jamais & (kR == 5)).sum() / nj)
    res["E2_part_deja_fragiles"] = float((jamais & (kR <= 2)).sum() / nj)
    if kV42 is not None:                                 # E3 : au moins une réussite en 5 graines
        moy, best = kV42.mean() / 5 - kV.mean() / 5, (kV42 > 0).mean() - (kV > 0).mean()
        res["E3"] = {"au_moins_une_version": float((kV > 0).mean()), "au_moins_une_pleine_precision": float((kV42 > 0).mean()),
                     "perte_moyenne": float(moy), "perte_au_meilleur_des_5": float(best),
                     "rapport_meilleur_sur_moyenne": float(best / moy) if moy > 0 else None}
    return res


def analyser(racine, verifier=True):
    r = Path(racine)
    for t, d in REF.items():                              # fichiers de référence, chacun de son commit et propre
        for g in G_R + G_V:
            x = np.load(r / d / f"fp32_s{g}.npz")
            A.exiger(not bool(x["git_sale"]), f"{d}/fp32_s{g}.npz : arbre git modifié")
    kR = {t: comptes(r, t, REF[t], "fp32", G_R) for t in REF}
    k42 = {t: comptes(r, t, REF[t], "fp32", G_V) for t in REF}
    prior = {t: ajuster_beta(kR[t]) for t in REF}
    ajust = {}
    for t, (a, b_) in prior.items():
        att = [float(len(kR[t]) * np.exp(betaln(j + a, 5 - j + b_) - betaln(a, b_) + np.log(float(math.comb(5, j))))) for j in range(6)]
        ajust[t] = {"alpha": a, "beta": b_, "observe": [int((kR[t] == j).sum()) for j in range(6)], "attendu": att}
    sortie = {"loi_a_priori": ajust, "principales": {}, "negatifs": {}, "secondaires": {}}
    K = {}
    for groupe, liste in (("principales", PRINCIPALES), ("negatifs", NEGATIFS), ("secondaires", SECONDAIRES)):
        for t, nom, d, v in liste:
            kV = comptes(r, t, d, v, G_V)
            K[(t, nom)] = kV
            sortie[groupe][f"{t} | {nom}"] = {"dossier": d, "fichier": v, **etudier(kR[t], kV, *prior[t], kV42=k42[t])}
    noms = list(sortie["principales"])
    ps = [sortie["principales"][n]["amincissement"]["p"] for n in noms]
    for n, q in zip(noms, A.holm(ps)):
        sortie["principales"][n]["p_holm"] = float(q)
        sortie["principales"][n]["verdict"] = "soutenue" if q < A.ALPHA else "non soutenue"
    neg_ok = all(v["amincissement"]["p"] >= 0.05 for v in sortie["negatifs"].values())
    sortie["controles_negatifs_acceptes"] = neg_ok
    sortie["lecture"] = ("contrôles négatifs acceptés : verdicts lisibles" if neg_ok
                         else "un contrôle négatif rejette : modèle nul mal calibré, conclusions de S non concluantes")
    rng = np.random.default_rng(GRAINE + 1)               # E4 : recouvrement des ensembles cassés
    e4 = {}
    for (ta, na), (tb, nb) in PAIRES:
        A.exiger(ta == tb, "paire de tâches différentes")
        sa, sb = (kR[ta] == 5) & (K[(ta, na)] == 0), (kR[ta] == 5) & (K[(tb, nb)] == 0)
        inter, union = int((sa & sb).sum()), int((sa | sb).sum())
        pool = np.flatnonzero(kR[ta] == 5)
        jac = []
        for _ in range(B):
            a_ = rng.choice(pool, int(sa.sum()), replace=False)
            b2 = rng.choice(pool, int(sb.sum()), replace=False)
            u = len(np.union1d(a_, b2))
            jac.append(len(np.intersect1d(a_, b2)) / u if u else 0.0)
        e4[f"{na} et {nb} ({ta})"] = {"jaccard": inter / union if union else None, "jaccard_hasard": float(np.mean(jac)),
                                     "communs": inter, "tailles": [int(sa.sum()), int(sb.sum())]}
    sortie["E4_recouvrement"] = e4
    sha = git("rev-parse", "HEAD").stdout.strip()
    sortie["commit_analyse"] = sha
    if verifier:
        A.exiger(git("status", "--porcelain", "--untracked-files=no").stdout.strip() == "", "arbre git modifié")
        verifier_code(sha)
    return sortie


def rapport(s):
    L = ["# EXP-016 : rapport de l'analyse pré-enregistrée (épisodes cassés par la compression)", "",
         f"Commit de l'analyse : `{s['commit_analyse'][:7]}`. {s['lecture']}.", "",
         "| Tâche, version | Réussite | Toujours réussis (réf.) | Cassés (T1) | T1 attendu, amincissement | p | p Holm | p logistique | Verdict |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for n, v in s["principales"].items():
        L.append(f"| {n} | {100 * v['reussite_version']:.1f} % | {v['toujours_reussis_reference']} | {v['T1']} | "
                 f"{v['amincissement']['T1_attendu']:.2f} | {v['amincissement']['p']:.4f} | {v['p_holm']:.4f} | "
                 f"{v['logistique']['p']:.4f} | {v['verdict']} |")
    L += ["", "Contrôles négatifs (le nul doit les accepter) :", ""]
    for n, v in s["negatifs"].items():
        L.append(f"- {n} : T1 {v['T1']}, attendu {v['amincissement']['T1_attendu']:.2f}, p {v['amincissement']['p']:.4f} ; "
                 f"calibration (k_R = 5, k_V = 0 à 5) observé {v['calibration_kR5']['observe']}, attendu "
                 f"{[round(x, 1) for x in v['calibration_kR5']['attendu']]}")
    L += ["", "E2 et E3 :", ""]
    for g in ("principales", "secondaires"):
        for n, v in s[g].items():
            e3 = v["E3"]
            L.append(f"- {n} : jamais réussis par k_R {v['E2_jamais_reussis_par_kR']}, nouveaux {100 * v['E2_part_nouveaux']:.0f} %, "
                     f"déjà fragiles {100 * v['E2_part_deja_fragiles']:.0f} % ; perte moyenne {100 * e3['perte_moyenne']:.1f} points, "
                     f"au meilleur des 5 {100 * e3['perte_au_meilleur_des_5']:.1f} points ; T1 {v['T1']}, p {v['amincissement']['p']:.4f}")
    L += ["", "E4, recouvrement des épisodes cassés :", ""] + [
        f"- {k} : Jaccard {v['jaccard'] if v['jaccard'] is None else round(v['jaccard'], 3)} (hasard {v['jaccard_hasard']:.3f}), "
        f"{v['communs']} communs, tailles {v['tailles']}" for k, v in s["E4_recouvrement"].items()]
    L += ["", "Loi a priori des difficultés (bêta-binomiale, référence 47 à 51) :", ""] + [
        f"- {t} : alpha {v['alpha']:.3f}, beta {v['beta']:.3f}, k_R observé {v['observe']}, attendu {[round(x, 1) for x in v['attendu']]}"
        for t, v in s["loi_a_priori"].items()]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--racine", required=True)
    ap.add_argument("--sortie", required=True)
    a = ap.parse_args()
    s = A.pur(analyser(a.racine))
    out = Path(a.sortie)
    out.mkdir(parents=True, exist_ok=True)
    (out / "resultats.json").write_text(json.dumps(s, indent=1, ensure_ascii=False))
    (out / "rapport.md").write_text(rapport(s))
    print(rapport(s))


if __name__ == "__main__":
    main()
