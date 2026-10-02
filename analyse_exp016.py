"""Analyse pré-enregistrée d'EXP-016, version 2 (PREREG-EXP016.md) : la compression casse-t-elle des épisodes précis ?
Relit les fichiers existants, aucun nouveau run.

Par tâche : k_R(e) = réussites de la pleine précision sur les graines 47 à 51 (référence), k_42(e) sur les graines 42
à 46, k_V(e) = réussites d'une version sur les graines 42 à 46. Loi a priori des difficultés non paramétrique (grille,
EM sur k_R + k_42, 10 essais) ; loi a posteriori à partir de k_R seul. Test principal dans la strate k_R = 5 : T1 =
épisodes jamais réussis par la version ; nul de perte proportionnelle dans la strate (q = min(1, r p), r réglé sur la
réussite de la version dans la strate). Calibration sur des versions synthétiques (pleine précision 42 à 46 amincie).
Holm sur la famille principale, excès minimal, contrôles ; descriptifs E1 à E4.

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
from scipy.special import betaln
from scipy.stats import chi2

import analyse_exp003 as A
import analyse_exp006 as A6

G_V = (42, 43, 44, 45, 46)
G_R = (47, 48, 49, 50, 51)
GRILLE = np.linspace(0.0, 1.0, 201)
B, B_SYN, N_SYN = 10_000, 1_000, 1_000
GRAINE = 16
EXCES_MIN = 5
ALPHA = 0.05
SEUILS_SYN = {"0.05": 0.064, "0.05/7": 0.012}
REF = {"pusht": "exp003", "tworoom": "exp006"}
PRINCIPALES = (("pusht", "RTN 4 bits", "exp003", "all-4"), ("pusht", "RTN 3 bits", "exp003", "all-3"),
               ("pusht", "GPTQ 3 bits", "exp009-pusht", "all-3-gptq"), ("pusht", "int4 par canal, Neural Engine", "exp014-int4c-pusht", "fp32"),
               ("pusht", "palettes, Neural Engine", "exp014-pal4-pusht", "fp32"),
               ("tworoom", "RTN 4 bits", "exp006", "all-4"), ("tworoom", "RTN 3 bits", "exp006", "all-3"))
MOTEUR = (("pusht", "16 bits, Neural Engine", "exp012-ne-pusht", "fp32"), ("tworoom", "16 bits, Neural Engine", "exp012-ne-tworoom", "fp32"))
HUIT = (("pusht", "RTN 8 bits", "exp003", "all-8"), ("tworoom", "RTN 8 bits", "exp006", "all-8"))
DOSSIERS_SECONDAIRES = {"pusht": ("exp003", "exp004", "exp009-pusht", "exp011-pusht", "exp014-int8-pusht"),
                        "tworoom": ("exp006", "exp008", "exp009-tworoom", "exp011-tworoom", "exp014-int8-tworoom",
                                    "exp014-int4c-tworoom", "exp014-pal4-tworoom")}
PAIRES = (("RTN 4 bits", "RTN 3 bits"), ("RTN 3 bits", "GPTQ 3 bits"), ("RTN 4 bits", "int4 par canal, Neural Engine"),
          ("int4 par canal, Neural Engine", "palettes, Neural Engine"))
FICHIERS_ANALYSE = ("analyse_exp016.py", "analyse_exp003.py", "analyse_exp006.py", "PREREG-EXP016.md")
ICI = Path(__file__).resolve().parent


def git(*args):
    return subprocess.run(["git", "-C", str(ICI), *args], capture_output=True, text=True)


def verifier_code(sha):
    A.exiger(git("cat-file", "-e", f"{sha}^{{commit}}").returncode == 0, f"commit {sha[:7]} inconnu du dépôt")
    A.exiger(git("diff", "--quiet", sha, "--", *FICHIERS_ANALYSE).returncode == 0, f"code d'analyse différent du commit {sha[:7]}")


def resultats_par_graine(r, tache, d, v, graines):
    """Succès (graines, épisodes), fichiers contrôlés et appariés à la référence de la tâche ; runs Core ML : moteur
    coreml-ne et un seul modèle par dossier ; runs PyTorch : pas de moteur."""
    ref = dict(np.load(r / REF[tache] / "fp32_s42.npz"))
    controler = A6.controler if tache == "tworoom" else (lambda x, nom: A.controler_course(x, nom))
    X, modeles = [], set()
    for g in graines:
        f = r / d / f"{v}_s{g}.npz"
        A.exiger(f.exists(), f"{d}/{f.name} manquant")
        x = dict(np.load(f))
        A.exiger(int(x["seed"]) == g and str(x["spec"]).replace(":", "-") == v, f"{d}/{f.name} : version ou graine")
        A.exiger(not bool(x["git_sale"]), f"{d}/{f.name} : arbre git modifié")
        for k in ("episodes_idx", "start_steps", "init_state", "goal_state"):
            A.exiger(np.array_equal(x[k], ref[k]), f"{d}/{f.name} : {k} différent de la référence")
        if d.startswith(("exp012-ne", "exp014-")):
            A.exiger(str(x.get("moteur", "")) == "coreml-ne", f"{d}/{f.name} : moteur {x.get('moteur')}")
            modeles.add(str(x["coreml_sha256"]))
        else:
            A.exiger("moteur" not in x, f"{d}/{f.name} : moteur {x.get('moteur')} pour un run PyTorch")
        controler(x, f"{d}/{f.name}")
        X.append(x["success"].astype(np.int64))
    A.exiger(len(modeles) <= 1, f"{d} : modèle Core ML différent selon les graines")
    return np.stack(X)


def npmle(k, n, grille=GRILLE, iterations=5000):
    """Loi a priori non paramétrique : poids sur la grille maximisant la vraisemblance binomiale des comptes k (EM).
    Les épisodes de même compte ont la même vraisemblance : l'EM se fait sur les n + 1 comptes, pondérés."""
    c = np.bincount(np.asarray(k), minlength=n + 1) / len(k)
    L = np.stack([math.comb(n, j) * grille ** j * (1 - grille) ** (n - j) for j in range(n + 1)])   # (comptes, grille)
    w = np.full(len(grille), 1.0 / len(grille))
    for _ in range(iterations):
        m = L * w
        w = (c[:, None] * m / np.maximum(m.sum(axis=1, keepdims=True), 1e-300)).sum(axis=0)
    return w


def posterieure(w, k, n=5, grille=GRILLE):
    """Loi a posteriori de p sur la grille, pour un épisode avec k réussites sur n."""
    v = w * grille ** k * (1 - grille) ** (n - k)
    return v / v.sum()


def tirer(rng, poids, forme, grille=GRILLE):
    """Tirages de p sur la grille selon les poids (inversion de la fonction de répartition)."""
    return grille[np.minimum(np.searchsorted(np.cumsum(poids), rng.random(forme)), len(grille) - 1)]


def amincir(p, cible):
    """q = min(1, r p), r par tirage (ligne) pour que la moyenne de q égale cible : forme exacte si rien n'est plafonné,
    dichotomie sinon (version meilleure que la référence)."""
    r = cible / np.maximum(p.mean(axis=1), 1e-12)
    plafonne = (r * p.max(axis=1)) > 1
    if plafonne.any():
        bas, haut = np.zeros(plafonne.sum()), np.full(plafonne.sum(), 1e6)
        pp = p[plafonne]
        for _ in range(80):
            mil = (bas + haut) / 2
            trop = np.minimum(1.0, mil[:, None] * pp).mean(axis=1) > cible
            haut, bas = np.where(trop, mil, haut), np.where(trop, bas, mil)
        r[plafonne] = (bas + haut) / 2
    return np.minimum(1.0, r[:, None] * p)


def test_strate(kR, kV, w, b=B, graine=GRAINE, garder=False):
    """Test principal : strate k_R = 5, T1 = épisodes jamais réussis par la version, nul de perte proportionnelle."""
    s = kR == 5
    ns, obs = int(s.sum()), int((kV[s] == 0).sum())
    rng = np.random.default_rng(graine)
    p = tirer(rng, posterieure(w, 5), (b, ns))
    q = amincir(p, kV[s].mean() / 5)
    k = rng.binomial(5, q).astype(np.int8)
    sim = (k == 0).sum(axis=1)
    diff = obs - sim
    res = {"strate": ns, "T1": obs, "reussite_strate": float(kV[s].mean() / 5), "T1_attendu": float(sim.mean()),
           "exces": float(obs - sim.mean()),
           # intervalle de fluctuation de T1 - T1 simulé sous le nul (pas un intervalle de confiance du nombre d'épisodes
           # cassés ; l'excès est biaisé vers le bas : r est réglé sur une moyenne qui contient les épisodes cassés)
           "exces_fluctuation_nul": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))], "_sim": sim,
           "p": float((1 + (sim >= obs).sum()) / (b + 1)),
           "comptes_strate": {"observe": [int((kV[s] == j).sum()) for j in range(6)],
                              "attendu": [float((k == j).sum(axis=1).mean()) for j in range(6)]}}
    if not garder:
        res.pop("_sim")
        return res
    return res, k


def calibration_synthetique(kR, X42, rho, w, graine, n_syn=N_SYN, b=B_SYN):
    """Versions synthétiques : chaque réussite de la pleine précision 42 à 46 gardée avec la probabilité rho ; taux de
    rejet du test exact aux seuils 5 % et 5 % / 7, T1 synthétique moyen contre attendu."""
    rng = np.random.default_rng(graine)
    ps, t1, att = [], [], []
    for i in range(n_syn):
        kS = (X42 * (rng.random(X42.shape) < rho)).sum(axis=0)
        r_ = test_strate(kR, kS, w, b=b, graine=graine * 10_000 + i)
        ps.append(r_["p"])
        t1.append(r_["T1"])
        att.append(r_["T1_attendu"])
    ps = np.array(ps)
    taux = {"0.05": float((ps < ALPHA).mean()), "0.05/7": float((ps < ALPHA / 7).mean())}
    return {"rho": float(rho), "taux_de_rejet": taux, "T1_synthetique_moyen": float(np.mean(t1)), "T1_attendu_moyen": float(np.mean(att)),
            "acceptee": all(taux[k] <= SEUILS_SYN[k] for k in SEUILS_SYN)}


def e2_global(kR, kV, w, b=B, graine=GRAINE + 2):
    """Épisodes jamais réussis par la version, par valeur de k_R : observés et attendus sous une perte proportionnelle
    globale (r réglé sur les 500 épisodes, loi a posteriori de chaque épisode)."""
    rng = np.random.default_rng(graine)
    p = np.empty((b, len(kR)))
    for j in range(6):
        m = kR == j
        if m.any():
            p[:, m] = tirer(rng, posterieure(w, j), (b, int(m.sum())))
    k = rng.binomial(5, amincir(p, kV.mean() / 5))
    obs = [int(((kR == j) & (kV == 0)).sum()) for j in range(6)]
    att = [float(((kR == j)[None] & (k == 0)).sum(axis=1).mean()) for j in range(6)]
    return {"observe": obs, "attendu": att, "exces_nouveaux": obs[5] - att[5], "exces_amplifies": sum(obs[1:5]) - sum(att[1:5])}


def betabinomiale(k, n):
    """Ajustement bêta-binomial, pour comparaison avec la loi non paramétrique (khi-deux d'adéquation)."""
    k = np.asarray(k)
    nll = lambda t: -np.sum(betaln(k + np.exp(t[0]), n - k + np.exp(t[1])) - betaln(np.exp(t[0]), np.exp(t[1])))
    res = minimize(nll, x0=np.log([0.5, 0.5]), method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 4000})
    a, b_ = np.exp(res.x)
    att = np.array([len(k) * math.comb(n, j) * np.exp(betaln(j + a, n - j + b_) - betaln(a, b_)) for j in range(n + 1)])
    obs = np.array([(k == j).sum() for j in range(n + 1)])
    x2 = float(((obs - att) ** 2 / np.maximum(att, 1e-12)).sum())
    return {"alpha": float(a), "beta": float(b_), "observe": obs.tolist(), "attendu": att.tolist(), "khi2": x2,
            "p_adequation": float(chi2.sf(x2, n + 1 - 1 - 2)), "converge": bool(res.success)}


def secondaires(r, exclus):
    """Toutes les versions à 5 graines (42 à 46) des dossiers listés, hors famille principale, contrôles et fp32."""
    out = []
    for t, dossiers in DOSSIERS_SECONDAIRES.items():
        for d in dossiers:
            specs = sorted({f.name.rsplit("_s", 1)[0] for f in (r / d).glob("*_s4[2-6].npz")})
            for v in specs:
                if (d, v) in exclus or (v == "fp32" and not d.startswith("exp014-")):
                    continue
                if all((r / d / f"{v}_s{g}.npz").exists() for g in G_V):
                    out.append((t, f"{d}/{v}", d, v))
    return out


def analyser(racine, verifier=True, n_syn=N_SYN):
    r = Path(racine)
    kR, X42, w, ajust = {}, {}, {}, {}
    for t, d in REF.items():
        kR[t] = resultats_par_graine(r, t, d, "fp32", G_R).sum(axis=0)
        X42[t] = resultats_par_graine(r, t, d, "fp32", G_V)
        w[t] = npmle(kR[t] + X42[t].sum(axis=0), 10)
        ajust[t] = {"poids_grille_non_nuls": {f"{GRILLE[i]:.3f}": float(w[t][i]) for i in np.flatnonzero(w[t] > 1e-3)},
                    "betabinomiale_kR": betabinomiale(kR[t], 5), "betabinomiale_10": betabinomiale(kR[t] + X42[t].sum(axis=0), 10)}
    out = {"loi_a_priori": ajust, "principales": {}, "controles_negatifs": {}, "controle_moteur": {}, "huit_bits": {}, "secondaires": {}}
    # contrôles négatifs bloquants : la pleine précision 42 à 46 traitée comme une version (Holm sur 2)
    for t in REF:
        out["controles_negatifs"][t] = test_strate(kR[t], X42[t].sum(axis=0), w[t])
    qn = A.holm([out["controles_negatifs"][t]["p"] for t in REF])
    neg_ok = {t: bool(q >= ALPHA) for t, q in zip(REF, qn)}
    for t, q in zip(REF, qn):
        out["controles_negatifs"][t]["p_holm"] = float(q)
    sims16 = {}
    for groupe, liste in (("controle_moteur", MOTEUR), ("huit_bits", HUIT)):
        for j, (t, nom, d, v) in enumerate(liste):
            kV = resultats_par_graine(r, t, d, v, G_V).sum(axis=0)
            res_c, _ = test_strate(kR[t], kV, w[t], graine=GRAINE + 50 + j + (10 if groupe == "huit_bits" else 0), garder=True)
            if groupe == "controle_moteur":
                sims16[t] = res_c["_sim"]
            res_c.pop("_sim")
            out[groupe][f"{t} | {nom}"] = res_c
    huit_alerte = {t: any(x["p"] < ALPHA and x["exces"] >= EXCES_MIN for k, x in out["huit_bits"].items() if k.startswith(t)) for t in REF}

    # famille principale
    nulles, K = {}, {}
    for i, (t, nom, d, v) in enumerate(PRINCIPALES):
        XV = resultats_par_graine(r, t, d, v, G_V)
        kV = XV.sum(axis=0)
        K[(t, nom)] = kV
        res, k = test_strate(kR[t], kV, w[t], graine=GRAINE + i, garder=True)
        nulles[(t, nom)] = k
        rho = XV.mean() / X42[t].mean()
        res["calibration_synthetique"] = calibration_synthetique(kR[t], X42[t], rho, w[t], graine=GRAINE + 100 + i, n_syn=n_syn)
        res["E2"] = e2_global(kR[t], kV, w[t])
        best, moy = (X42[t].sum(axis=0) > 0).mean() - (kV > 0).mean(), X42[t].mean() - XV.mean()
        res["E3"] = {"au_moins_une_version": float((kV > 0).mean()), "au_moins_une_pleine_precision": float((X42[t].sum(axis=0) > 0).mean()),
                     "perte_moyenne": float(moy), "perte_au_meilleur_des_5": float(best), "rapport": float(best / moy) if moy > 0 else None}
        out["principales"][f"{t} | {nom}"] = {"dossier": d, "fichier": v, **res}
    noms = list(out["principales"])
    for n, q in zip(noms, A.holm([out["principales"][n]["p"] for n in noms])):
        x = out["principales"][n]
        t = n.split(" | ")[0]
        x["p_holm"] = float(q)
        if not neg_ok[t] or not x["calibration_synthetique"]["acceptee"]:
            x["verdict"] = "non concluant"
        elif q < ALPHA and x["exces"] >= EXCES_MIN:
            x["verdict"] = "soutenue"
        elif q < ALPHA:
            x["verdict"] = "excès sous le minimum"
        else:
            x["verdict"] = "non soutenue"
        if "Neural Engine" in n:                       # écart à l'excès du 16 bits sur le Neural Engine (P1-3)
            c16 = out["controle_moteur"][f"{t} | 16 bits, Neural Engine"]
            d = (x["T1"] - x["_sim"]) - (c16["T1"] - sims16[t])
            x["exces_moins_16_bits"] = float(x["exces"] - c16["exces"])
            x["exces_moins_16_bits_fluctuation"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
            x["lecture_moteur"] = ("codage" if x["exces_moins_16_bits"] >= EXCES_MIN and x["exces_moins_16_bits_fluctuation"][0] > 0
                                   else "codage et moteur, ou non séparés")
        else:
            x["lecture_moteur"] = None
        x.pop("_sim")
    out["controles_negatifs_acceptes"] = neg_ok
    out["alerte_huit_bits"] = huit_alerte              # RTN 8 bits en excès : les versions PyTorch de la tâche « à confirmer »
    out["E2_repere_pleine_precision"] = {t: e2_global(kR[t], X42[t].sum(axis=0), w[t]) for t in REF}
    # E4 : recouvrement des épisodes de la strate jamais réussis, contre le nul : p partagé (même épisode, même
    # difficulté), binomiales indépendantes des deux versions (relecture de la version 2, P1-1)
    e4 = {}
    s = kR["pusht"] == 5
    for i, (na, nb) in enumerate(PAIRES):
        a_, b_ = K[("pusht", na)][s] == 0, K[("pusht", nb)][s] == 0
        u = int((a_ | b_).sum())
        g0 = np.random.default_rng([GRAINE + 3, i])
        pn = tirer(g0, posterieure(w["pusht"], 5), (B, int(s.sum())))
        ka = np.random.default_rng([GRAINE + 4, i]).binomial(5, amincir(pn, K[("pusht", na)][s].mean() / 5)) == 0
        kb = np.random.default_rng([GRAINE + 5, i]).binomial(5, amincir(pn, K[("pusht", nb)][s].mean() / 5)) == 0
        un = (ka | kb).sum(axis=1)
        jn = np.where(un > 0, (ka & kb).sum(axis=1) / np.maximum(un, 1), 0.0)
        e4[f"{na} et {nb}"] = {"jaccard": float((a_ & b_).sum() / u) if u else None, "jaccard_nul": float(jn.mean()),
                              "communs": int((a_ & b_).sum()), "tailles": [int(a_.sum()), int(b_.sum())]}
    out["E4_recouvrement_pusht"] = e4
    exclus = {(d, v) for _, _, d, v in PRINCIPALES + MOTEUR + HUIT}
    for t, nom, d, v in secondaires(r, exclus):
        kV = resultats_par_graine(r, t, d, v, G_V).sum(axis=0)
        out["secondaires"][f"{t} | {nom}"] = {"reussite": float(kV.mean() / 5), **test_strate(kR[t], kV, w[t])}
    sha = git("rev-parse", "HEAD").stdout.strip()
    out["commit_analyse"] = sha
    if verifier:
        A.exiger(git("status", "--porcelain", "--untracked-files=no").stdout.strip() == "", "arbre git modifié")
        verifier_code(sha)
    return out


def rapport(s):
    L = ["# EXP-016 : rapport de l'analyse pré-enregistrée (version 2 ; épisodes cassés par la compression)", "",
         f"Commit de l'analyse : `{s['commit_analyse'][:7]}`. Contrôles négatifs acceptés : "
         + ", ".join(f"{t} {'oui' if v else 'non'}" for t, v in s["controles_negatifs_acceptes"].items()) + ".", "",
         "Lecture : un rejet dit « plus d'épisodes de la strate jamais réussis qu'une perte proportionnelle ne le prévoit » ; il ne "
         "sépare pas des épisodes précis d'une perte concentrée sur des épisodes fragiles ; les verdicts ne se comparent pas entre "
         "versions de pertes différentes. Excès biaisé vers le bas ; crochets : fluctuation sous le nul.", "",
         "| Tâche, version | Strate (k_R = 5) | Réussite dans la strate | Jamais réussis (T1) | Attendu | Excès [fluctuation] | p | p Holm | "
         "Calibration synthétique (rejets à 5 % ; 5 %/7) | Verdict |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for n, v in s["principales"].items():
        c = v["calibration_synthetique"]
        L.append(f"| {n} | {v['strate']} | {100 * v['reussite_strate']:.1f} % | {v['T1']} | {v['T1_attendu']:.2f} | "
                 f"{v['exces']:+.1f} [{v['exces_fluctuation_nul'][0]:+.0f} ; {v['exces_fluctuation_nul'][1]:+.0f}] | {v['p']:.4f} | {v['p_holm']:.4f} | "
                 f"{100 * c['taux_de_rejet']['0.05']:.1f} % ; {100 * c['taux_de_rejet']['0.05/7']:.1f} % "
                 f"({'acceptée' if c['acceptee'] else 'refusée'}) | {v['verdict']}{' ; ' + v['lecture_moteur'] + ' (excès moins 16 bits ' + format(v['exces_moins_16_bits'], '+.1f') + ')' if v['lecture_moteur'] else ''} |")
    L += ["", "Contrôles :", ""]
    for g in ("controles_negatifs", "controle_moteur", "huit_bits"):
        for n, v in s[g].items():
            L.append(f"- {g} {n} : T1 {v['T1']}, attendu {v['T1_attendu']:.2f}, excès {v['exces']:+.1f}, p {v['p']:.4f} ; "
                     f"k_V dans la strate observé {v['comptes_strate']['observe']}, attendu {[round(x, 1) for x in v['comptes_strate']['attendu']]}")
    L += ["", "E2 (jamais réussis par valeur de k_R, observé contre attendu sous une perte proportionnelle globale) et E3 :", ""]
    for n, v in s["principales"].items():
        e2, e3 = v["E2"], v["E3"]
        L.append(f"- {n} : observé {e2['observe']}, attendu {[round(x, 1) for x in e2['attendu']]} ; excès à k_R = 5 "
                 f"{e2['exces_nouveaux']:+.1f}, à k_R de 1 à 4 {e2['exces_amplifies']:+.1f} ; perte moyenne {100 * e3['perte_moyenne']:.1f} "
                 f"points, au meilleur des 5 {100 * e3['perte_au_meilleur_des_5']:.1f}")
    for t, e2 in s["E2_repere_pleine_precision"].items():
        L.append(f"- repère {t}, pleine précision 42 à 46 : observé {e2['observe']}, attendu {[round(x, 1) for x in e2['attendu']]}")
    L += ["", "E4 (PushT, strate) :", ""] + [
        f"- {k} : Jaccard {'' if v['jaccard'] is None else round(v['jaccard'], 3)} (nul {v['jaccard_nul']:.3f}), {v['communs']} communs, "
        f"tailles {v['tailles']}" for k, v in s["E4_recouvrement_pusht"].items()]
    L += ["", "Secondaires :", ""] + [
        f"- {n} : réussite {100 * v['reussite']:.1f} %, T1 {v['T1']} sur {v['strate']}, attendu {v['T1_attendu']:.2f}, excès "
        f"{v['exces']:+.1f}, p {v['p']:.4f}" for n, v in s["secondaires"].items()]
    L += ["", "Loi a priori (grille, EM sur 10 graines) et bêta-binomiale pour comparaison :", ""]
    for t, v in s["loi_a_priori"].items():
        bb = v["betabinomiale_10"]
        L.append(f"- {t} : poids non nuls {v['poids_grille_non_nuls']} ; bêta-binomiale sur 10 essais : alpha {bb['alpha']:.3f}, "
                 f"beta {bb['beta']:.3f}, p d'adéquation {bb['p_adequation']:.4f}")
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
