"""EXP-003 : analyse pré-enregistrée (amendement 2 de PREREG.md).

Lit les 55 fichiers de la grille, passe les contrôles bloquants, puis calcule, dans l'ordre de
l'amendement 2 : P1, P2, P3' (inversion de signe, Holm) ; les descriptifs par version ; la
décomposition SS/SF/FS/FF ; les classes de courses ; le plancher de bruit ; les protocoles simulés
des études existantes ; la part du planificateur dans la variance ; les épisodes cassés. Écrit
`resultats.json` et `rapport.md` dans --sortie.

Aucune constante ni aucun calcul de ce fichier ne change après la première lecture des fichiers de la
grille : toute analyse décidée ensuite est exploratoire et vit dans un autre script.

Usage :
    python analyse_exp003.py --grille ../resultats/exp003 --sortie ../resultats/analyse-exp003
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy import stats

from metriques import ecart_angle, marges

ICI = Path(__file__).resolve().parent
VERSIONS = ("fp32", "all:8", "all:6", "all:5", "all:4", "all:3",
            "encoder:4", "encoder:3", "predictor:4", "predictor:3")
QUANTIFIEES = VERSIONS[1:]
GA = (42, 43, 44, 45, 46)          # graines de la grille, les mêmes pour toutes les versions
GB = (47, 48, 49, 50, 51)          # réplique de la pleine précision, jamais dans P1, P2, P3'
N, T = 500, 50
COMMIT_CODE = "c5703db"
FICHIERS_CODE = ("run_grid.py", "quant.py", "lancer_grille.sh")
B = 10_000                         # vecteurs d'inversion de signe, default_rng(0)
B_BOOT = 10_000                    # rééchantillons bootstrap, default_rng(2) (strate : default_rng(3))
N_SIM = 1_000                      # tirages des protocoles simulés, default_rng(1)
REGLAGES_SIM = ((50, 1), (500, 1), (10, 3), (50, 5))   # (épisodes, graines), amendement 3
ALPHA = 0.05
MARGE_TOST = 0.02                  # ± 2 points de réussite, un épisode sur 50 du protocole officiel
SEUIL_CASSE = 3                    # baisse d'au moins 3 succès sur 5
TOL_ARRONDI = 1e-4                 # état enregistré en float32 : |M2 - 1| sous ce seuil n'est pas bloquant
PLANCHER_LOG = 0.05                # log(max(M, 0,05)) : 1 px ou 1 degré, sous l'arrondi physique utile
COUCHES = {"fp32": 0, "all": 109, "encoder": 73, "predictor": 30}
LIM = np.pi / 9
CLASSES = ("reussite", "T place sans reussite", "T jamais place")


class ControleEchoue(RuntimeError):
    pass


def exiger(cond, message):
    if not cond:
        raise ControleEchoue(message)


# ---------------------------------------------------------------- chargement et contrôles bloquants

def git(*args):
    return subprocess.run(["git", "-C", str(ICI), *args], capture_output=True, text=True).returncode


def verifier_code(sha):
    """Le commit `sha` existe, descend de c5703db et a run_grid.py, quant.py, lancer_grille.sh identiques."""
    exiger(git("cat-file", "-e", f"{sha}^{{commit}}") == 0, f"commit {sha[:7]} inconnu du dépôt")
    exiger(git("merge-base", "--is-ancestor", COMMIT_CODE, sha) == 0, f"le commit {sha[:7]} ne descend pas de {COMMIT_CODE}")
    exiger(git("diff", "--quiet", COMMIT_CODE, sha, "--", *FICHIERS_CODE) == 0,
           f"le commit {sha[:7]} n'a pas le code de {COMMIT_CODE}")


def lg(M):
    """Log des marges avec plancher fixe (amendement 3) : M3 vaut exactement 0 quand le T part au but."""
    return np.log(np.maximum(M, PLANCHER_LOG))


def charger(dossier):
    """Charge les 55 fichiers et vérifie qu'ils décrivent tous le même protocole et le même code."""
    dossier = Path(dossier)
    proto = json.loads((dossier / "meta.json").read_text())["protocole"]
    exiger(proto["n"] == N, f"meta.json : {proto['n']} épisodes au lieu de {N}")
    ep_ref, st_ref = np.asarray(proto["episodes_idx"]), np.asarray(proto["start_steps"])
    res = {}
    for f in sorted(dossier.glob("*.npz")):
        if ".tmp." in f.name:
            continue
        d = dict(np.load(f))
        cle = (str(d["spec"]), int(d["seed"]))
        exiger(cle not in res, f"{f.name} : doublon de {cle}")
        res[cle] = d
    attendus = {(v, g) for v in VERSIONS for g in GA} | {("fp32", g) for g in GB}
    exiger(set(res) == attendus, f"fichiers manquants {sorted(attendus - set(res))}, "
                                 f"en trop {sorted(set(res) - attendus)}")
    ref = res[("fp32", GA[0])]
    shas = set()
    for (v, g), d in res.items():
        nom = f"{v} graine {g}"
        exiger(not bool(d["git_sale"]), f"{nom} : arbre git modifié au lancement")
        shas.add(str(d["git"]))
        exiger(np.array_equal(d["positions"], np.arange(N)), f"{nom} : positions")
        exiger(np.array_equal(d["episodes_idx"], ep_ref) and np.array_equal(d["start_steps"], st_ref),
               f"{nom} : épisodes différents de meta.json")
        exiger(int(d["chunk"]) == proto["chunk"], f"{nom} : taille des paquets")
        exiger(int(d["n_couches"]) == COUCHES[v.split(":")[0]], f"{nom} : {int(d['n_couches'])} couches quantifiées")
        # un paquet s'arrête avant 50 pas si tous ses épisodes ont réussi (mode « wait » de stable-worldmodel)
        ch = int(d["chunk"])
        for c, pas in enumerate(np.asarray(d["n_steps"]).tolist()):
            sl = slice(c * ch, (c + 1) * ch)
            exiger(pas == T or (d["success"][sl].all() and int(d["first_success"][sl].max()) == pas - 1),
                   f"{nom} : paquet {c} arrêté à {pas} pas")
        exiger(d["state"].shape == (T, N, 7), f"{nom} : forme de l'état {d['state'].shape}")
        for k in ("init_state", "goal_state"):
            exiger(np.array_equal(d[k], ref[k]), f"{nom} : {k} différent")
    for sha in sorted(shas):
        verifier_code(sha)
    return res, sorted(shas)


def controler_course(d, nom):
    """Contrôles bloquants d'une (version, graine).

    Renvoie M2, M3, le nombre de cas limites d'arrondi et le nombre de courses divergentes. Une course
    divergente (amendement 3) est un échec dont l'état devient NaN sur ses 7 colonnes à un pas j et le reste
    jusqu'au pas 50 ; ses marges sont prises sur les pas finis, ou sur l'état de départ si j = 0.
    """
    s = d["success"].astype(bool)
    k = d["first_success"].astype(np.int64)
    exiger(np.array_equal(s, k >= 0) and np.all(k < T), f"{nom} : succès et pas du premier succès incohérents")
    etat = d["state"]
    nan_pas = np.isnan(etat).any(axis=2)                          # (T, N)
    j = np.where(nan_pas.any(axis=0), nan_pas.argmax(axis=0), T)  # premier pas avec un NaN
    t = np.arange(T)[:, None]
    suffixe = np.isnan(etat).all(axis=2) == (t >= j[None, :])     # NaN complets exactement à partir de j
    exiger(suffixe.all(), f"{nom} : NaN isolés dans l'état")
    exiger(np.all(j[s] == k[s] + 1), f"{nom} : état gelé ailleurs qu'au pas suivant le succès")
    diverge = ~s & (j < T)
    if (diverge & (j == 0)).any():
        etat = etat.copy()
        etat[0, diverge & (j == 0)] = d["init_state"][diverge & (j == 0)]
    m, mT, _ = marges({"state": etat, "goal_state": d["goal_state"]})
    exiger(np.isfinite(m).all() and np.isfinite(mT).all(), f"{nom} : marge non finie")
    au_but = (d["init_state"][:, 2:5] == d["goal_state"][:, 2:5]).all(axis=1)
    exiger((m > 0).all(), f"{nom} : marge de succès nulle")
    exiger(((mT > 0) | au_but).all(), f"{nom} : marge du T nulle alors que le T ne partait pas au but")
    desaccord = (m < 1) != s
    limites = desaccord & (np.abs(m - 1) <= TOL_ARRONDI)
    exiger(not (desaccord & ~limites).any(),
           f"{nom} : M2 < 1 et succès diffèrent sur {int((desaccord & ~limites).sum())} courses")
    return m, mT, int(limites.sum()), int(diverge.sum())


def tables(res):
    """Par (version, groupe de graines) : succès, K, M2, M3, classes, pas du premier succès, contacts."""
    D, limites, divergences = {}, 0, {}
    for v in VERSIONS:
        for groupe, graines in (("A", GA), ("B", GB)):
            if groupe == "B" and v != "fp32":
                continue
            S, M2, M3, P, C = [], [], [], [], []
            for g in graines:
                d = res[(v, g)]
                m, mT, lim, div = controler_course(d, f"{v} graine {g}")
                limites += lim
                divergences[f"{v} graine {g}"] = div
                S.append(d["success"].astype(np.int64))
                M2.append(m)
                M3.append(mT)
                P.append(d["first_success"].astype(np.int64))
                c = d["contacts"].astype(np.int64)
                C.append(np.where(c >= 0, c, 0).sum(axis=0))
            S, M3 = np.stack(S), np.stack(M3)
            D[(v, groupe)] = {"S": S, "K": S.sum(axis=0), "M2": np.stack(M2), "M3": M3,
                              "classe": np.where(S == 1, 0, np.where(M3 < 1, 1, 2)),
                              "premier": np.stack(P), "contacts": np.stack(C)}
    return D, limites, divergences


def depart(res):
    """État de départ (copie du jeu de données) contre le but : distances, angle, marge du T."""
    d = res[("fp32", GA[0])]
    i, g = d["init_state"].astype(np.float64), d["goal_state"].astype(np.float64)
    dT = np.linalg.norm(i[:, 2:4] - g[:, 2:4], axis=1)
    dA = np.linalg.norm(i[:, :2] - g[:, :2], axis=1)
    dth = ecart_angle(i[:, 4], g[:, 4])
    mT0 = np.maximum(dT / 20, dth / LIM)
    au_but = (d["init_state"][:, 2:5] == d["goal_state"][:, 2:5]).all(axis=1)
    return {"dT0": dT, "dA0": dA, "dth0": dth, "mT0": mT0, "bouge": mT0 >= 1, "au_but": au_but}


# ---------------------------------------------------------------- outils statistiques

class Inversion:
    """Test par inversion de signe de la somme des différences appariées par épisode.

    Les vecteurs de signes sont tirés une fois (default_rng(0)) et servent partout ; un sous-échantillon
    de n épisodes prend les n premières colonnes. Les différences entières restent exactes en float64.
    p = (1 + nombre de sommes permutées au moins aussi extrêmes, égalité comprise) / (B + 1).
    """

    def __init__(self, n=N, b=B, graine=0):
        self.s = np.random.default_rng(graine).choice(np.array([-1.0, 1.0]), size=(b, n))
        self.b = b

    def p(self, x, sens="bilateral"):
        x = np.asarray(x, dtype=np.float64)
        return float(self.p_lot(x[None, :], sens)[0])

    def p_lot(self, X, sens="bilateral"):
        """Une p-valeur par ligne de X (tirages, n)."""
        X = np.asarray(X, dtype=np.float64)
        exiger(np.isfinite(X).all(), "différences non finies : une statistique NaN donnerait p = 1 / (B + 1)")
        t_obs = X.sum(axis=1)[:, None]
        # numpy 2 avec Accelerate (macOS) lève de faux avertissements de dépassement dans matmul ; le produit
        # est vérifié identique au calcul en entiers (test_analyse_exp003.py), et sa finitude est exigée.
        with np.errstate(all="ignore"):
            t = X @ self.s[:, :X.shape[1]].T
        exiger(np.isfinite(t).all(), "sommes permutées non finies")
        if np.array_equal(X, np.round(X)):                     # différences entières : exactitude vérifiée
            Xi = X.astype(np.int64)
            exiger(np.array_equal(t[:, :100], Xi @ self.s[:100, :X.shape[1]].astype(np.int64).T),
                   "produit float64 différent du produit en entiers")
        if sens == "bilateral":
            compte = np.count_nonzero(np.abs(t) >= np.abs(t_obs), axis=1)
        elif sens == "plus":
            compte = np.count_nonzero(t >= t_obs, axis=1)
        elif sens == "moins":
            compte = np.count_nonzero(t <= t_obs, axis=1)
        else:
            raise ValueError(sens)
        return (1 + compte) / (self.b + 1)


class Bootstrap:
    """Intervalle percentile à 95 % de la moyenne sur les épisodes, mêmes rééchantillons partout."""

    def __init__(self, n=N, b=B_BOOT, graine=2):
        self.idx = np.random.default_rng(graine).integers(0, n, size=(b, n))

    def ic(self, x):
        x = np.asarray(x, dtype=np.float64)
        moy = x[self.idx].mean(axis=1)
        return {"moyenne": float(x.mean()), "bas": float(np.percentile(moy, 2.5)),
                "haut": float(np.percentile(moy, 97.5))}


def holm(p):
    p = np.asarray(p, dtype=np.float64)
    m = len(p)
    adj = np.empty(m)
    courant = 0.0
    for rang, i in enumerate(np.argsort(p, kind="stable")):
        courant = max(courant, min(1.0, (m - rang) * p[i]))
        adj[i] = courant
    return adj


def mcnemar_exact(R, A):
    """McNemar exact bilatéral par ligne ; R et A binaires de forme (tirages, n). 0 discordant : p = 1."""
    b = ((R == 1) & (A == 0)).sum(axis=1)
    c = ((R == 0) & (A == 1)).sum(axis=1)
    return np.minimum(1.0, 2 * stats.binom.cdf(np.minimum(b, c), b + c, 0.5))


def tost_t(d, marge=MARGE_TOST):
    """Contrôle de P3'(a) : TOST t apparié (Schuirmann) et intervalle à 1 - 2 alpha."""
    n = len(d)
    moy, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
    if se == 0:
        p_bas = p_haut = 0.0 if abs(moy) < marge else 1.0
        demi = 0.0
    else:
        p_bas = float(stats.t.sf((moy + marge) / se, n - 1))
        p_haut = float(stats.t.cdf((moy - marge) / se, n - 1))
        demi = float(stats.t.ppf(1 - ALPHA, n - 1) * se)
    return {"p_bas": p_bas, "p_haut": p_haut, "p": max(p_bas, p_haut),
            "ic90": [float(moy - demi), float(moy + demi)]}


def verdict(p_adj, somme, sens_attendu):
    if p_adj >= ALPHA:
        return "non soutenue"
    return "soutenue" if np.sign(somme) == sens_attendu else "contredite (sens contraire)"


# ---------------------------------------------------------------- analyses de l'amendement 2

def tests_preenregistres(D, dep, inv):
    """Points 4 à 6 : P1, P2, P3' et la correction de Holm ; P3'(b) dans la strate « bouge » (amendement 3)."""
    f = D[("fp32", "A")]
    x1 = D[("all:4", "A")]["K"] - f["K"]
    x2 = D[("predictor:4", "A")]["K"] - D[("encoder:4", "A")]["K"]
    xa = D[("all:5", "A")]["K"] - f["K"]
    w = (lg(D[("all:5", "A")]["M3"]).mean(axis=0) - lg(f["M3"]).mean(axis=0))[dep["bouge"]]
    p1 = inv.p(x1, "bilateral")
    p2 = inv.p(x2, "bilateral")
    p_bas = inv.p(10 * xa + 1, "plus")       # H0 : delta <= -0,02 ; 10 x + 1 = 50 (x / 5 + 0,02)
    p_haut = inv.p(10 * xa - 1, "moins")     # H0 : delta >= +0,02
    p_b = inv.p(w, "plus")                    # H0 : log M3 non dégradée
    p3 = max(p_bas, p_haut, p_b)
    adj = holm([p1, p2, p3])
    return {
        "P1": {"statistique": "somme K(all:4) - K(fp32)", "somme": int(x1.sum()),
               "delta_reussite": float(x1.mean() / 5), "p": p1, "p_holm": float(adj[0]),
               "verdict": verdict(adj[0], x1.sum(), -1)},
        "P2": {"statistique": "somme K(predictor:4) - K(encoder:4)", "somme": int(x2.sum()),
               "delta_reussite": float(x2.mean() / 5), "p": p2, "p_holm": float(adj[1]),
               "verdict": verdict(adj[1], x2.sum(), +1)},
        "P3'": {"delta_reussite": float(xa.mean() / 5), "p_bas": p_bas, "p_haut": p_haut,
                "delta_log_M3_bouge": float(w.mean()), "episodes_bouge": int(w.size), "p_b": p_b, "p": p3, "p_holm": float(adj[2]),
                "verdict": "établie" if adj[2] < ALPHA else "non établie",
                "controle_tost_t": tost_t(xa / 5)},
    }


def parts(classe):
    c = np.asarray(classe).ravel()
    return {nom: float(np.mean(c == i)) if c.size else None for i, nom in enumerate(CLASSES)}


def descriptifs(D, dep, boot, boot_strate):
    """Point 7 (réussite, marges, intervalles), point 8 (classes), strate « le T doit bouger »."""
    f = D[("fp32", "A")]
    bouge = dep["bouge"]
    out = {}
    for v in VERSIONS + ("fp32 replique",):
        d = D[("fp32", "B")] if v == "fp32 replique" else D[(v, "A")]
        e = {"reussite": float(d["S"].mean()), "reussite_par_graine": d["S"].mean(axis=1).tolist(),
             "classes": parts(d["classe"]), "classes_bouge": parts(d["classe"][:, bouge])}
        if v != "fp32":
            dr = (d["K"] - f["K"]) / 5
            d3 = lg(d["M3"]).mean(axis=0) - lg(f["M3"]).mean(axis=0)
            d2 = lg(d["M2"]).mean(axis=0) - lg(f["M2"]).mean(axis=0)
            e["delta_reussite"] = boot.ic(dr)
            e["delta_log_M3"] = boot.ic(d3)
            e["delta_log_M2"] = boot.ic(d2)
            e["bouge"] = {"delta_reussite": boot_strate.ic(dr[bouge]), "delta_log_M3": boot_strate.ic(d3[bouge])}
        out[v] = e
    return out


def decomposition(D):
    """Point 7 : part de chaque paire de résultats par graine (fp32, version) dans la différence."""
    f = D[("fp32", "A")]
    out = {}
    for v in QUANTIFIEES:
        d = D[(v, "A")]
        out[v] = {}
        for nom, (sf, sv) in (("SS", (1, 1)), ("SF", (1, 0)), ("FS", (0, 1)), ("FF", (0, 0))):
            masque = (f["S"] == sf) & (d["S"] == sv)
            e = {"courses": int(masque.sum())}
            for M in ("M3", "M2"):
                diff = lg(d[M]) - lg(f[M])
                e[f"contribution_log_{M}"] = float(diff[masque].sum() / f["S"].size)
            if nom == "SS":
                e["delta_pas_premier_succes"] = (float((d["premier"] - f["premier"])[masque].mean())
                                                 if masque.any() else None)
            out[v][nom] = e
    return out


def plancher(D, boot):
    """Point 10 : bascules perdues et gagnées par 50 épisodes, entre graines différentes."""
    f = D[("fp32", "A")]["S"]
    paires = [(a, b) for a in range(5) for b in range(5) if a != b]

    def bascules(R, X, pp):
        perdus = np.array([((R[a] == 1) & (X[b] == 0)).sum() for a, b in pp]) * 50 / N
        gagnes = np.array([((R[a] == 0) & (X[b] == 1)).sum() for a, b in pp]) * 50 / N
        return perdus, gagnes

    def resume(x):
        return {"moyenne": float(x.mean()), "min": float(x.min()), "max": float(x.max())}

    pn, gn = bascules(f, f, paires)
    discord = np.mean([f[a] != f[b] for a, b in paires], axis=0) * 50
    fB = D[("fp32", "B")]["S"]
    discord_rep = np.mean([f[a] != fB[b] for a in range(5) for b in range(5)], axis=0) * 50
    out = {"fp32": {"perdus_par_50": resume(pn), "gagnes_par_50": resume(gn),
                    "discordants_par_50": boot.ic(discord),
                    "discordants_par_50_contre_replique": boot.ic(discord_rep)}}
    for v in QUANTIFIEES:
        X = D[(v, "A")]["S"]
        pv, gv = bascules(f, X, paires)
        ps, gs = bascules(f, X, [(a, a) for a in range(5)])
        out[v] = {"perdus_par_50": resume(pv), "gagnes_par_50": resume(gv),
                  "memes_graines": {"perdus_par_50": resume(ps), "gagnes_par_50": resume(gs)},
                  "attribution_au_bruit": float(pn.mean() / pv.mean()) if pv.mean() > 0 else None}
    return out


def protocoles_simules(D, inv, reference):
    """Point 9 (amendement 3) : ce que concluraient des études à peu d'épisodes et peu de graines.

    m = 1 : toutes les graines sont énumérées (5 couples appariés, 25 non appariés) pour chaque tirage
    d'épisodes ; m = 3 : 3 graines sur 5 tirées de chaque côté ; m = 5 : toutes. Test : McNemar exact sur
    les courses (épisode, graine) si m = 1 ou 3, inversion de signe sur K si m = 5.
    """
    rng = np.random.default_rng(1)
    comps = [("fp32", v, True) for v in QUANTIFIEES] + [("encoder:4", "predictor:4", False),
                                                          ("encoder:3", "predictor:3", False)]
    out = {}
    for n, m in REGLAGES_SIM:
        pos = np.empty((N_SIM, n), dtype=np.int64)
        ca = np.zeros((N_SIM, 3), dtype=np.int64)
        cb = np.zeros((N_SIM, 3), dtype=np.int64)
        for t in range(N_SIM):
            pos[t] = rng.choice(N, n, replace=False)
            if m == 3:
                ca[t] = rng.choice(5, 3, replace=False)
                cb[t] = rng.choice(5, 3, replace=False)

        def courses(v, groupe, graines):
            """(tirages, unités) : issues 0/1 si m = 1 ou 3 (unités = courses), K si m = 5 (unités = épisodes)."""
            S = D[(v, groupe)]["S"]
            if m == 5:
                return S[:, pos].sum(axis=0)
            if m == 1:
                return S[graines][pos]
            return S[graines[:, :, None], pos[:, None, :]].reshape(N_SIM, 3 * n)

        def evaluer(R, A, signe):
            delta = (A - R).mean(axis=1)
            p = inv.p_lot(A - R, "bilateral") if m == 5 else mcnemar_exact(R, A)
            sig, sg = p < ALPHA, np.sign(delta)
            e = {"part_significatif": sig.mean(), "part_difference_nulle": np.mean(delta == 0)}
            if signe != 0:
                e.update({"part_meme_signe": np.mean(sg == signe), "part_signe_contraire": np.mean(sg == -signe),
                          "part_conclut_meme_sens": np.mean(sig & (sg == signe)), "part_ne_conclut_pas": np.mean(~sig),
                          "part_conclut_sens_contraire": np.mean(sig & (sg == -signe))})
            return e

        def moyenne(liste):
            return {k: float(np.mean([e[k] for e in liste])) for k in liste[0]}

        def variante(ref, gref, alt, apparie, signe):
            if m == 1:
                couples = [(a, a) for a in range(5)] if apparie else [(a, b) for a in range(5) for b in range(5)]
                return moyenne([evaluer(courses(ref, gref, b), courses(alt, "A", a), signe) for a, b in couples])
            return moyenne([evaluer(courses(ref, gref, ca if apparie else cb), courses(alt, "A", ca), signe)])

        cle = f"n{n}_graines{m}"
        out[cle] = {}
        for ref, alt, a_une_replique in comps:
            r = reference[(ref, alt)]
            var = {"apparie": variante(ref, "A", alt, True, r["signe"])}
            if a_une_replique:
                var["non_apparie"] = variante("fp32", "B", alt, False, r["signe"])
            out[cle][f"{ref} -> {alt}"] = {"reference": r, "variantes": var}
        out[cle]["nul vrai : fp32 replique -> fp32"] = {
            "reference": {"signe": 0, "retenue": False, "signe_stable_sans_une_graine": None},
            "variantes": {"non_apparie": variante("fp32", "B", "fp32", False, 0)}}
    return out


def references(D, desc):
    """Signe du plan complet, intervalle qui exclut 0 (retenue pour le message) et stabilité du signe
    quand on retire une graine (contre la circularité des études simulées à une graine)."""
    ref = {}
    paires = [("fp32", v, desc[v]["delta_reussite"]) for v in QUANTIFIEES] + \
             [(f"encoder:{b}", f"predictor:{b}", desc[f"predictor:{b} - encoder:{b}"]["delta_reussite"]) for b in (4, 3)]
    for a, b, ic in paires:
        d = D[(b, "A")]["S"] - D[(a, "A")]["S"]                 # (5, N)
        signe = int(np.sign(d.sum()))
        stable = all(int(np.sign(np.delete(d, g, axis=0).sum())) == signe for g in range(5))
        ref[(a, b)] = {"signe": signe, "retenue": bool(ic["bas"] > 0 or ic["haut"] < 0),
                       "signe_stable_sans_une_graine": stable}
    return ref


def part_planificateur(D):
    """Point 9 : part de la variance d'une étude à une graine qui vient du planificateur.

    Appariée (mêmes graines, hasard partagé) : borne basse ; non appariée (version contre la réplique,
    graines 42 à 46 contre 47 à 51, rang à rang) : plus proche des études qui ne partagent pas le hasard.
    """
    f, fB = D[("fp32", "A")]["S"], D[("fp32", "B")]["S"]

    def parts_var(dg):
        dg = dg.astype(np.float64)                             # (5, N)
        s2_plan = float(dg.var(axis=0, ddof=1).mean())
        s2_ep = float(dg.mean(axis=0).var(ddof=1) - s2_plan / 5)
        tot = s2_plan + max(s2_ep, 0.0)
        return {"sigma2_plan": s2_plan, "sigma2_ep": s2_ep, "part_planificateur": s2_plan / tot if tot > 0 else None}

    out = {}
    for v in QUANTIFIEES:
        X = D[(v, "A")]["S"]
        out[v] = {"apparie_borne_basse": parts_var(X - f), "non_apparie": parts_var(X - fB)}
    out["fp32 replique"] = {"non_apparie": parts_var(fB - f)}
    return out


def episodes_casses(D, dep, inv):
    """Point 11 : épisodes cassés, mesurés contre la réplique des deux côtés."""
    KB, KAf = D[("fp32", "B")]["K"], D[("fp32", "A")]["K"]
    c0 = (KB - KAf) >= SEUIL_CASSE
    strates = {"replique 5/5": KB == 5, "replique 1 a 4": (KB >= 1) & (KB <= 4), "replique 0/5": KB == 0}
    contacts = D[("fp32", "A")]["contacts"].mean(axis=0)
    S, P = D[("fp32", "A")]["S"], D[("fp32", "A")]["premier"].astype(np.float64)
    premier = np.where(S.sum(axis=0) > 0, np.where(S == 1, P, 0).sum(axis=0) / np.maximum(S.sum(axis=0), 1), np.nan)
    traits = {"dT0": dep["dT0"], "dth0": dep["dth0"], "dA0": dep["dA0"], "mT0": dep["mT0"],
              "contacts_fp32": contacts, "pas_premier_succes_fp32": premier}

    def decrire(masque):
        e = {"episodes": int(masque.sum())}
        for nom, x in traits.items():
            xs = x[masque]
            xs = xs[np.isfinite(xs)]
            e[f"mediane_{nom}"] = float(np.median(xs)) if xs.size else None
        return e

    out = {"nuls": int(c0.sum()), "strates_replique": {s: int(mk.sum()) for s, mk in strates.items()}}
    for v in QUANTIFIEES:
        KAv = D[(v, "A")]["K"]
        cv = (KB - KAv) >= SEUIL_CASSE
        u = cv.astype(np.float64) - c0.astype(np.float64)
        out[v] = {"observes": int(cv.sum()), "nuls": int(c0.sum()), "p_plus": inv.p(u, "plus"),
                  "par_strate": {s: {"observes": int(cv[mk].sum()), "nuls": int(c0[mk].sum())}
                                 for s, mk in strates.items()},
                  "memes_graines_descriptif": int(((KAf - KAv) >= SEUIL_CASSE).sum()),
                  "description": {"casses": decrire(cv),
                                  "robustes": decrire(strates["replique 5/5"] & ~cv),
                                  "fragiles": decrire(strates["replique 1 a 4"])}}
    return out


def analyser(dossier):
    res, shas = charger(dossier)
    D, limites, divergences = tables(res)
    dep = depart(res)
    inv = Inversion()
    boot = Bootstrap()
    exiger(dep["bouge"].sum() > 0, "strate « le T doit bouger » vide")
    boot_strate = Bootstrap(n=int(dep["bouge"].sum()), graine=3)
    desc = descriptifs(D, dep, boot, boot_strate)
    for b in (4, 3):
        x = D[(f"predictor:{b}", "A")]["K"] - D[(f"encoder:{b}", "A")]["K"]
        desc[f"predictor:{b} - encoder:{b}"] = {"delta_reussite": boot.ic(x / 5)}
    return {
        "controles": {"fichiers": len(res), "commits": shas, "cas_limites_arrondi": limites,
                      "episodes_bouge": int(dep["bouge"].sum()),
                      "episodes_T_au_but": int(dep["au_but"].sum()),
                      "courses_divergentes": {k: v for k, v in divergences.items() if v}},
        "tests": tests_preenregistres(D, dep, inv),
        "descriptifs": desc,
        "decomposition": decomposition(D),
        "plancher": plancher(D, boot),
        "protocoles_simules": protocoles_simules(D, inv, references(D, desc)),
        "part_planificateur": part_planificateur(D),
        "episodes_casses": episodes_casses(D, dep, inv),
    }


# ---------------------------------------------------------------- sorties

def ic_txt(e, facteur=100, d=1, signe="+"):
    return (f"{facteur * e['moyenne']:{signe}.{d}f} [{facteur * e['bas']:{signe}.{d}f}, "
            f"{facteur * e['haut']:{signe}.{d}f}]")


def rapport(r):
    L = ["# EXP-003 : analyse pré-enregistrée (amendement 2)", "",
         f"Contrôles bloquants passés : {r['controles']['fichiers']} fichiers, commits "
         f"{', '.join(s[:7] for s in r['controles']['commits'])}, "
         f"{r['controles']['cas_limites_arrondi']} cas limites d'arrondi, "
         f"{r['controles']['episodes_bouge']} épisodes où le T doit bouger, "
         f"{r['controles']['episodes_T_au_but']} où il part au but, courses divergentes : "
         f"{r['controles']['courses_divergentes'] or 'aucune'}.", "",
         "## Tests pré-enregistrés (Holm sur P1, P2, P3')", "",
         "| Test | Effet | p | p Holm | Verdict |", "| --- | --- | --- | --- | --- |"]
    t = r["tests"]
    L.append(f"| P1 : all:4 contre fp32 | {100 * t['P1']['delta_reussite']:+.2f} points | {t['P1']['p']:.4f} | "
             f"{t['P1']['p_holm']:.4f} | {t['P1']['verdict']} |")
    L.append(f"| P2 : predictor:4 contre encoder:4 | {100 * t['P2']['delta_reussite']:+.2f} points | {t['P2']['p']:.4f} | "
             f"{t['P2']['p_holm']:.4f} | {t['P2']['verdict']} |")
    p3 = t["P3'"]
    L.append(f"| P3' : all:5, réussite à ± 2 points et log M3 dégradée | {100 * p3['delta_reussite']:+.2f} points, "
             f"log M3 {p3['delta_log_M3_bouge']:+.3f} ({p3['episodes_bouge']} épisodes où le T doit bouger) | {p3['p']:.4f} (bas {p3['p_bas']:.4f}, haut {p3['p_haut']:.4f}, "
             f"M3 {p3['p_b']:.4f}) | {p3['p_holm']:.4f} | {p3['verdict']} |")
    c = p3["controle_tost_t"]
    L += ["", f"Contrôle de P3'(a) par TOST t : p = {c['p']:.4f}, intervalle à 90 % "
              f"[{100 * c['ic90'][0]:+.2f}, {100 * c['ic90'][1]:+.2f}] points.", "",
          "## Par version (5 graines, 500 épisodes ; écarts à fp32, intervalle bootstrap à 95 %)", "",
          "| Version | Réussite | Écart (points) | Écart log M3 | Écart log M2 | Réussite / T placé seul / T jamais placé |",
          "| --- | --- | --- | --- | --- | --- |"]
    for v, e in r["descriptifs"].items():
        if "reussite" not in e:
            continue
        cl = e["classes"]
        cls = " / ".join(f"{100 * cl[k]:.1f}" for k in CLASSES)
        if v == "fp32":
            L.append(f"| {v} | {100 * e['reussite']:.1f} % | | | | {cls} |")
        else:
            L.append(f"| {v} | {100 * e['reussite']:.1f} % | {ic_txt(e['delta_reussite'])} | "
                     f"{ic_txt(e['delta_log_M3'], 1, 3)} | {ic_txt(e['delta_log_M2'], 1, 3)} | {cls} |")
    for b in (4, 3):
        e = r["descriptifs"][f"predictor:{b} - encoder:{b}"]
        L.append(f"\npredictor:{b} moins encoder:{b} : {ic_txt(e['delta_reussite'])} points.")
    pl = r["plancher"]["fp32"]
    L += ["", "## Plancher de bruit", "",
          f"Entre deux graines de fp32 : {ic_txt(pl['discordants_par_50'], 1, 2, '')} épisodes discordants sur 50 "
          f"(contre la réplique : {ic_txt(pl['discordants_par_50_contre_replique'], 1, 2, '')}).", "",
          "| Version | Perdus par 50 (graines différentes) | Gagnés par 50 | Part des pertes que le bruit seul produit |",
          "| --- | --- | --- | --- |"]
    L.append(f"| fp32 | {pl['perdus_par_50']['moyenne']:.2f} | {pl['gagnes_par_50']['moyenne']:.2f} | |")
    for v in QUANTIFIEES:
        e = r["plancher"][v]
        a = "" if e["attribution_au_bruit"] is None else f"{100 * e['attribution_au_bruit']:.0f} %"
        L.append(f"| {v} | {e['perdus_par_50']['moyenne']:.2f} | {e['gagnes_par_50']['moyenne']:.2f} | {a} |")
    L += ["", "## Protocoles simulés (1 000 tirages d'épisodes ; m = 1 : toutes les graines énumérées)", "",
          "Retenue : l'intervalle du plan complet exclut 0 (seules ces lignes portent le message). Stable : le signe "
          "du plan complet ne change pas quand on retire une graine.", ""]
    for cle, bloc in r["protocoles_simules"].items():
        L += [f"### {cle}", "", "| Comparaison | Retenue | Stable | Variante | Même signe | Nul | Signe contraire | "
              "Conclut même sens | Ne conclut pas | Conclut sens contraire | Significatif |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for comp, x in bloc.items():
            ref = x["reference"]
            ret = "oui" if ref["retenue"] else "non"
            stab = {True: "oui", False: "non", None: ""}[ref["signe_stable_sans_une_graine"]]
            for var, e in x["variantes"].items():
                g = lambda k: f"{100 * e[k]:.1f}" if k in e else ""
                L.append(f"| {comp} | {ret} | {stab} | {var} | {g('part_meme_signe')} | {g('part_difference_nulle')} | "
                         f"{g('part_signe_contraire')} | {g('part_conclut_meme_sens')} | {g('part_ne_conclut_pas')} | "
                         f"{g('part_conclut_sens_contraire')} | {g('part_significatif')} |")
        L.append("")
    L += ["## Part du planificateur dans la variance d'une étude à une graine", "",
          "| Version | Appariée (borne basse) | Non appariée |", "| --- | --- | --- |"]

    def pct(e):
        return "" if e is None or e["part_planificateur"] is None else f"{100 * e['part_planificateur']:.0f} %"

    for v, e in r["part_planificateur"].items():
        L.append(f"| {v} | {pct(e.get('apparie_borne_basse'))} | {pct(e.get('non_apparie'))} |")
    ec = r["episodes_casses"]
    L += ["", f"## Épisodes cassés (baisse d'au moins {SEUIL_CASSE} succès sur 5 contre la réplique)", "",
          f"Nul (fp32 contre sa réplique) : {ec['nuls']} épisodes. Strates de la réplique : "
          + ", ".join(f"{s} {k}" for s, k in ec["strates_replique"].items()) + ".", "",
          "| Version | Observés | Nuls | p (unilatérale) | Réplique 5/5 | Réplique 1 à 4 | Mêmes graines (descriptif) |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for v in QUANTIFIEES:
        e = ec[v]
        s5, s14 = e["par_strate"]["replique 5/5"], e["par_strate"]["replique 1 a 4"]
        L.append(f"| {v} | {e['observes']} | {e['nuls']} | {e['p_plus']:.4f} | {s5['observes']} contre {s5['nuls']} | "
                 f"{s14['observes']} contre {s14['nuls']} | {e['memes_graines_descriptif']} |")
    return "\n".join(L) + "\n"


def pur(x):
    if isinstance(x, dict):
        return {str(k): pur(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [pur(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grille", required=True)
    ap.add_argument("--sortie", required=True)
    args = ap.parse_args()
    r = analyser(args.grille)
    out = Path(args.sortie)
    out.mkdir(parents=True, exist_ok=True)
    (out / "resultats.json").write_text(json.dumps(pur(r), indent=1, ensure_ascii=False))
    (out / "rapport.md").write_text(rapport(r))
    print((out / "rapport.md").read_text())


if __name__ == "__main__":
    main()
