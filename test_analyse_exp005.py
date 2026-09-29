"""Tests de analyse_exp005.py : cas construits à la main (comptages refaits en numpy pur, sans le code testé),
fonctions de décision, grilles simulées de 2 000 épisodes (simulateur d'EXP-003) passées par les vraies
vérifications du code, et refus des fichiers corrompus. À lancer sur un arbre commité.

Usage :
    python test_analyse_exp005.py
"""
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy import stats

import analyse_exp003 as A
import analyse_exp005 as A5
import test_analyse_exp003 as T
from metriques import ecart_angle

SIGNES = np.random.default_rng(0).choice(np.array([-1.0, 1.0]), size=(10_000, A5.N5))   # refaits ici, à part
TETE = subprocess.run(["git", "-C", str(A.ICI), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
PROPRE = subprocess.run(["git", "-C", str(A.ICI), "status", "--porcelain", "--untracked-files=no"],
                        capture_output=True, text=True).stdout.strip() == ""


def p_numpy(x, sens, n=None):
    """Comptage d'inversion de signe écrit indépendamment d'analyse_exp003.Inversion."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x) if n is None else n
    with np.errstate(all="ignore"):                          # faux avertissements de numpy avec Accelerate
        t, obs = SIGNES[:, :n] @ x, x.sum()
    compte = {"plus": np.sum(t >= obs), "moins": np.sum(t <= obs), "bilateral": np.sum(np.abs(t) >= abs(obs))}[sens]
    return (1 + int(compte)) / 10_001


def pct(x, idx, q):
    return float(np.percentile(np.asarray(x, float)[idx].mean(axis=1), q))


# ---------------------------------------------------------------- statistique et décisions, sans simulateur

def test_statistique_a_la_main():
    inv = A.Inversion(n=A5.N5)
    w0 = np.zeros(1333)
    x = np.zeros(A5.N5, dtype=int); x[:250] = 1                  # +2,5 points : seule la borne haute échoue
    s = A5.stat_h6(x, w0 + 0.01, inv)
    assert s["p_haut"] > 0.05 and s["p"] == s["p_haut"] == p_numpy(10 * x - 1, "moins") and s["p_bas"] == p_numpy(10 * x + 1, "plus")
    x = np.zeros(A5.N5, dtype=int); x[:150] = -1                 # -1,5 point : dans la marge
    s = A5.stat_h6(x, w0 + 0.01, inv)
    assert s["p_bas"] == p_numpy(10 * x + 1, "plus") < 0.05 and s["p_haut"] == p_numpy(10 * x - 1, "moins")
    x = np.zeros(A5.N5, dtype=int); x[:250] = -1                 # -2,5 points : hors de la marge
    s = A5.stat_h6(x, w0 + 0.01, inv)
    assert s["p_bas"] > 0.05 and s["p"] == s["p_bas"] == p_numpy(10 * x + 1, "plus")
    rng = np.random.default_rng(99)
    w = rng.normal(0, 0.258, 1333); w = w - w.mean() + 2.1 * 0.258 / np.sqrt(1333)
    s = A5.stat_h6(np.zeros(A5.N5, dtype=int), w, inv)           # petit effet : unilatéral vers 0,02
    assert s["p_b"] == p_numpy(w, "plus", 1333) and 0.005 < s["p_b"] < 0.04 and p_numpy(w, "bilateral", 1333) > 1.5 * s["p_b"]
    assert A5.verdict_h6(0.0499) == "établie" and A5.verdict_h6(0.05) == A5.verdict_h6(0.0501) == "non établie"
    print("ok statistique : bornes du TOST, p_b unilatérale, seuil de 5 %, égales au comptage numpy indépendant")


def test_formulation():
    ic = lambda b, h: {"moyenne": (b + h) / 2, "bas": b, "haut": h}
    zero, bas, haut = ic(-0.01, 0.01), ic(-0.02, -0.005), ic(0.003, 0.012)
    F = A5.formulation_h6
    for prog in (-1, 0, 1):
        assert F(False, zero, zero, prog) == "H6 non établie"
    assert F(True, zero, zero, 1).startswith("la pose du T se dégrade à 6 bits alors que la réussite ne bouge pas ;")
    assert F(True, bas, zero, 1).startswith("à 6 bits, la pose du T et la réussite se dégradent toutes deux, la réussite "
                                            "restant à ± 2 points sur l'ensemble des épisodes ;")
    assert F(True, zero, bas, 1).startswith("à 6 bits, la pose du T et la réussite se dégradent toutes deux")
    assert F(True, haut, zero, 1).startswith("la pose du T se dégrade à 6 bits alors que la réussite augmente légèrement, "
                                             "en restant à ± 2 points sur l'ensemble des épisodes ;")
    assert F(True, haut, bas, 1).startswith("la pose du T se dégrade à 6 bits ; l'écart de réussite change de sens")
    assert F(True, zero, zero, 1).endswith("; la dégradation est plus faible à 8 bits (progression selon les bits)")
    assert F(True, zero, zero, 0).endswith("; mais l'effet à 6 bits ne se distingue pas du contrôle à 8 bits")
    assert F(True, zero, zero, -1).endswith("; mais la pose est plus mauvaise à 8 bits qu'à 6 bits (pas de progression "
                                            "selon les bits)")
    assert [A5.signe_ic(e) for e in (zero, bas, haut)] == [0, -1, 1]
    assert A5.signe_ic(ic(0.0, 0.01)) == 0 and A5.signe_ic(ic(-0.01, 0.0)) == 0      # une borne à 0 : contient 0
    print("ok règle de rédaction : les 4 sens de l'écart de réussite et les 3 sens de la progression")


def test_comparer_sur_M3():
    """comparer_version porte bien sur M3 (pas M2) et décompose selon les issues, sur des tableaux construits."""
    rng = np.random.default_rng(5)
    n, bouge = A5.N5, np.zeros(A5.N5, dtype=bool)
    bouge[:1333] = True

    def version(decalage3, decalage2):
        S = (rng.uniform(size=(5, n)) < 0.8).astype(np.int64)
        M3 = np.exp(rng.normal(0, 0.3, (5, n)) + decalage3)
        M2 = np.exp(rng.normal(0, 0.3, (5, n)) + decalage2)
        return {"S": S, "K": S.sum(0), "M3": M3, "M2": M2, "dT_min": rng.uniform(0, 40, (5, n)),
                "dth_min": rng.uniform(0, 30, (5, n)), "classe": np.zeros((5, n), dtype=int)}
    f, q = version(0.0, 0.0), version(0.01, 0.2)
    inv, boot, boots = A.Inversion(n=n), A.Bootstrap(n=n), A.Bootstrap(n=1333, graine=3)
    r = A5.comparer_version(f, q, bouge, inv, boot, boots)
    w3 = (np.log(np.maximum(q["M3"], 0.05)).mean(0) - np.log(np.maximum(f["M3"], 0.05)).mean(0))[bouge]
    w2 = (np.log(np.maximum(q["M2"], 0.05)).mean(0) - np.log(np.maximum(f["M2"], 0.05)).mean(0))[bouge]
    assert r["p_b"] == p_numpy(w3, "plus", 1333) != p_numpy(w2, "plus", 1333)
    assert abs(r["delta_log_M3_bouge"] - w3.mean()) < 1e-12 and r["delta_reussite"] == (q["K"] - f["K"]).mean() / 5
    x = q["K"] - f["K"]
    assert r["ecarts"]["reussite"]["bas"] == pct(x / 5, boot.idx, 2.5)
    assert r["ecarts"]["reussite_bouge"]["haut"] == pct((x / 5)[bouge], boots.idx, 97.5)
    d2 = np.log(np.maximum(q["M2"], 0.05)).mean(0) - np.log(np.maximum(f["M2"], 0.05)).mean(0)
    assert abs(r["ecarts"]["log_M2"]["moyenne"] - d2.mean()) < 1e-12 and r["ecarts"]["log_M2"]["bas"] == pct(d2, boot.idx, 2.5)
    kita = kitagawa_a_part(f, q, bouge, boots.idx)
    dc = r["decomposition_strate"]
    for k in ("reussites", "echecs", "composition"):
        assert abs(dc[k]["moyenne"] - kita[k][0]) < 1e-12 and abs(dc[k]["bas"] - kita[k][1]) < 1e-9 \
            and abs(dc[k]["haut"] - kita[k][2]) < 1e-9, (k, dc[k], kita[k])
    assert abs(dc["total"] - r["delta_log_M3_bouge"]) < 1e-12
    # lecture : si seuls les échecs empirent, à réussites identiques, toute la dégradation est « dans les échecs »
    q2 = dict(f, M3=np.where(f["S"] == 1, f["M3"], f["M3"] * np.exp(0.1)))
    dc2 = A5.comparer_version(f, q2, bouge, inv, boot, boots)["decomposition_strate"]
    assert dc2["reussites"]["moyenne"] == 0 and dc2["composition"]["moyenne"] == 0
    assert abs(dc2["echecs"]["moyenne"] - dc2["total"]) < 1e-12 and dc2["total"] > 0.01
    assert A5.decomposition_strate(f, dict(f, S=np.ones_like(f["S"])), bouge, boots) is None   # une issue manque
    ddt = (q["dT_min"].mean(0) - f["dT_min"].mean(0))[bouge]
    assert r["physique_strate"]["dT_min_px"]["moyenne"] == ddt.mean() and r["physique_strate"]["dT_min_px"]["bas"] == pct(ddt, boots.idx, 2.5)
    d = x / 5
    m_, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
    assert abs(r["controle_tost_t"]["p_bas"] - stats.t.sf((m_ + 0.02) / se, n - 1)) < 1e-12
    print("ok comparaison : test sur M3 (p différente avec M2), écarts, bornes bootstrap, décomposition sans "
          "appariement (valeurs, bornes, somme, lecture), écart en unités physiques et TOST t recalculés à part")


def kitagawa_a_part(f, q, bouge, idx):
    """Décomposition recalculée autrement : poids des épisodes par comptage (bincount), moyennes par issue."""
    lf, lq = np.log(np.maximum(f["M3"], 0.05))[:, bouge], np.log(np.maximum(q["M3"], 0.05))[:, bouge]
    Sf, Sq = f["S"][:, bouge] == 1, q["S"][:, bouge] == 1

    def calcul(w):                                    # w : poids de chaque épisode
        W = np.broadcast_to(w, Sf.shape)
        pf, pq = (W * Sf).sum() / W.sum(), (W * Sq).sum() / W.sum()
        mSf, mFf = (W * lf)[Sf].sum() / W[Sf].sum(), (W * lf)[~Sf].sum() / W[~Sf].sum()
        mSq, mFq = (W * lq)[Sq].sum() / W[Sq].sum(), (W * lq)[~Sq].sum() / W[~Sq].sum()
        return np.array([(pf + pq) / 2 * (mSq - mSf), (1 - (pf + pq) / 2) * (mFq - mFf),
                         (pq - pf) * ((mSq + mSf) / 2 - (mFq + mFf) / 2)])

    point = calcul(np.ones(Sf.shape[1]))
    boot = np.array([calcul(np.bincount(i, minlength=Sf.shape[1]).astype(float)) for i in idx])
    return {k: (point[j], np.percentile(boot[:, j], 2.5), np.percentile(boot[:, j], 97.5))
            for j, k in enumerate(("reussites", "echecs", "composition"))}


def physique_a_part(d, v, bouge):
    """Position du T et angle au plus près, course par course, sur les pas où M3 est mesurée (amendement 4)."""
    dT, dth = [], []
    for g in A5.GRAINES:
        x = dict(np.load(d / f"{v.replace(':', '-')}_s{g}.npz"))
        but = x["goal_state"].astype(float)
        lT, lth = [], []
        for i in range(x["state"].shape[1]):
            e = x["state"][:, i].astype(float)
            fini = ~np.isnan(e).any(axis=1)
            if not x["success"][i] and not fini.all():               # course divergente : pas finis seulement
                j = int(np.argmin(fini))
                e = e[:j] if j > 0 else x["init_state"][i][None].astype(float)
            else:                                                     # succès : pas gelés ignorés
                e = e[~np.isnan(e).all(axis=1)]
            lT.append(np.hypot(e[:, 2] - but[i, 2], e[:, 3] - but[i, 3]).min())
            lth.append(np.degrees(ecart_angle(e[:, 4], but[i, 4]).min()))
        dT.append(lT)
        dth.append(lth)
    return np.array(dT)[:, bouge].mean(0), np.array(dth)[:, bouge].mean(0)


# ---------------------------------------------------------------- grilles simulées

def grille(tmp, nom, sha=None, **kw):
    """Grille simulée : 3 versions × graines 52 à 56, 2 000 trajectoires distinctes, meta.json d'EXP-005."""
    d, fin, excl = Path(tmp) / f"{nom}-exp005", Path(tmp) / f"{nom}-fin", Path(tmp) / f"{nom}-exclus.json"
    sauve = A.N, A.VERSIONS, A.GA
    A.N, A.VERSIONS, A.GA = A5.N5, A5.VERSIONS5, A5.GRAINES
    kw.setdefault("paquet_reussi", ("all:6", 53))
    try:
        T.simuler(d, **kw)
    finally:
        A.N, A.VERSIONS, A.GA = sauve
    for g in A.GB:
        (d / f"fp32_s{g}.npz").unlink()
    ep = np.arange(A5.N5) + 1000                              # trajectoires distinctes, absentes de l'exclusion
    for f in d.glob("*.npz"):
        x = dict(np.load(f))
        x["episodes_idx"] = ep
        if sha:
            x["git"] = sha
        np.savez_compressed(f, **x)
    exclus = [[100_000 + i, i % 90] for i in range(A5.N_EXCLUS)]
    excl.write_text(json.dumps({"protocole": {"episodes_idx": [a for a, _ in exclus], "start_steps": [b for _, b in exclus]}}))
    m = json.loads((d / "meta.json").read_text())
    m["protocole"].update(episodes_idx=ep.tolist(), graine_episodes=A5.GRAINE_EPISODES, exclus=exclus,
                          regle_tirage=A5.REGLE_TIRAGE)
    (d / "meta.json").write_text(json.dumps(m))
    fin.mkdir()
    shutil.copy(d / "meta.json", fin / "meta.json")
    shutil.copy(d / "all-6_s52.npz", fin / "all-6_s52.npz")
    return d, fin, excl


def sans_verif_code(fonction):
    vrais = A5.verifier_code5, A5.verifier_analyse
    A5.verifier_code5 = A5.verifier_analyse = lambda sha: None
    try:
        return fonction()
    finally:
        A5.verifier_code5, A5.verifier_analyse = vrais


def analyser(d, fin, excl, effectifs=None):
    return sans_verif_code(lambda: A5.analyser(d, fin, [excl], effectifs=effectifs))


def charger(d, v, cle):
    return np.stack([dict(np.load(d / f"{v.replace(':', '-')}_s{g}.npz"))[cle] for g in A5.GRAINES])


def marges(d, v):
    return [np.stack(x) for x in zip(*[A.controler_course(dict(np.load(d / f"{v.replace(':', '-')}_s{g}.npz")), v)[:2]
                                        for g in A5.GRAINES])]


def test_code():
    for f, empreinte in (("run_grid.py", A5.RUN_GRID_SHA256), ("quant.py", A5.QUANT_SHA256)):
        assert hashlib.sha256((A.ICI / f).read_bytes()).hexdigest() == empreinte, f"{f} a changé"
    for sha, motif in (("c5703dbbacb1537be9cc03377a3a922f4a81c1ed", "run_grid.py d'EXP-005"), ("0" * 40, "inconnu")):
        try:
            A5.verifier_code5(sha)
            raise AssertionError(sha)
        except A.ControleEchoue as err:
            assert motif in str(err), err
    assert PROPRE, "arbre modifié : committer avant de lancer les tests"
    A5.verifier_code5(TETE)
    A5.verifier_analyse(TETE)
    print(f"ok code : empreintes, code d'analyse identique à {TETE[:7]} ; c5703db et commit inconnu refusés")


def test_scenarios(tmp):
    # 1. effet cherché à 6 bits (réussite inchangée, échecs plus loin), rien à 8 bits ; vraies vérifications du code
    d, fin, excl = grille(tmp, "effet", sha=TETE, effets={"all:6": (0.0, 1.5)}, graine=11, divergents=(("all:6", 54),))
    r = A5.analyser(d, fin, [excl], effectifs=None)
    h, k = r["H6"], r["controle_8_bits"]
    assert r["controles"]["commits"] == [TETE] and r["controles"]["commit_analyse"] == TETE and not r["controles"]["analyse_modifiee"]
    assert h["verdict"] == "établie" and r["progression_8_vers_6"]["etablie"] and r["progression_8_vers_6"]["sens"] == 1, h
    assert r["controles"]["divergences"]["all:6"] == 2
    bouge = A.depart({("fp32", A.GA[0]): dict(np.load(d / "fp32_s52.npz"))})["bouge"]
    nb = int(bouge.sum())
    K = {v: charger(d, v, "success").astype(np.int64).sum(0) for v in A5.VERSIONS5}
    M = {v: marges(d, v) for v in A5.VERSIONS5}                   # (M2, M3)
    lg3 = {v: np.log(np.maximum(M[v][1], 0.05)).mean(0) for v in A5.VERSIONS5}
    x, w = K["all:6"] - K["fp32"], (lg3["all:6"] - lg3["fp32"])[bouge]
    assert h["p_bas"] == p_numpy(10 * x + 1, "plus") and h["p_haut"] == p_numpy(10 * x - 1, "moins")
    assert h["p_b"] == p_numpy(w, "plus", nb) and h["p"] == max(h["p_bas"], h["p_haut"], h["p_b"])
    assert h["delta_reussite"] == x.mean() / 5 and abs(h["delta_log_M3_bouge"] - w.mean()) < 1e-12 and h["episodes_bouge"] == nb
    w8 = (lg3["all:8"] - lg3["fp32"])[bouge]
    assert k["p_b"] == p_numpy(w8, "plus", nb) and k["delta_reussite"] == (K["all:8"] - K["fp32"]).mean() / 5
    d68 = (lg3["all:6"] - lg3["all:8"])[bouge]
    g = r["progression_8_vers_6"]
    assert g["p_unilaterale"] == p_numpy(d68, "plus", nb) and g["ecart_6_moins_8"]["bas"] == pct(d68, A.Bootstrap(n=nb, graine=3).idx, 2.5)
    S = charger(d, "fp32", "success")
    paires = [(a, b) for a in range(5) for b in range(5) if a != b]
    plancher = np.mean([S[a] != S[b] for a, b in paires], axis=0) * 50
    assert r["descriptifs"]["plancher_discordants_par_50"]["moyenne"] == plancher.mean()
    for v in A5.VERSIONS5:
        assert r["descriptifs"][v]["reussite"] == charger(d, v, "success").mean()
    (tf, af), (t6, a6) = physique_a_part(d, "fp32", bouge), physique_a_part(d, "all:6", bouge)
    bs = A.Bootstrap(n=nb, graine=3).idx
    for cle, ecart in (("dT_min_px", t6 - tf), ("dth_min_deg", a6 - af)):
        e = h["physique_strate"][cle]
        assert abs(e["moyenne"] - ecart.mean()) < 1e-9 and abs(e["bas"] - pct(ecart, bs, 2.5)) < 1e-9, (cle, e)
    # une course divergente : ses pas après la divergence (T encore fini) ne comptent pas
    x = dict(np.load(d / "all-6_s54.npz"))
    i = int(np.where(~x["success"] & np.isnan(x["state"]).any(axis=(0, 2)) & ~np.isnan(x["state"][0]).any(axis=1))[0][0])
    e = A5.etat_mesure(x)[:, i]
    assert np.isfinite(x["state"][25:, i, 2:5]).all() and np.array_equal(e[20:], np.repeat(x["state"][19:20, i], 30, axis=0))
    txt = A5.rapport(r)
    assert "établie" in txt and "Progression selon les bits" in txt and not re.search(r"\bnan\b|\binf\b", txt.lower())
    assert "dans les échecs" in txt and "20 px sur l'agent et le T ensemble" in txt
    r_inv = dict(r, progression_8_vers_6=dict(r["progression_8_vers_6"], sens=-1, etablie=False))
    assert "non établie, pose plus mauvaise à 8 bits qu'à 6 bits" in A5.rapport(r_inv)
    print("ok effet cherché (vraies vérifications du code) : H6 établie, progression établie ; p, écarts, contrôle à "
          "8 bits, progression, plancher, réussites et unités physiques (course divergente comprise) recalculés à part")

    # 2. la réussite baisse aussi : H6 ne passe pas
    d2, fin2, excl2 = grille(tmp, "baisse", effets={"all:6": (1.2, 1.5)}, graine=12, divergents=())
    r2 = analyser(d2, fin2, excl2)
    assert r2["H6"]["verdict"] == "non établie" and r2["H6"]["p_bas"] > 0.5 and r2["H6"]["p_b"] < 0.05
    assert r2["progression_8_vers_6"] is None and "Progression selon les bits" not in A5.rapport(r2)
    print("ok réussite en baisse : H6 non établie, progression non rapportée")

    # 3. aucun effet : versions identiques épisode par épisode ; 8 bits identique à 6 bits, donc pas de progression
    d3, fin3, excl3 = grille(tmp, "nul", graine=13, divergents=(), identiques=("all:6", "all:8"), paquet_reussi=())
    r3 = analyser(d3, fin3, excl3)
    h = r3["H6"]
    assert h["delta_reussite"] == 0 and h["delta_log_M3_bouge"] == 0 and h["p_b"] == 1.0 and h["verdict"] == "non établie"
    assert r3["progression_8_vers_6"] is None and r3["H6"]["decomposition_strate"]["total"] == 0
    print("ok sans effet : écarts nuls, p_b = 1, H6 non établie, progression non rapportée")
    return d, fin, excl


def test_refus(tmp, d, fin, excl):
    def doit_echouer(nom, alterer, motif, reel=False, effectifs=None):
        e, ef, ex = Path(tmp) / f"r-{nom}", Path(tmp) / f"r-{nom}-fin", Path(tmp) / f"r-{nom}-exclus.json"
        shutil.copytree(d, e)
        shutil.copytree(fin, ef)
        shutil.copy(excl, ex)
        alterer(e, ef, ex)
        try:
            if reel:
                A5.analyser(e, ef, [ex], effectifs=effectifs)
            else:
                analyser(e, ef, ex, effectifs)
        except A.ControleEchoue as err:
            assert motif in str(err), (nom, str(err))
            print(f"   refusé comme attendu ({nom}) : {err}")
            return
        raise AssertionError(f"{nom} : contrôle non déclenché")

    def meta(dossier, **champs):
        m = json.loads((dossier / "meta.json").read_text())
        m["protocole"].update(champs)
        (dossier / "meta.json").write_text(json.dumps(m))

    def modifier(dossier, f, **champs):
        x = dict(np.load(dossier / f))
        x.update(champs)
        np.savez_compressed(dossier / f, **x)

    def changer(dossier, f, cle, fonction):
        x = dict(np.load(dossier / f))
        x[cle] = fonction(x[cle])
        np.savez_compressed(dossier / f, **x)

    def trajectoire_exclue(e, ef, ex):
        x = json.loads(ex.read_text()); x["protocole"]["episodes_idx"][0] = 1000; ex.write_text(json.dumps(x))
        for dd in (e, ef):
            m = json.loads((dd / "meta.json").read_text()); m["protocole"]["exclus"][0][0] = 1000
            (dd / "meta.json").write_text(json.dumps(m))

    def doublon(e, ef, ex):
        ep = list(range(1000, 1000 + A5.N5)); ep[1] = ep[0]
        for dd in (e, ef):
            meta(dd, episodes_idx=ep)
            for f in dd.glob("*.npz"):
                modifier(dd, f.name, episodes_idx=np.array(ep))

    def exclusion_courte(e, ef, ex):
        x = json.loads(ex.read_text())
        x["protocole"]["episodes_idx"].pop(); x["protocole"]["start_steps"].pop(); ex.write_text(json.dumps(x))

    inverser = lambda dossier, f, cle="success": changer(dossier, f, cle, lambda a: np.r_[~a[:1], a[1:]] if a.dtype == bool else a + 1)
    cas = [
        ("trajectoire exclue", trajectoire_exclue, "trajectoires en commun"),
        ("deux positions par trajectoire", doublon, "plusieurs positions dans une même trajectoire"),
        ("exclusion de 549 couples", exclusion_courte, "549 couples au lieu de 550"),
        ("graine de tirage", lambda e, ef, ex: meta(e, graine_episodes=42), "graine de tirage"),
        ("règle de tirage", lambda e, ef, ex: meta(e, regle_tirage="couples"), "règle de tirage différente"),
        ("nombre d'épisodes", lambda e, ef, ex: meta(e, n=1999), "1999 épisodes au lieu de 2000"),
        ("fichier manquant", lambda e, ef, ex: (e / "all-8_s54.npz").unlink(), "fichiers manquants"),
        ("couches", lambda e, ef, ex: modifier(e, "all-6_s55.npz", n_couches=73), "73 couches"),
        ("arbre modifié", lambda e, ef, ex: modifier(e, "all-8_s53.npz", git_sale=True), "arbre git modifié"),
        ("deux commits", lambda e, ef, ex: modifier(e, "fp32_s56.npz", git="1" * 40), "plusieurs commits"),
        ("positions", lambda e, ef, ex: changer(e, "all-8_s55.npz", "positions", lambda a: a[::-1].copy()), "positions"),
        ("épisodes du fichier", lambda e, ef, ex: changer(e, "all-6_s54.npz", "episodes_idx", lambda a: a[::-1].copy()),
         "épisodes différents de meta.json"),
        ("taille des paquets", lambda e, ef, ex: modifier(e, "fp32_s53.npz", chunk=25), "taille des paquets"),
        ("forme de l'état", lambda e, ef, ex: changer(e, "fp32_s54.npz", "state", lambda a: a[:40]), "forme de l'état"),
        ("état de départ", lambda e, ef, ex: changer(e, "all-8_s52.npz", "init_state", lambda a: a + 1), "init_state différent"),
        ("paquet arrêté tôt", lambda e, ef, ex: modifier(e, "fp32_s55.npz", n_steps=np.r_[30, np.full(39, A.T)].astype(np.int32)),
         "paquet 0 arrêté à 30 pas"),
        ("clôture sur une autre version", lambda e, ef, ex: shutil.copy(e / "fp32_s52.npz", ef / "all-6_s52.npz"),
         "sanité de clôture : pas la même version"),
        ("clôture : succès", lambda e, ef, ex: inverser(ef, "all-6_s52.npz"), "sanité de clôture : success diffère"),
        ("clôture : contacts", lambda e, ef, ex: inverser(ef, "all-6_s52.npz", "contacts"), "sanité de clôture : contacts diffère"),
        ("clôture : nombre de pas", lambda e, ef, ex: modifier(ef, "all-6_s52.npz", n_steps=np.full(40, 49, dtype=np.int32)),
         "sanité de clôture : n_steps diffère"),
        ("clôture : arbre modifié", lambda e, ef, ex: modifier(ef, "all-6_s52.npz", git_sale=True), "sanité de clôture : arbre git modifié"),
        ("clôture : torch", lambda e, ef, ex: (ef / "meta.json").write_text(
            json.dumps(dict(json.loads((ef / "meta.json").read_text()), torch="2.15.0"))), "protocole ou torch différent"),
    ]
    for nom, alterer, motif in cas:
        doit_echouer(nom, alterer, motif)
    doit_echouer("strates", lambda e, ef, ex: None, "strates de départ inattendues", effectifs=A5.EFFECTIFS5)
    # vraies vérifications du code : un commit dont le code d'analyse diffère de celui du dossier est refusé
    ancien = subprocess.run(["git", "-C", str(A.ICI), "rev-parse", "35f312f"], capture_output=True, text=True).stdout.strip()

    def ancien_commit(e, ef, ex):
        for dd in (e, ef):
            for f in dd.glob("*.npz"):
                modifier(dd, f.name, git=ancien)
    doit_echouer("code d'analyse d'un autre commit", ancien_commit, "code d'analyse diffère", reel=True)
    print(f"ok contrôles bloquants d'EXP-005 ({len(cas) + 2} refus, chacun avec son motif)")


if __name__ == "__main__":
    test_statistique_a_la_main()
    test_formulation()
    test_comparer_sur_M3()
    test_code()
    with tempfile.TemporaryDirectory() as tmp:
        d, fin, excl = test_scenarios(tmp)
        test_refus(tmp, d, fin, excl)
    print("tous les tests passent")
