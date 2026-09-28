"""Tests de analyse_exp003.py sur des données simulées, avant toute lecture des fichiers de la grille.

Le simulateur écrit 55 fichiers au format de run_grid.py, avec des effets connus : succès tirés avec un
hasard partagé en partie entre versions à la même graine (comme les graines de la CEM), marges
cohérentes avec le succès (M2 < 1 au pas du succès, gel ensuite), état de départ et but.

Usage :
    python test_analyse_exp003.py            # tout
    python test_analyse_exp003.py rapide     # sans les tests de niveau (plus longs)
"""
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy import stats

import analyse_exp003 as A

SHA_BON = "c5703dbbacb1537be9cc03377a3a922f4a81c1ed"
SHA_AUTRE_CODE = "3b178ce"                 # run_grid.py différent : doit être refusé
LIM = np.pi / 9


def sigmoide(x):
    return 1 / (1 + np.exp(-x))


def simuler(dossier, effets=None, rho=0.7, graine=0):
    """Écrit une grille simulée. effets : {version: (baisse de logit, facteur des marges des échecs)}."""
    effets = effets or {}
    rng = np.random.default_rng(graine)
    n = A.N
    dossier = Path(dossier)
    dossier.mkdir(parents=True, exist_ok=True)
    goal = np.zeros((n, 7), dtype=np.float32)
    goal[:, :4] = rng.uniform(150, 350, (n, 4))
    goal[:, 4] = rng.uniform(0, 2 * np.pi, n)
    m0 = rng.uniform(0.3, 4, n)                               # marge du T au départ
    u0 = rng.normal(size=(n, 4))
    u0 /= np.linalg.norm(u0, axis=1, keepdims=True)
    init = goal.copy()
    init[:, :4] += (20 * m0[:, None] * u0 / np.maximum(np.linalg.norm(u0[:, 2:4], axis=1), 1e-3)[:, None])
    init[:, 4] = np.mod(goal[:, 4] + LIM * m0 * 0.8, 2 * np.pi)
    eta = rng.normal(2.0, 2.0, n)                             # logit de réussite en pleine précision
    h = rng.exponential(1.0, n)                               # sensibilité de l'épisode à la compression
    U = {"A": rng.uniform(size=(5, n)), "B": rng.uniform(size=(5, n))}
    base = {g: 1.05 + np.minimum(rng.exponential(1.5, (5, n)), 5) for g in ("A", "B")}
    ep_idx = rng.integers(0, 10_000, n)
    st_idx = rng.integers(0, 100, n)
    (dossier / "meta.json").write_text(json.dumps({"protocole": {
        "n": n, "chunk": 50, "episodes_idx": ep_idx.tolist(), "start_steps": st_idx.tolist()}}))
    for v in A.VERSIONS:
        delta, facteur = effets.get(v, (0.0, 1.0))
        p = sigmoide(eta - delta * h)
        rv = np.random.default_rng([graine, A.VERSIONS.index(v)])
        for groupe, graines in (("A", A.GA), ("B", A.GB)):
            if groupe == "B" and v != "fp32":
                continue
            for j, g in enumerate(graines):
                u = U[groupe][j].copy()
                if v != "fp32":                               # hasard partagé en partie avec fp32
                    frais = rv.uniform(size=n) > rho
                    u[frais] = rv.uniform(size=frais.sum())
                succes = u < p
                rr = np.random.default_rng([graine, g])       # géométrie partagée entre versions à la même graine
                ecrire(dossier, v, g, succes, base[groupe][j] * np.where(succes, 1.0, facteur), goal, init,
                       ep_idx, st_idx, rr)


def ecrire(dossier, v, g, succes, mfin, goal, init, ep_idx, st_idx, rr, sha=SHA_BON, sale=False):
    n, T = A.N, A.T
    premier = np.where(succes, rr.integers(3, T, n), -1).astype(np.int32)
    r_ang = rr.uniform(0.2, 0.95, n)                          # terme d'angle sous le terme de position
    u = rr.normal(size=(n, 4))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    t = np.arange(T)[:, None]
    m_ech = mfin[None, :] + 3 * (1 - t / (T - 1))            # échecs : décroît jusqu'à mfin >= 1,05
    k = np.maximum(premier, 1)[None, :]
    eps = rr.uniform(0.01, 0.3, n)[None, :]
    m_suc = np.where(t < k, 1 + 3 * (1 - t / k), 1 - eps)     # succès : franchit 1 au pas du succès
    m = np.where(succes[None, :], m_suc, m_ech)
    etat = np.empty((T, n, 7))
    etat[..., :4] = goal[None, :, :4] + 20 * m[..., None] * u[None]
    etat[..., 4] = np.mod(goal[None, :, 4] + LIM * m * r_ang[None, :], 2 * np.pi)
    etat[..., 5:] = rr.normal(size=(T, n, 2))
    gele = succes[None, :] & (t > premier[None, :])
    etat[gele] = np.nan
    contacts = np.where(gele, -1, rr.integers(0, 3, (T, n))).astype(np.int32)
    np.savez_compressed(Path(dossier) / f"{v.replace(':', '-')}_s{g}.npz", spec=v, seed=g, n_couches=0,
                        git=sha, git_sale=sale, positions=np.arange(n), episodes_idx=ep_idx, start_steps=st_idx,
                        chunk=50, state=etat, block_pose=etat[..., 2:5], agent_pos=etat[..., :2],
                        contacts=contacts, env_state_dist=np.where(gele, np.nan, 1.0), success=succes,
                        first_success=premier, n_steps=np.full(10, T, dtype=np.int32),
                        init_state=init, goal_state=goal)


# ---------------------------------------------------------------- tests unitaires

def test_inversion_exacte():
    """p par inversion de signe contre l'énumération complète des 2^12 signes, sur un réseau d'entiers."""
    rng = np.random.default_rng(5)
    n = 12
    signes = np.array(np.meshgrid(*[[-1, 1]] * n)).reshape(n, -1).T
    inv = A.Inversion(n=n, b=200_000, graine=0)
    for _ in range(20):
        x = rng.integers(-5, 6, n)
        t = signes @ x
        exact = np.mean(np.abs(t) >= abs(x.sum()))
        assert abs(inv.p(x) - exact) < 0.01, (inv.p(x), exact)
        exact_plus = np.mean(t >= x.sum())
        assert abs(inv.p(x, "plus") - exact_plus) < 0.01
    inv = A.Inversion(n=60, b=1000)
    assert inv.p(np.zeros(60)) == 1.0                         # égalités comprises : somme nulle, p = 1
    assert inv.p(np.full(60, 5)) == 1 / 1001                  # 2^-60 : aucun tirage n'atteint la somme maximale
    try:
        inv.p(np.r_[np.nan, np.zeros(59)])
        raise AssertionError("NaN accepté")
    except A.ControleEchoue:
        pass
    inv = A.Inversion()                                       # produit en float64 = produit exact en entiers
    X = rng.integers(-11, 12, (20, A.N))
    with np.errstate(all="ignore"):
        assert np.array_equal(X @ inv.s.astype(np.int64).T, np.round(inv.s @ X.T.astype(float)).T.astype(np.int64))
    with np.errstate(all="ignore"):
        assert np.array_equal((X.astype(float) @ inv.s.T), (X @ inv.s.astype(np.int64).T).astype(float))
    print("ok inversion de signe (exacte sur réseau, égalités comprises)")


def test_holm():
    adj = A.holm([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06]), adj
    adj = A.holm([0.5, 0.001, 0.02])
    assert np.allclose(adj, [0.5, 0.003, 0.04]), adj
    print("ok Holm")


def test_mcnemar():
    rng = np.random.default_rng(1)
    R = rng.integers(0, 2, (200, 40))
    Aa = np.where(rng.uniform(size=R.shape) < 0.2, 1 - R, R)
    p = A.mcnemar_exact(R, Aa)
    for i in range(200):
        b = int(((R[i] == 1) & (Aa[i] == 0)).sum())
        c = int(((R[i] == 0) & (Aa[i] == 1)).sum())
        ref = 1.0 if b + c == 0 else stats.binomtest(b, b + c, 0.5).pvalue
        assert abs(p[i] - ref) < 1e-12, (b, c, p[i], ref)
    assert A.mcnemar_exact(np.ones((1, 5)), np.ones((1, 5)))[0] == 1.0
    print("ok McNemar exact")


def test_tost():
    """TOST par inversion de signe proche du TOST t apparié ; décalage de ± 1 dans le bon sens."""
    rng = np.random.default_rng(3)
    inv = A.Inversion()
    for a in (0.0, 0.02, 0.05, 0.08):                        # écart de réussite -2 a / 5 : 0 à -3,2 points
        x = rng.choice([-1, 0, 1], size=A.N, p=[0.1 + a, 0.8, 0.1 - a])
        p_bas, p_haut = inv.p(10 * x + 1, "plus"), inv.p(10 * x - 1, "moins")
        c = A.tost_t(x / 5)
        assert abs(max(p_bas, p_haut) - c["p"]) < 0.02 or max(p_bas, p_haut, c["p"]) < 0.002, (p_bas, p_haut, c)
    x = np.zeros(A.N, dtype=int)
    x[:20] = -1                                               # écart de -0,008 : dans la marge
    assert max(inv.p(10 * x + 1, "plus"), inv.p(10 * x - 1, "moins")) < 0.05
    x[:100] = -1                                              # écart de -0,04 : hors de la marge
    assert inv.p(10 * x + 1, "plus") > 0.5
    print("ok TOST (inversion de signe contre t apparié)")


def test_niveau():
    """Niveau des tests sous le nul, avec un hasard partagé en partie à la même graine."""
    rng = np.random.default_rng(11)
    inv = A.Inversion(b=2000)
    rejets, rejets_casse = [], []
    for _ in range(400):
        p = sigmoide(rng.normal(2.0, 2.0, A.N))
        U = rng.uniform(size=(5, A.N))
        V = np.where(rng.uniform(size=U.shape) < 0.7, U, rng.uniform(size=U.shape))
        W = rng.uniform(size=(5, A.N))
        Kf, Kv, KB = (U < p).sum(0), (V < p).sum(0), (W < p).sum(0)
        rejets.append(inv.p(Kv - Kf) < 0.05)
        u = ((KB - Kv) >= 3).astype(float) - ((KB - Kf) >= 3).astype(float)
        rejets_casse.append(inv.p(u, "plus") < 0.05)
    a, b = np.mean(rejets), np.mean(rejets_casse)
    assert 0.02 <= a <= 0.08, a
    assert b <= 0.08, b
    print(f"ok niveau sous le nul : P1 {a:.3f}, épisodes cassés {b:.3f} (attendu 0,05 ou moins)")


# ---------------------------------------------------------------- contrôles bloquants

def test_controles(tmp):
    d = Path(tmp) / "propre"
    simuler(d, graine=1)
    res, shas = A.charger(d)
    D, lim = A.tables(res)
    assert lim == 0 and shas == [SHA_BON]

    def doit_echouer(nom, alterer):
        e = Path(tmp) / nom
        shutil.copytree(d, e)
        alterer(e)
        try:
            A.tables(A.charger(e)[0])
        except A.ControleEchoue as err:
            print(f"   refusé comme attendu ({nom}) : {err}")
            return
        raise AssertionError(f"{nom} : contrôle non déclenché")

    def modifier(e, f, **champs):
        x = dict(np.load(e / f))
        x.update(champs)
        np.savez_compressed(e / f, **x)

    def nan_avant(e):
        x = dict(np.load(e / "all-4_s43.npz"))
        i = int(np.where(~x["success"])[0][0])
        x["state"][10, i, 2] = np.nan
        np.savez_compressed(e / "all-4_s43.npz", **x)

    def succes_faux(e):
        x = dict(np.load(e / "all-3_s44.npz"))
        i = int(np.where(~x["success"])[0][0])
        x["success"][i] = True
        x["first_success"][i] = 49
        np.savez_compressed(e / "all-3_s44.npz", **x)

    doit_echouer("fichier manquant", lambda e: (e / "encoder-3_s45.npz").unlink())
    doit_echouer("arbre modifié", lambda e: modifier(e, "fp32_s49.npz", git_sale=True))
    doit_echouer("autre code", lambda e: modifier(e, "fp32_s50.npz", git=SHA_AUTRE_CODE))
    doit_echouer("NaN avant le succès", nan_avant)
    doit_echouer("succès sans marge", succes_faux)
    doit_echouer("autres épisodes", lambda e: modifier(e, "all-8_s42.npz", episodes_idx=np.arange(A.N)))

    # cas limite d'arrondi : un échec dont la marge minimale vaut 1 - 5e-5 passe, et il est compté
    e = Path(tmp) / "limite"
    shutil.copytree(d, e)
    x = dict(np.load(e / "all-6_s46.npz"))
    i = int(np.where(~x["success"])[0][0])
    m, _, _ = A.marges(x)
    d4 = np.linalg.norm(x["state"][:, i, :4] - x["goal_state"][i, :4], axis=-1).min() / 20
    x["state"][:, i, :4] = x["goal_state"][i, :4] + (x["state"][:, i, :4] - x["goal_state"][i, :4]) * (1 - 5e-5) / d4
    x["state"][:, i, 4] = x["goal_state"][i, 4]            # angle au but : la marge vaut celle de la position
    np.savez_compressed(e / "all-6_s46.npz", **x)
    assert A.tables(A.charger(e)[0])[1] >= 1
    print("ok contrôles bloquants (6 refus, 1 cas limite accepté et compté)")


# ---------------------------------------------------------------- bout en bout

def test_bout_en_bout(tmp):
    effets = {"all:3": (3.0, 1.0), "all:4": (1.0, 1.0), "encoder:4": (1.5, 1.0), "predictor:4": (0.2, 1.0),
              "encoder:3": (3.0, 1.0), "predictor:3": (1.0, 1.0), "all:5": (0.0, 1.5)}
    d = Path(tmp) / "effets"
    simuler(d, effets=effets, graine=2)
    r = A.analyser(d)
    t = r["tests"]
    assert t["P1"]["verdict"] == "soutenue" and t["P1"]["somme"] < 0, t["P1"]
    assert t["P2"]["verdict"] == "soutenue" and t["P2"]["somme"] > 0, t["P2"]
    assert t["P3'"]["verdict"] == "établie", t["P3'"]
    # la dégradation de M3 injectée sur les échecs de all:5 (facteur 1,5) se retrouve dans la classe FF
    dec = r["decomposition"]["all:5"]
    somme = sum(dec[c]["contribution_log_M3"] for c in ("SS", "SF", "FS", "FF"))
    assert abs(somme - r["descriptifs"]["all:5"]["delta_log_M3"]["moyenne"]) < 1e-12
    ff = dec["FF"]
    assert abs(ff["contribution_log_M3"] / (ff["courses"] / (5 * A.N)) - np.log(1.5)) < 0.02, ff
    assert abs(dec["SS"]["contribution_log_M3"]) < 0.02
    ec = r["episodes_casses"]
    assert ec["all:3"]["observes"] > 3 * max(ec["nuls"], 1) and ec["all:3"]["p_plus"] < 0.01
    assert ec["all:8"]["p_plus"] > 0.01
    ps = r["protocoles_simules"]
    assert ps["n500_graines1"]["fp32 -> all:3"]["apparie"]["part_conclut_meme_sens"] > 0.99
    nul = ps["n50_graines1"]["nul vrai : fp32 replique -> fp32"]["non_apparie"]["part_significatif"]
    assert nul <= 0.08, nul
    assert r["descriptifs"]["all:3"]["delta_reussite"]["haut"] < 0
    assert r["plancher"]["all:3"]["attribution_au_bruit"] < 0.5
    txt = A.rapport(r)
    assert "P3'" in txt and not re.search(r"\bnan\b", txt.lower())
    json.dumps(A.pur(r), ensure_ascii=False)
    print("ok bout en bout, effets connus : P1, P2, P3' retrouvés, décomposition exacte, épisodes cassés")

    d = Path(tmp) / "nul"
    simuler(d, graine=3)
    r = A.analyser(d)
    couvrent = sum(r["descriptifs"][v]["delta_reussite"]["bas"] <= 0 <= r["descriptifs"][v]["delta_reussite"]["haut"]
                   for v in A.QUANTIFIEES)
    assert couvrent >= 7, couvrent
    assert r["tests"]["P3'"]["verdict"] == "non établie"
    print(f"ok bout en bout, sans effet : {couvrent} intervalles sur 9 couvrent 0, P3' non établie")
    return r


if __name__ == "__main__":
    rapide = len(sys.argv) > 1 and sys.argv[1] == "rapide"
    test_inversion_exacte()
    test_holm()
    test_mcnemar()
    test_tost()
    if not rapide:
        test_niveau()
    with tempfile.TemporaryDirectory() as tmp:
        test_controles(tmp)
        test_bout_en_bout(tmp)
    print("tous les tests passent")
