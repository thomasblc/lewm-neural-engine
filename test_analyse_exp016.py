"""Tests de analyse_exp016.py (version 2), sur données simulées (aucun résultat réel lu avant le commit de l'analyse) et
sur de vrais fichiers copiés pour les refus.
1. La loi a priori non paramétrique retrouve un mélange connu.
2. Calibration du test de strate sur une grille : r de 0,45 à 0,97 ; loi a priori bêta réaliste (0,5 ; 0,08) ou mélange
   à trois points ; vérité « amincissement » ou « perte plafonnée » (q = min(p, c)) ; rejets aux seuils 5 % et 5 % / 7 au
   plus seuil + 3 erreurs-types (300 jeux par case).
3. Puissance : cassures plantées (2 et 5 à r = 0,97, 10 à r = 0,85, 10 et 30 à r = 0,45), taux de détection rapporté ;
   au moins 80 % exigés pour 5 à r = 0,97 et 10 à r = 0,85.
4. Recomptage indépendant de T1, de la p et de l'excès (mêmes tirages, code écrit à part).
5. Calibration synthétique acceptée sous une vraie perte proportionnelle.
6. Refus des fichiers non conformes, chacun avec son motif. À lancer sur un arbre commité.

Usage :
    python test_analyse_exp016.py
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

import analyse_exp003 as A
import analyse_exp016 as A16

RES = A16.ICI.parent / "resultats"
TETE = subprocess.run(["git", "-C", str(A16.ICI), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
PROPRE = subprocess.run(["git", "-C", str(A16.ICI), "status", "--porcelain", "--untracked-files=no"],
                        capture_output=True, text=True).stdout.strip() == ""


def tirer_p(rng, loi, n):
    if loi == "beta":
        return rng.beta(0.5, 0.08, n)
    return rng.choice(np.array([1.0, 0.0, 0.7]), n, p=[0.4, 0.2, 0.4])      # mélange à trois points


def simuler(rng, loi, verite, r, n_ep=500, casses=0):
    """k_R (5 essais), X42 (5 graines x épisodes), k_V d'une version : amincissement q = r p ou plafond q = min(p, c) de
    même moyenne ; cassures plantées parmi les épisodes de la strate."""
    p = tirer_p(rng, loi, n_ep)
    kR = rng.binomial(5, p)
    X42 = (rng.random((5, n_ep)) < p).astype(np.int64)
    if verite == "amincissement":
        q = r * p
    else:
        cible, bas, haut = r * p.mean(), 0.0, 1.0
        for _ in range(60):
            c = (bas + haut) / 2
            bas, haut = (c, haut) if np.minimum(p, c).mean() < cible else (bas, c)
        q = np.minimum(p, (bas + haut) / 2)
    kV = rng.binomial(5, q)
    if casses:
        idx = rng.choice(np.flatnonzero(kR == 5), casses, replace=False)
        kV[idx] = 0
    return kR, X42, kV


def test_npmle():
    rng = np.random.default_rng(1)
    p = tirer_p(rng, "melange", 20_000)
    k = rng.binomial(10, p)
    w = A16.npmle(k, 10)
    g = A16.GRILLE
    masses = (w[g >= 0.95].sum(), w[g <= 0.05].sum(), w[(g >= 0.6) & (g <= 0.8)].sum())
    assert abs(masses[0] - 0.4) < 0.05 and abs(masses[1] - 0.2) < 0.05 and abs(masses[2] - 0.4) < 0.05, masses
    assert abs(w.sum() - 1) < 1e-9
    print(f"ok loi a priori non paramétrique : masses {masses[0]:.3f}, {masses[1]:.3f}, {masses[2]:.3f} (0,4 ; 0,2 ; 0,4)")


def test_amincir():
    rng = np.random.default_rng(2)
    p = rng.beta(0.5, 0.08, size=(50, 400))
    for cible in (0.3, 0.8, 0.95, 0.99):
        q = A16.amincir(p, cible)
        assert np.allclose(q.mean(axis=1), cible, atol=1e-6) and (q >= 0).all() and (q <= 1).all(), cible
    print("ok amincissement : moyenne visée atteinte à 1e-6 près, plafonnement compris")


def test_calibration():
    rng = np.random.default_rng(3)
    n = 300
    se = lambda a: 3 * np.sqrt(a * (1 - a) / n)
    lignes = []
    for loi in ("beta", "melange"):
        for verite in ("amincissement", "plafond"):
            for r in (0.45, 0.83, 0.90, 0.97):
                ps = []
                for i in range(n):
                    kR, X42, kV = simuler(rng, loi, verite, r)
                    w = A16.npmle(kR + X42.sum(axis=0), 10, iterations=2000)
                    ps.append(A16.test_strate(kR, kV, w, b=1_000, graine=1000 + i)["p"])
                ps = np.array(ps)
                t5, t7 = (ps < 0.05).mean(), (ps < 0.05 / 7).mean()
                lignes.append(f"{loi}/{verite}/r={r} : {100 * t5:.1f} % ; {100 * t7:.1f} %")
                assert t5 <= 0.05 + se(0.05) and t7 <= 0.05 / 7 + se(0.05 / 7), lignes[-1]
    print("ok calibration du test de strate (rejets à 5 % ; à 5 %/7) :\n   " + "\n   ".join(lignes))


def test_puissance():
    rng = np.random.default_rng(4)
    out = []
    for r, casses in ((0.97, 5), (0.97, 2), (0.85, 10), (0.45, 10), (0.45, 30)):
        det = []
        for i in range(200):
            kR, X42, kV = simuler(rng, "beta", "amincissement", r, casses=casses)
            w = A16.npmle(kR + X42.sum(axis=0), 10, iterations=2000)
            det.append(A16.test_strate(kR, kV, w, b=1_000, graine=5000 + i)["p"] < 0.05 / 7)
        out.append((r, casses, float(np.mean(det))))
    # à forte perte (r = 0,45), une vingtaine d'épisodes de la strate sont déjà à 0 sous le nul : 10 cassures de plus ne
    # se voient pas au seuil 5 %/7 ; rapporté, pas exigé
    assert out[0][2] >= 0.8 and out[2][2] >= 0.8, out
    print("ok puissance au seuil 5 %/7 : " + " ; ".join(f"{c} cassures à r = {r} : {100 * d:.0f} %" for r, c, d in out))


def test_recomptage():
    """Le test de strate réécrit à part, avec les mêmes appels au générateur : mêmes T1, p, excès."""
    rng = np.random.default_rng(6)
    kR, X42, kV = simuler(rng, "beta", "amincissement", 0.8, casses=8)
    w = A16.npmle(kR + X42.sum(axis=0), 10)
    res = A16.test_strate(kR, kV, w, b=2_000, graine=77)
    s = kR == 5
    post = w * A16.GRILLE ** 5
    post /= post.sum()
    g = np.random.default_rng(77)
    u = g.random((2_000, int(s.sum())))
    p = A16.GRILLE[np.minimum(np.searchsorted(np.cumsum(post), u), 200)]
    r = (kV[s].mean() / 5) / p.mean(axis=1)
    assert (r[:, None] * p).max() <= 1
    k = g.binomial(5, r[:, None] * p)
    sim = (k == 0).sum(axis=1)
    obs = int((kV[s] == 0).sum())
    assert res["T1"] == obs and res["p"] == (1 + (sim >= obs).sum()) / 2_001 and abs(res["exces"] - (obs - sim.mean())) < 1e-9
    e2 = A16.e2_global(kR, kV, w, b=500)
    assert sum(e2["observe"]) == int((kV == 0).sum()) and e2["observe"][5] == obs
    print(f"ok recomptage indépendant : T1 {obs}, p {res['p']:.4f}, excès {res['exces']:+.1f} ; E2 cohérent")


def test_calibration_synthetique():
    rng = np.random.default_rng(7)
    kR, X42, _ = simuler(rng, "beta", "amincissement", 1.0)
    w = A16.npmle(kR + X42.sum(axis=0), 10)
    c = A16.calibration_synthetique(kR, X42, 0.85, w, graine=8, n_syn=300, b=500)
    assert c["taux_de_rejet"]["0.05"] <= 0.05 + 3 * np.sqrt(0.0475 / 300), c
    print(f"ok calibration synthétique sous une vraie perte proportionnelle : rejets {100 * c['taux_de_rejet']['0.05']:.1f} % "
          f"et {100 * c['taux_de_rejet']['0.05/7']:.1f} %, T1 moyen {c['T1_synthetique_moyen']:.2f} contre attendu {c['T1_attendu_moyen']:.2f}")


def test_refus(tmp):
    r = Path(tmp) / "r"
    for d, v in (("exp003", "fp32"), ("exp003", "all-4"), ("exp014-int4c-pusht", "fp32")):
        (r / d).mkdir(parents=True, exist_ok=True)
        for g in A16.G_V:
            shutil.copy(RES / d / f"{v}_s{g}.npz", r / d / f"{v}_s{g}.npz")
    assert A16.resultats_par_graine(r, "pusht", "exp003", "all-4", A16.G_V).shape == (5, 500)
    assert A16.resultats_par_graine(r, "pusht", "exp014-int4c-pusht", "fp32", A16.G_V).shape == (5, 500)

    def champ(d, nom, cle, val):
        def m(rr):
            x = dict(np.load(rr / d / nom))
            if val is None:
                del x[cle]
            else:
                x[cle] = val(x)
            np.savez_compressed(rr / d / nom, **x)
        return m
    cas = [("exp003", "all-4", "manquant", lambda rr: (rr / "exp003" / "all-4_s44.npz").unlink()),
           ("exp003", "all-4", "version ou graine", champ("exp003", "all-4_s43.npz", "seed", lambda x: np.int64(42))),
           ("exp003", "all-4", "version ou graine", champ("exp003", "all-4_s45.npz", "spec", lambda x: np.str_("all:3"))),
           ("exp003", "all-4", "arbre git modifié", champ("exp003", "all-4_s46.npz", "git_sale", lambda x: np.bool_(True))),
           ("exp003", "all-4", "episodes_idx différent", champ("exp003", "all-4_s42.npz", "episodes_idx", lambda x: x["episodes_idx"][::-1].copy())),
           ("exp003", "all-4", "pour un run PyTorch", champ("exp003", "all-4_s43.npz", "moteur", lambda x: np.str_("coreml-ne"))),
           ("exp014-int4c-pusht", "fp32", "moteur", champ("exp014-int4c-pusht", "fp32_s44.npz", "moteur", None)),
           ("exp014-int4c-pusht", "fp32", "modèle Core ML différent", champ("exp014-int4c-pusht", "fp32_s45.npz", "coreml_sha256", lambda x: np.str_("autre")))]
    for d, v, motif, m in cas:
        rr = Path(tmp) / "refus"
        shutil.rmtree(rr, ignore_errors=True)
        shutil.copytree(r, rr)
        m(rr)
        try:
            A16.resultats_par_graine(rr, "pusht", d, v, A16.G_V)
        except A.ControleEchoue as e:
            assert motif in str(e), (motif, str(e))
            continue
        raise AssertionError(f"accepté à tort : {motif}")
    print(f"ok refus ({len(cas)}, chacun avec son motif)")


def test_code():
    assert PROPRE, "arbre modifié : committer avant de lancer les tests"
    A16.verifier_code(TETE)
    print(f"ok code : fichiers d'analyse identiques à {TETE[:7]}")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        test_npmle()
        test_amincir()
        test_recomptage()
        test_calibration_synthetique()
        test_refus(tmp)
        test_puissance()
        test_calibration()
        test_code()
    print("tous les tests passent")
