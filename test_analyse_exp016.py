"""Tests de analyse_exp016.py, sur données simulées (aucun résultat réel lu avant le commit de l'analyse) et sur de vrais
fichiers d'EXP-003 copiés pour les refus. 1. L'ajustement bêta-binomial retrouve une loi connue. 2. Calibration : sous
le nul d'amincissement simulé, le test rejette au plus environ 5 % du temps. 3. Puissance : des cassures plantées sont
trouvées. 4. T1 et la p recomptés indépendamment. 5. Le réglage de r et de delta atteint la réussite visée. 6. Refus
des fichiers non conformes, chacun avec son motif. À lancer sur un arbre commité.

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


def simuler(rng, n_ep=500, a=0.3, b=0.2, r=0.8, casses=0):
    p = rng.beta(a, b, n_ep)
    kR = rng.binomial(5, p)
    kV = rng.binomial(5, r * p)
    if casses:                                          # cassures plantées parmi les épisodes toujours réussis
        idx = rng.choice(np.flatnonzero(kR == 5), casses, replace=False)
        kV[idx] = 0
    return kR, kV


def test_ajustement():
    rng = np.random.default_rng(1)
    p = rng.beta(0.3, 0.2, 20_000)
    a, b = A16.ajuster_beta(rng.binomial(5, p))
    assert abs(a / 0.3 - 1) < 0.1 and abs(b / 0.2 - 1) < 0.1, (a, b)
    print(f"ok ajustement bêta-binomial : alpha {a:.3f} (0,3), beta {b:.3f} (0,2)")


def test_reglage():
    rng = np.random.default_rng(2)
    p = rng.beta(0.3, 0.2, size=(50, 500))
    for nul in ("amincissement", "logistique"):
        for cible in (0.3, 0.6, 0.8):
            q = A16.regler(p, cible, nul)
            assert np.allclose(q.mean(axis=1), cible, atol=1e-6) and (q >= 0).all() and (q <= 1).all(), (nul, cible)
    print("ok réglage de r et de delta : réussite moyenne atteinte à 1e-6 près, q dans [0, 1]")


def test_calibration():
    rng = np.random.default_rng(3)
    rejets, n = 0, 300
    for i in range(n):
        kR, kV = simuler(rng)
        a, b = A16.ajuster_beta(kR)
        sim, _ = A16.tirages_nuls(kR, kV, a, b, "amincissement", b=2_000, graine=100 + i)
        p = (1 + (sim >= A16.t1(kR, kV)).sum()) / (len(sim) + 1)
        rejets += p < 0.05
    taux = rejets / n
    assert taux <= 0.08, taux                       # 5 % attendu, erreur de Monte-Carlo environ 1,3 %
    print(f"ok calibration sous le nul simulé : {100 * taux:.1f} % de rejets à 5 % ({n} jeux de données)")


def test_puissance_et_recomptage():
    rng = np.random.default_rng(4)
    kR, kV = simuler(rng, r=0.85, casses=15)
    a, b = A16.ajuster_beta(kR)
    res = A16.etudier(kR, kV, a, b, kV42=rng.binomial(5, rng.beta(a + kR, b + 5 - kR)))
    assert res["T1"] == int(((kR == 5) & (kV == 0)).sum()) >= 15
    sim, _ = A16.tirages_nuls(kR, kV, a, b, "amincissement")
    assert res["amincissement"]["p"] == (1 + int((sim >= res["T1"]).sum())) / 10_001
    assert res["amincissement"]["p"] < 0.001 and res["logistique"]["p"] < 0.001, res
    assert sum(res["E2_jamais_reussis_par_kR"]) == int((kV == 0).sum())
    assert res["calibration_kR5"]["observe"][0] == res["T1"] and sum(res["calibration_kR5"]["observe"]) == int((kR == 5).sum())
    kR0, kV0 = simuler(rng, r=0.85)                     # même chose sans cassure : pas de rejet attendu au seuil de 1 %
    a0, b0 = A16.ajuster_beta(kR0)
    assert A16.etudier(kR0, kV0, a0, b0)["amincissement"]["p"] > 0.01
    print(f"ok puissance : 15 cassures plantées, T1 {res['T1']}, attendu {res['amincissement']['T1_attendu']:.2f}, "
          f"p {res['amincissement']['p']:.4f} ; recomptage de T1, de la p et des descriptifs")


def test_refus(tmp):
    r = Path(tmp) / "r"
    (r / "exp003").mkdir(parents=True)
    for g in A16.G_V:
        for v in ("fp32", "all-4"):
            shutil.copy(RES / "exp003" / f"{v}_s{g}.npz", r / "exp003" / f"{v}_s{g}.npz")
    S = A16.comptes(r, "pusht", "exp003", "all-4", A16.G_V)
    assert S.shape == (500,) and S.max() <= 5

    def champ(nom, cle, val):
        def m(rr):
            x = dict(np.load(rr / "exp003" / nom))
            x[cle] = val(x)
            np.savez_compressed(rr / "exp003" / nom, **x)
        return m
    cas = [("manquant", lambda rr: (rr / "exp003" / "all-4_s44.npz").unlink()),
           ("version ou graine", champ("all-4_s43.npz", "seed", lambda x: np.int64(42))),
           ("version ou graine", champ("all-4_s45.npz", "spec", lambda x: np.str_("all:3"))),
           ("arbre git modifié", champ("all-4_s46.npz", "git_sale", lambda x: np.bool_(True))),
           ("episodes_idx différent", champ("all-4_s42.npz", "episodes_idx", lambda x: x["episodes_idx"][::-1].copy())),
           ("init_state différent", champ("all-4_s43.npz", "init_state", lambda x: x["init_state"] + 1))]
    for motif, m in cas:
        rr = Path(tmp) / "refus"
        shutil.rmtree(rr, ignore_errors=True)
        shutil.copytree(r, rr)
        m(rr)
        try:
            A16.comptes(rr, "pusht", "exp003", "all-4", A16.G_V)
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
        test_ajustement()
        test_reglage()
        test_puissance_et_recomptage()
        test_calibration()
        test_refus(tmp)
        test_code()
    print("tous les tests passent")
