"""Tests de analyse_exp008.py : grilles TwoRoom simulées (EXP-006 et EXP-008, mêmes épisodes), réussites et écarts
recalculés à part, refus des fichiers incohérents, code vérifié au commit. À lancer sur un arbre commité.

Usage :
    python test_analyse_exp008.py
"""
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

import analyse_exp003 as A
import analyse_exp006 as A6
import analyse_exp008 as A8
import test_analyse_exp006 as T6

TETE = subprocess.run(["git", "-C", str(A8.ICI), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
PROPRE = subprocess.run(["git", "-C", str(A8.ICI), "status", "--porcelain", "--untracked-files=no"],
                        capture_output=True, text=True).stdout.strip() == ""


def simuler(racine):
    r = Path(racine)
    sauve = A6.VERSIONS6, dict(A6.COUCHES)
    A6.COUCHES["encpred"] = 103
    try:
        T6.simuler(r / "exp006")
        A6.VERSIONS6 = ("fp32", "all:4") + A8.VERSIONS8          # le simulateur écrit aussi fp32 et all:4, retirés ensuite
        T6.simuler(r / "exp008", effets={**{v: -0.5 for v in A8.VERSIONS8}, "fp32": 0.0, "all:4": 0.0}, sha=TETE)
    finally:
        A6.VERSIONS6 = sauve[0]
        A6.COUCHES.clear(); A6.COUCHES.update(sauve[1])
    for f in list((r / "exp008").glob("fp32_*.npz")) + list((r / "exp008").glob("all-4_*.npz")):
        f.unlink()
    shutil.rmtree(r / "exp008-fin")
    (r / "exp008-repro").mkdir()
    shutil.copy(r / "exp006" / "all-4_s42.npz", r / "exp008-repro" / "all-4_s42.npz")
    return r


def test_valeurs(tmp):
    r = simuler(Path(tmp) / "g")
    res = A8.analyser(r, verifier=False)
    K = lambda d, v: np.stack([np.load(r / d / f"{v.replace(':', '-')}_s{g}.npz")["success"] for g in A6.GA]).sum(0).astype(float)
    f = K("exp006", "fp32")
    boot = A.Bootstrap(n=A6.N6)
    for v in A8.VERSIONS8:
        x = (K("exp008", v) - f) / 5
        d = res["descriptifs"][v]
        assert abs(d["reussite"] - K("exp008", v).mean() / 5) < 1e-12 and abs(d["ecart_points"]["moyenne"] - x.mean()) < 1e-12
        assert d["ecart_points"]["bas"] == float(np.percentile(x[boot.idx].mean(1), 2.5)), v
    assert res["controles"]["fichiers"] == 35 and "encpred:4" in A8.rapport(res)
    print("ok valeurs : réussites, écarts et bornes bootstrap recalculés à part")
    return r


def test_refus(tmp, r):
    def essai(motif, modifier):
        rr = Path(tmp) / "refus"
        shutil.rmtree(rr, ignore_errors=True)
        shutil.copytree(r, rr)
        modifier(rr)
        try:
            A8.analyser(rr, verifier=False)
        except A.ControleEchoue as e:
            assert motif in str(e), (motif, str(e))
            return
        raise AssertionError(f"accepté à tort : {motif}")

    def champ(nom, cle, val):
        def m(rr):
            x = dict(np.load(rr / "exp008" / nom))
            x[cle] = val(x)
            np.savez_compressed(rr / "exp008" / nom, **x)
        return m
    cas = [("manquant", lambda rr: (rr / "exp008" / "autres-3_s44.npz").unlink()),
           ("reproduction d'EXP-006", lambda rr: shutil.copy(rr / "exp006" / "all-3_s42.npz", rr / "exp008-repro" / "all-4_s42.npz")),
           ("différent d'EXP-006", champ("encoder-3_s43.npz", "goal_state", lambda x: x["goal_state"] + 1)),
           ("plusieurs commits", champ("all-5_s42.npz", "git", lambda x: np.str_("b" * 40))),
           ("couches quantifiées", champ("encpred-4_s45.npz", "n_couches", lambda x: np.int64(109)))]
    for motif, m in cas:
        essai(motif, m)
    print(f"ok contrôles bloquants ({len(cas)} refus, chacun avec son motif)")


def test_code():
    assert PROPRE, "arbre modifié : committer avant de lancer les tests"
    A8.verifier_code(TETE)
    print(f"ok code : fichiers d'analyse et de calcul identiques à {TETE[:7]}")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp, np.errstate(all="ignore"):
        r = test_valeurs(tmp)
        test_refus(tmp, r)
        test_code()
    print("tous les tests passent")
