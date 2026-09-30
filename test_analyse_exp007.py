"""Tests de analyse_exp007.py : quatre conditions simulées (simulateur d'EXP-003, mêmes épisodes), p et Holm
recomptés en numpy pur, refus des fichiers incohérents, code vérifié au commit. À lancer sur un arbre commité.

Usage :
    python test_analyse_exp007.py
"""
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

import analyse_exp003 as A
import analyse_exp007 as A7
import test_analyse_exp003 as T

SIGNES = np.random.default_rng(0).choice(np.array([-1.0, 1.0]), size=(10_000, A7.N7))
TETE = subprocess.run(["git", "-C", str(A7.ICI), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
PROPRE = subprocess.run(["git", "-C", str(A7.ICI), "status", "--porcelain", "--untracked-files=no"],
                        capture_output=True, text=True).stdout.strip() == ""


def simuler(racine, effets):
    """effets : {condition: {version: (baisse de logit, facteur des marges)}} ; même graine, donc mêmes épisodes."""
    sauve = A.N, A.VERSIONS, A.GA
    A.N, A.VERSIONS, A.GA = A7.N7, A7.VERSIONS, A7.GRAINES
    try:
        for c, (dossier, rh, it) in A7.CONDITIONS.items():
            d = Path(racine) / dossier
            T.simuler(d, effets=effets.get(c, {}), graine=5, divergents=(), paquet_reussi=())
            for f in d.glob("*.npz"):
                if int(np.load(f)["seed"]) not in A7.GRAINES:
                    f.unlink()
                elif c != "r25":
                    x = dict(np.load(f))
                    x["git"] = TETE
                    np.savez_compressed(f, **x)
            m = json.loads((d / "meta.json").read_text())
            m["protocole"]["config"] = {"plan_config": {"receding_horizon": rh}, "solver": {"n_steps": it}}
            if c != "r25":
                m["protocole"]["planification"] = {"receding_horizon": rh, "cem_n_steps": it}
            (d / "meta.json").write_text(json.dumps(m))
    finally:
        A.N, A.VERSIONS, A.GA = sauve
    (Path(racine) / "exp007-sanite-r25").mkdir()
    shutil.copy(Path(racine) / "exp003" / "all-4_s42.npz", Path(racine) / "exp007-sanite-r25" / "all-4_s42.npz")
    fin = Path(racine) / "fin"
    fin.mkdir()
    shutil.copy(Path(racine) / "exp007-r5c6" / "all-4_s42.npz", fin / "all-4_s42.npz")
    return fin


def p_numpy(x):
    return (1 + int((SIGNES @ x >= x.sum()).sum())) / 10_001


def K(racine, dossier, v):
    return np.stack([np.load(Path(racine) / dossier / f"{v.replace(':', '-')}_s{g}.npz")["success"]
                     for g in A7.GRAINES]).sum(0).astype(float)


def test_tests(tmp):
    base = {"all:4": (1.5, 1.0)}
    r = Path(tmp) / "effet"
    fin = simuler(r, {"r25": base, "r5": {"all:4": (0.5, 1.0)}, "r5c6": {"all:4": (0.9, 1.0)}})
    res = A7.analyser(r, fin, verifier=False)
    k = {(c, v): K(r, d, v) for c, (d, *_) in A7.CONDITIONS.items() for v in A7.VERSIONS}
    xs = {"Q4a": k[("r5", "all:4")] - k[("r25", "all:4")],
          "Q4b": (k[("r5", "all:4")] - k[("r25", "all:4")]) - (k[("r5", "fp32")] - k[("r25", "fp32")]),
          "Q4c": k[("r5c6", "all:4")] - k[("r25", "all:4")]}
    ps = [p_numpy(x) for x in xs.values()]
    ordre, holm, cour = np.argsort(ps), [0.0] * 3, 0.0
    for rang, i in enumerate(ordre):
        cour = max(cour, min(1.0, (3 - rang) * ps[i]))
        holm[i] = cour
    for (n, x), p, q in zip(xs.items(), ps, holm):
        t = res["tests"][n]
        assert t["p"] == p and t["somme"] == int(x.sum()) and abs(t["p_holm"] - q) < 1e-12, (n, t, p)
        assert t["verdict"] == "soutenue", (n, t)
    assert "Q4c" in A7.rapport(res) and res["controles"]["fichiers"] == 18
    r0 = Path(tmp) / "nul"
    fin0 = simuler(r0, {c: base for c in A7.CONDITIONS})
    res0 = A7.analyser(r0, fin0, verifier=False)
    assert all(t["somme"] == 0 and t["p"] == 1.0 and t["verdict"] == "non soutenue" for t in res0["tests"].values())
    print("ok tests : p par inversion de signe et Holm égaux au recomptage numpy, effets retrouvés, rien sans effet")
    return r, fin


def test_refus(tmp, r, fin):
    def essai(motif, modifier):
        rr, ff = Path(tmp) / "refus", Path(tmp) / "refus-fin"
        shutil.rmtree(rr, ignore_errors=True)
        shutil.rmtree(ff, ignore_errors=True)
        shutil.copytree(r, rr)
        shutil.copytree(fin, ff)
        modifier(rr, ff)
        try:
            A7.analyser(rr, ff, verifier=False)
        except A.ControleEchoue as e:
            assert motif in str(e), (motif, str(e))
            return
        raise AssertionError(f"accepté à tort : {motif}")

    def meta(dossier, f):
        def m(rr, ff):
            p = rr / dossier / "meta.json"
            x = json.loads(p.read_text())
            f(x["protocole"])
            p.write_text(json.dumps(x))
        return m

    def champ(dossier, nom, cle, val):
        def m(rr, ff):
            x = dict(np.load(rr / dossier / nom))
            x[cle] = val(x)
            np.savez_compressed(rr / dossier / nom, **x)
        return m

    cas = [
        ("planification", meta("exp007-r5", lambda p: p["config"]["plan_config"].update(receding_horizon=5))),
        ("manquant", lambda rr, ff: (rr / "exp007-r5" / "fp32_s43.npz").unlink()),
        ("différent d'EXP-003", champ("exp007-r5c6", "all-4_s44.npz", "start_steps", lambda x: x["start_steps"][::-1].copy())),
        ("plusieurs commits", champ("exp007-r5c6", "all-4_s43.npz", "git", lambda x: np.str_("b" * 40))),
        ("couches quantifiées", champ("exp007-r5", "fp32_s42.npz", "n_couches", lambda x: np.int64(109))),
        ("sanité de clôture", lambda rr, ff: shutil.copy(rr / "exp007-r5" / "all-4_s42.npz", ff / "all-4_s42.npz")),
        ("reproduction d'EXP-003", lambda rr, ff: shutil.copy(rr / "exp007-r5" / "all-4_s42.npz",
                                                              rr / "exp007-sanite-r25" / "all-4_s42.npz")),
    ]
    for motif, m in cas:
        essai(motif, m)
    print(f"ok contrôles bloquants ({len(cas)} refus, chacun avec son motif)")


def test_code():
    assert PROPRE, "arbre modifié : committer avant de lancer les tests"
    A7.verifier_code(TETE)
    print(f"ok code : fichiers d'analyse et de calcul identiques à {TETE[:7]}")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp, np.errstate(all="ignore"):
        r, fin = test_tests(tmp)
        test_refus(tmp, r, fin)
        test_code()
    print("tous les tests passent")
