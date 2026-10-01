"""Tests de analyse_exp010.py sur les vrais fichiers d'EXP-005 (copiés dans un dossier temporaire) et des fichiers
« GPTQ » fabriqués à effet connu : copies de la pleine précision (écart nul) ou du 6 bits (écart -1,26 point).
p recomptés en numpy, séquence fixée, refus. À lancer sur un arbre commité.

Usage :
    python test_analyse_exp010.py
"""
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

import analyse_exp003 as A
import analyse_exp010 as A10

RES = A10.ICI.parent / "resultats"
SIGNES = np.random.default_rng(0).choice(np.array([-1.0, 1.0]), size=(10_000, A10.N))
TETE = subprocess.run(["git", "-C", str(A10.ICI), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
PROPRE = subprocess.run(["git", "-C", str(A10.ICI), "status", "--porcelain", "--untracked-files=no"],
                        capture_output=True, text=True).stdout.strip() == ""


def fabriquer(racine, sources):
    """sources : {version GPTQ : version d'EXP-005 copiée}."""
    r = Path(racine)
    shutil.copytree(RES / "exp005", r / "exp005", ignore=shutil.ignore_patterns("*.log"))
    (r / "exp009-calib-pusht").mkdir()
    os.symlink(RES / A10.CALIB, r / A10.CALIB)
    (r / "exp010-repro").mkdir()
    shutil.copy(r / "exp005" / "fp32_s52.npz", r / "exp010-repro" / "fp32_s52.npz")
    (r / "exp010").mkdir()
    proto = json.loads((r / "exp005" / "meta.json").read_text())["protocole"]
    proto["gptq_calib"] = {"fichier": "calib.pt", "sha256": hashlib.sha256((r / A10.CALIB).read_bytes()).hexdigest()}
    (r / "exp010" / "meta.json").write_text(json.dumps({"protocole": proto, "git": TETE, "git_sale": False}))
    for v, src in sources.items():
        for g in A10.GRAINES:
            x = dict(np.load(r / "exp005" / f"{src}_s{g}.npz"))
            x.update(spec=np.str_(v), git=np.str_(TETE), n_couches=np.int64(109))
            np.savez_compressed(r / "exp010" / f"{v.replace(':', '-')}_s{g}.npz", **x)
    return r


def K(r, d, nom):
    return sum(np.load(Path(r) / d / f"{nom}_s{g}.npz")["success"].astype(float) for g in A10.GRAINES)


def p_numpy(y):
    return (1 + int((SIGNES @ y >= y.sum()).sum())) / 10_001


def test_sequences(tmp):
    r = fabriquer(Path(tmp) / "a", {"all:4:gptq": "fp32", "all:3:gptq": "all-6"})
    res = A10.analyser(r, verifier=False)
    f = K(r, "exp005", "fp32")
    for nom, v, m in A10.SEQUENCE:
        x = K(r, "exp010", v.replace(":", "-")) - f
        assert res["tests"][nom]["p"] == p_numpy(100 // (5 * m) * x + 1), nom
    assert [res["tests"][n]["verdict"] for n in ("T1", "T2", "T3", "T4")] == ["établie", "établie", "établie", "non établie"], res["tests"]
    assert abs(res["descriptifs"]["all:4:gptq"]["ecart_reussite"]["moyenne"]) < 1e-12 and "T4" in A10.rapport(res)
    r2 = fabriquer(Path(tmp) / "b", {"all:4:gptq": "all-6", "all:3:gptq": "fp32"})
    res2 = A10.analyser(r2, verifier=False)
    assert [res2["tests"][n]["verdict"] for n in ("T1", "T2", "T3", "T4")] == \\
        ["établie", "non établie", "non testée (séquence arrêtée)", "non testée (séquence arrêtée)"], res2["tests"]
    print("ok séquence : p recomptés en numpy ; T1 à T3 établies et T4 non (6 bits, -1,26 point) ; arrêt après un échec")
    return r


def test_refus(tmp, r):
    def essai(motif, modifier):
        rr = Path(tmp) / "refus"
        shutil.rmtree(rr, ignore_errors=True)
        shutil.copytree(r, rr, symlinks=True)
        modifier(rr)
        try:
            A10.analyser(rr, verifier=False)
        except A.ControleEchoue as e:
            assert motif in str(e), (motif, str(e))
            return
        raise AssertionError(f"accepté à tort : {motif}")

    def champ(nom, cle, val):
        def m(rr):
            x = dict(np.load(rr / "exp010" / nom))
            x[cle] = val(x)
            np.savez_compressed(rr / "exp010" / nom, **x)
        return m

    def meta(f):
        def m(rr):
            p = rr / "exp010" / "meta.json"
            x = json.loads(p.read_text())
            f(x["protocole"])
            p.write_text(json.dumps(x))
        return m
    cas = [("manquant", lambda rr: (rr / "exp010" / "all-3-gptq_s54.npz").unlink()),
           ("reproduction", lambda rr: shutil.copy(rr / "exp005" / "all-8_s52.npz", rr / "exp010-repro" / "fp32_s52.npz")),
           ("calibration différente", meta(lambda p: p["gptq_calib"].update(sha256="0" * 64))),
           ("protocole différent", meta(lambda p: p.update(n=1999))),
           ("différent d'EXP-005", champ("all-4-gptq_s53.npz", "goal_state", lambda x: x["goal_state"] + 1)),
           ("couches quantifiées", champ("all-3-gptq_s52.npz", "n_couches", lambda x: np.int64(30))),
           ("plusieurs commits", champ("all-4-gptq_s55.npz", "git", lambda x: np.str_("b" * 40)))]
    for motif, m in cas:
        essai(motif, m)
    print(f"ok contrôles bloquants ({len(cas)} refus, chacun avec son motif)")


def test_code():
    assert PROPRE, "arbre modifié : committer avant de lancer les tests"
    A10.verifier_code(TETE)
    print(f"ok code : fichiers d'analyse et de calcul identiques à {TETE[:7]}")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp, np.errstate(all="ignore"):
        r = test_sequences(tmp)
        test_refus(tmp, r)
        test_code()
    print("tous les tests passent")
