"""Tests de analyse_exp005.py sur des grilles simulées de 1 000 épisodes (simulateur d'EXP-003).

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

import analyse_exp003 as A
import analyse_exp005 as A5
import test_analyse_exp003 as T


def grille(tmp, nom, **kw):
    """Grille simulée : fp32 et all:6 × graines 42 à 46, meta.json d'EXP-005, liste d'exclusion, sanité de clôture."""
    d, fin, excl = Path(tmp) / f"{nom}-exp005", Path(tmp) / f"{nom}-fin", Path(tmp) / f"{nom}-exclus.json"
    n0, v0 = A.N, A.VERSIONS
    A.N, A.VERSIONS = A5.N5, A5.VERSIONS5
    try:
        T.simuler(d, **kw)
    finally:
        A.N, A.VERSIONS = n0, v0
    for g in A.GB:                                           # pas de réplique dans EXP-005
        (d / f"fp32_s{g}.npz").unlink()
    exclus = [[999_999, 1], [999_998, 2]]                    # couples absents du tirage simulé
    excl.write_text(json.dumps({"protocole": {"episodes_idx": [a for a, _ in exclus], "start_steps": [b for _, b in exclus]}}))
    m = json.loads((d / "meta.json").read_text())
    m["protocole"].update(graine_episodes=A5.GRAINE_EPISODES, exclus=exclus)
    (d / "meta.json").write_text(json.dumps(m))
    fin.mkdir()
    shutil.copy(d / "meta.json", fin / "meta.json")
    shutil.copy(d / "all-6_s42.npz", fin / "all-6_s42.npz")
    return d, fin, excl


def sans_verif_code(fonction):
    vrai = A5.verifier_code5
    A5.verifier_code5 = lambda sha: None
    try:
        return fonction()
    finally:
        A5.verifier_code5 = vrai


def analyser(d, fin, excl):
    return sans_verif_code(lambda: A5.analyser(d, fin, [excl], effectifs=None))


def test_code():
    for f, empreinte in (("run_grid.py", A5.RUN_GRID_SHA256), ("quant.py", A5.QUANT_SHA256)):
        assert hashlib.sha256((A.ICI / f).read_bytes()).hexdigest() == empreinte, f"{f} a changé"
    for sha, motif in (("c5703dbbacb1537be9cc03377a3a922f4a81c1ed", "run_grid.py d'EXP-005"), ("0" * 40, "inconnu")):
        try:
            A5.verifier_code5(sha)
            raise AssertionError(sha)
        except A.ControleEchoue as err:
            assert motif in str(err), err
    tete = subprocess.run(["git", "-C", str(A.ICI), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    A5.verifier_code5(tete)
    print(f"ok code : empreintes de run_grid.py et quant.py, c5703db et commit inconnu refusés, {tete[:7]} accepté")


def test_scenarios(tmp):
    # 1. effet cherché : réussite inchangée, échecs plus loin du but (marges × 1,5)
    d, fin, excl = grille(tmp, "effet", effets={"all:6": (0.0, 1.5)}, graine=11, divergents=())
    r = analyser(d, fin, excl)
    h = r["H6"]
    assert h["verdict"] == "établie", h
    K = {v: sum(np.load(d / f"{v.replace(':', '-')}_s{g}.npz")["success"].astype(np.int64) for g in A.GA) for v in A5.VERSIONS5}
    x = K["all:6"] - K["fp32"]
    dep = A.depart({("fp32", 42): dict(np.load(d / "fp32_s42.npz"))})
    M3 = {v: np.stack([A.controler_course(dict(np.load(d / f"{v.replace(':', '-')}_s{g}.npz")), v)[1] for g in A.GA])
          for v in A5.VERSIONS5}
    w = (A.lg(M3["all:6"]).mean(0) - A.lg(M3["fp32"]).mean(0))[dep["bouge"]]
    inv = A.Inversion(n=A5.N5)
    assert h["p_bas"] == inv.p(10 * x + 1, "plus") and h["p_haut"] == inv.p(10 * x - 1, "moins")
    assert h["p_b"] == inv.p(w, "plus") and h["p"] == max(h["p_bas"], h["p_haut"], h["p_b"])
    assert h["episodes_bouge"] == int(dep["bouge"].sum()) < A5.N5
    txt = A5.rapport(r)
    assert "établie" in txt and not re.search(r"\bnan\b|\binf\b", txt.lower())
    print("ok effet cherché : H6 établie ; p_bas, p_haut, p_b recalculées à la main")

    # 2. la réussite baisse aussi : H6 ne doit pas passer, même si la pose se dégrade
    d2, fin2, excl2 = grille(tmp, "baisse", effets={"all:6": (1.2, 1.5)}, graine=12, divergents=())
    h = analyser(d2, fin2, excl2)["H6"]
    assert h["verdict"] == "non établie" and h["p_bas"] > 0.5 and h["p_b"] < 0.05, h
    print("ok réussite en baisse : H6 non établie (équivalence refusée) malgré la pose dégradée")

    # 3. aucun effet : versions identiques épisode par épisode
    d3, fin3, excl3 = grille(tmp, "nul", graine=13, divergents=(), identiques=("all:6",), paquet_reussi=())
    h = analyser(d3, fin3, excl3)["H6"]
    assert h["delta_reussite"] == 0 and h["delta_log_M3_bouge"] == 0 and h["p_b"] == 1.0 and h["verdict"] == "non établie", h
    print("ok sans effet : écarts nuls, p_b = 1, H6 non établie")
    return d, fin, excl


def test_refus(tmp, d, fin, excl):
    def doit_echouer(nom, alterer, motif):
        e, ef, ex = Path(tmp) / f"r-{nom}", Path(tmp) / f"r-{nom}-fin", Path(tmp) / f"r-{nom}-exclus.json"
        shutil.copytree(d, e)
        shutil.copytree(fin, ef)
        shutil.copy(excl, ex)
        alterer(e, ef, ex)
        try:
            analyser(e, ef, ex)
        except A.ControleEchoue as err:
            assert motif in str(err), (nom, str(err))
            print(f"   refusé comme attendu ({nom}) : {err}")
            return
        raise AssertionError(f"{nom} : contrôle non déclenché")

    def meta(dossier, **champs):
        m = json.loads((dossier / "meta.json").read_text())
        m["protocole"].update(champs)
        (dossier / "meta.json").write_text(json.dumps(m))

    def recouvrement(e, ef, ex):                             # un épisode de la grille dans la liste d'exclusion
        p = json.loads((e / "meta.json").read_text())["protocole"]
        paire = [p["episodes_idx"][0], p["start_steps"][0]]
        x = json.loads(ex.read_text())
        x["protocole"]["episodes_idx"].append(paire[0]); x["protocole"]["start_steps"].append(paire[1])
        ex.write_text(json.dumps(x))
        for dd in (e, ef):
            meta(dd, exclus=json.loads((dd / "meta.json").read_text())["protocole"]["exclus"] + [paire])

    def modifier(dossier, f, **champs):
        x = dict(np.load(dossier / f))
        x.update(champs)
        np.savez_compressed(dossier / f, **x)

    def inverser(dossier, f):
        x = dict(np.load(dossier / f))
        x["success"][0] = ~x["success"][0]
        np.savez_compressed(dossier / f, **x)

    doit_echouer("recouvrement", recouvrement, "épisodes en commun")
    doit_echouer("graine de tirage", lambda e, ef, ex: meta(e, graine_episodes=42), "graine de tirage")
    doit_echouer("liste d'exclusion", lambda e, ef, ex: meta(e, exclus=[]), "liste d'exclusion différente")
    doit_echouer("fichier manquant", lambda e, ef, ex: (e / "all-6_s44.npz").unlink(), "fichiers manquants")
    doit_echouer("couches", lambda e, ef, ex: modifier(e, "all-6_s45.npz", n_couches=73), "73 couches")
    doit_echouer("sanité de clôture", lambda e, ef, ex: inverser(ef, "all-6_s42.npz"), "sanité de clôture : success diffère")
    doit_echouer("torch de la sanité", lambda e, ef, ex: (ef / "meta.json").write_text(
        json.dumps(dict(json.loads((ef / "meta.json").read_text()), torch="2.15.0"))), "protocole ou torch différent")
    print("ok contrôles bloquants d'EXP-005 (7 refus, chacun avec son motif)")


if __name__ == "__main__":
    test_code()
    with tempfile.TemporaryDirectory() as tmp:
        d, fin, excl = test_scenarios(tmp)
        test_refus(tmp, d, fin, excl)
    print("tous les tests passent")
