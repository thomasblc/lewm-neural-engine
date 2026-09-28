"""Tests de analyse_exp004.py sur des grilles simulées (EXP-003 et EXP-004 avec le même hasard par épisode).

Le simulateur d'EXP-003 est réutilisé tel quel, les versions d'EXP-004 ajoutées à la fin de sa liste : les
fichiers simulés d'EXP-003 restent les mêmes, ceux d'EXP-004 partagent la géométrie et le hasard.

Usage :
    python test_analyse_exp004.py
"""
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

import analyse_exp003 as A
import analyse_exp004 as A4
import test_analyse_exp003 as T


def grilles(tmp, nom, **kw):
    """Écrit une grille simulée complète, puis la répartit en EXP-003, EXP-004 et sanité."""
    tout, d3, d4, san = (Path(tmp) / f"{nom}-{x}" for x in ("tout", "exp003", "exp004", "sanite"))
    versions, couches = A.VERSIONS, dict(A.COUCHES)
    A.VERSIONS = versions + A4.NOUVELLES
    A.COUCHES.update({"autres": 6, "encpred": 103})
    try:
        T.simuler(tout, **kw)
    finally:
        A.VERSIONS = versions
        A.COUCHES.clear()
        A.COUCHES.update(couches)
    for d in (d3, d4, san):
        d.mkdir()
    for f in tout.glob("*.npz"):
        spec = str(np.load(f)["spec"])
        shutil.copy(f, (d4 if spec in A4.NOUVELLES else d3) / f.name)
    for d in (d3, d4):
        shutil.copy(tout / "meta.json", d / "meta.json")
    shutil.copy(d3 / "all-4_s42.npz", san / "all-4_s42.npz")
    return d3, d4, san


def sans_verif_code(fonction):
    """Les fichiers simulés portent le commit c5703db : on neutralise la vérification du quant.py d'EXP-004."""
    vrai = A4.verifier_code4
    A4.verifier_code4 = lambda sha: None
    try:
        return fonction()
    finally:
        A4.verifier_code4 = vrai


def test_verifier_code4():
    try:
        A4.verifier_code4("c5703dbbacb1537be9cc03377a3a922f4a81c1ed")
        raise AssertionError("c5703db accepté")
    except A.ControleEchoue as err:
        assert "quant.py d'EXP-004" in str(err), err
    try:
        A4.verifier_code4("0" * 40)
        raise AssertionError("commit inconnu accepté")
    except A.ControleEchoue as err:
        assert "inconnu" in str(err)
    tete = subprocess.run(["git", "-C", str(A.ICI), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    try:
        A4.verifier_code4(tete)
        print(f"ok vérification du code : c5703db et commit inconnu refusés, {tete[:7]} accepté")
    except A.ControleEchoue:
        print(f"ok vérification du code : c5703db et commit inconnu refusés ({tete[:7]} n'a pas encore le quant.py figé)")


def test_bout_en_bout(tmp):
    effets = {"all:4": (1.2, 1.0), "encoder:4": (0.5, 1.0), "autres:4": (0.6, 1.0), "encpred:4": (0.5, 1.0),
              "all:4:g64": (0.6, 1.0), "all:3": (3.0, 1.0), "all:3:g64": (2.0, 1.0), "autres:3": (1.5, 1.0)}
    d3, d4, san = grilles(tmp, "effets", effets=effets, graine=5)
    r = sans_verif_code(lambda: A4.analyser(d3, d4, san, effectifs=None, verifier_exp003=False))
    for nom in ("Q3a", "Q3b", "Q5a", "Q5b"):
        assert r["tests"][nom]["verdict"] == "soutenue", (nom, r["tests"][nom])
    assert r["tests"]["Q3a"]["somme"] < 0 < r["tests"]["Q3b"]["somme"]
    # additivité recalculée à la main depuis les fichiers
    K = {}
    for v in ("fp32", "all:4", "encoder:4", "predictor:4"):
        K[v] = sum(np.load(d3 / f"{v.replace(':', '-')}_s{g}.npz")["success"].astype(int) for g in A.GA)
    for v in ("autres:4", "encpred:4"):
        K[v] = sum(np.load(d4 / f"{v.replace(':', '-')}_s{g}.npz")["success"].astype(int) for g in A.GA)
    i_main = ((K["all:4"] - K["encoder:4"] - K["predictor:4"] - K["autres:4"] + 2 * K["fp32"]) / 5).mean()
    cle = next(k for k in r["additivite"] if k.startswith("I"))
    assert abs(r["additivite"][cle]["moyenne"] - i_main) < 1e-12
    assert r["controles"]["divergences"] == {v: 0 for v in A4.NOUVELLES}
    txt = A4.rapport(r)
    assert "Q5b" in txt and not re.search(r"\bnan\b|\binf\b", txt.lower())
    print("ok bout en bout, effets connus : Q3a, Q3b, Q5a, Q5b retrouvés dans le bon sens, additivité exacte")

    iden = ("autres:4", "encpred:4", "all:4:g64", "all:4", "autres:3", "all:3:g64", "all:3")
    nul = {"all:4": (1.0, 1.0), "encpred:4": (1.0, 1.0), "all:4:g64": (1.0, 1.0), "all:3": (2.0, 1.0),
           "all:3:g64": (2.0, 1.0)}
    d3n, d4n, sann = grilles(tmp, "nul", effets=nul, graine=6, identiques=iden)
    r = sans_verif_code(lambda: A4.analyser(d3n, d4n, sann, effectifs=None, verifier_exp003=False))
    for nom in ("Q3a", "Q3b", "Q5a", "Q5b"):
        t = r["tests"][nom]
        assert t["somme"] == 0 and t["p"] == 1.0 and t["verdict"] == "non soutenue", (nom, t)
    print("ok bout en bout, sans effet : les 4 différences sont nulles, p = 1, rien n'est soutenu")
    return d3, d4, san


def test_refus(tmp, d3, d4, san):
    def doit_echouer(nom, alterer, motif):
        e3, e4, es = (Path(tmp) / f"{nom}-{x}" for x in ("3", "4", "s"))
        for src, dst in ((d3, e3), (d4, e4), (san, es)):
            shutil.copytree(src, dst)
        alterer(e3, e4, es)
        try:
            sans_verif_code(lambda: A4.analyser(e3, e4, es, effectifs=None, verifier_exp003=False))
        except A.ControleEchoue as err:
            assert motif in str(err), (nom, str(err))
            print(f"   refusé comme attendu ({nom}) : {err}")
            return
        raise AssertionError(f"{nom} : contrôle non déclenché")

    def modifier(dossier, f, **champs):
        x = dict(np.load(dossier / f))
        x.update(champs)
        np.savez_compressed(dossier / f, **x)

    def sanite_differente(e3, e4, es):
        x = dict(np.load(es / "all-4_s42.npz"))
        x["success"][0] = ~x["success"][0]
        np.savez_compressed(es / "all-4_s42.npz", **x)

    doit_echouer("fichier manquant", lambda e3, e4, es: (e4 / "all-3-g64_s44.npz").unlink(), "fichiers manquants")
    doit_echouer("couches", lambda e3, e4, es: modifier(e4, "autres-4_s43.npz", n_couches=109), "109 couches")
    doit_echouer("autres épisodes", lambda e3, e4, es: modifier(e4, "encpred-4_s45.npz", start_steps=np.zeros(A.N, int)),
                 "start_steps différent")
    doit_echouer("arbre modifié", lambda e3, e4, es: modifier(e4, "all-4-g64_s46.npz", git_sale=True), "arbre git modifié")
    doit_echouer("sanité", sanite_differente, "sanité : success diffère")
    print("ok contrôles bloquants d'EXP-004 (5 refus)")


if __name__ == "__main__":
    test_verifier_code4()
    with tempfile.TemporaryDirectory() as tmp:
        d3, d4, san = test_bout_en_bout(tmp)
        test_refus(tmp, d3, d4, san)
    print("tous les tests passent")
