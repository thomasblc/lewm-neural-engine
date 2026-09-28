"""Tests de analyse_exp004.py sur des grilles simulées (EXP-003 et EXP-004 avec le même hasard par épisode).

Le simulateur d'EXP-003 est réutilisé tel quel, les versions d'EXP-004 ajoutées à la fin de sa liste : les
fichiers simulés d'EXP-003 restent les mêmes, ceux d'EXP-004 partagent la géométrie et le hasard. Les
statistiques sont recalculées à la main depuis les fichiers, ce qui fixe aussi l'appariement des épisodes.

Usage :
    python test_analyse_exp004.py
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
import analyse_exp004 as A4
import test_analyse_exp003 as T


# les comparaisons de PREREG-EXP004.md, recopiées ici indépendamment du script : nom, version, référence, sens
PREENREGISTREES = (("Q3a", "autres:4", "fp32", -1), ("Q3b", "encpred:4", "all:4", +1),
                   ("Q5a", "all:4:g64", "all:4", +1), ("Q5b", "all:3:g64", "all:3", +1))


def grilles(tmp, nom, **kw):
    """Écrit une grille simulée complète, puis la répartit en EXP-003, EXP-004 et les deux sanités."""
    tout, d3, d4, san, fin = (Path(tmp) / f"{nom}-{x}" for x in ("tout", "exp003", "exp004", "sanite", "sanite-fin"))
    versions, couches = A.VERSIONS, dict(A.COUCHES)
    A.VERSIONS = versions + A4.NOUVELLES
    A.COUCHES.update({"autres": 6, "encpred": 103})
    try:
        T.simuler(tout, **kw)
    finally:
        A.VERSIONS = versions
        A.COUCHES.clear()
        A.COUCHES.update(couches)
    for d in (d3, d4, san, fin):
        d.mkdir()
        shutil.copy(tout / "meta.json", d / "meta.json")
    for f in tout.glob("*.npz"):
        spec = str(np.load(f)["spec"])
        shutil.copy(f, (d4 if spec in A4.NOUVELLES else d3) / f.name)
    shutil.copy(d3 / "all-4_s42.npz", san / "all-4_s42.npz")
    shutil.copy(d4 / "autres-4_s42.npz", fin / "autres-4_s42.npz")
    return d3, d4, san, fin


def sans_verif_code(fonction):
    """Les fichiers simulés portent le commit c5703db : on neutralise la vérification du quant.py d'EXP-004."""
    vrai = A4.verifier_code4
    A4.verifier_code4 = lambda sha: None
    try:
        return fonction()
    finally:
        A4.verifier_code4 = vrai


def K_fichiers(d3, d4, v, graines=A.GA):
    d = d4 if v in A4.NOUVELLES else d3
    return sum(np.load(d / f"{v.replace(':', '-')}_s{g}.npz")["success"].astype(np.int64) for g in graines)


def test_verifier_code4():
    empreinte = hashlib.sha256((A.ICI / "quant.py").read_bytes()).hexdigest()
    assert empreinte == A4.QUANT_SHA256, "quant.py a changé : l'empreinte figée ne correspond plus"
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
    A4.verifier_code4(tete)
    print(f"ok vérification du code : empreinte de quant.py, c5703db et commit inconnu refusés, {tete[:7]} accepté")


def test_bout_en_bout(tmp):
    effets = {"all:4": (1.2, 1.0), "encoder:4": (0.5, 1.0), "autres:4": (0.6, 1.0), "encpred:4": (0.5, 1.0),
              "all:4:g64": (0.6, 1.0), "all:3": (3.0, 1.0), "all:3:g64": (2.0, 1.0), "autres:3": (1.5, 1.0)}
    d3, d4, san, fin = grilles(tmp, "effets", effets=effets, graine=5, divergents=())
    r = sans_verif_code(lambda: A4.analyser(d3, d4, san, fin, effectifs=None, verifier_exp003=False))
    for nom in ("Q3a", "Q3b", "Q5a", "Q5b"):
        assert r["tests"][nom]["verdict"] == "soutenue", (nom, r["tests"][nom])
    # tout recalculer à la main depuis les fichiers : sommes, p, Holm, sensibilité (fixe références et appariement)
    inv = A.Inversion()
    K = {v: K_fichiers(d3, d4, v) for v in A4.REFERENCES + A4.NOUVELLES}
    Ks = {v: K_fichiers(d3, d4, v, A4.GRAINES_SENSIBILITE) for v in A4.REFERENCES + A4.NOUVELLES}
    assert tuple(A4.COMPARAISONS) == PREENREGISTREES
    for Kx, cle in ((K, None), (Ks, "sensibilite_44_46")):
        p = [inv.p(Kx[v] - Kx[ref]) for _, v, ref, _ in PREENREGISTREES]
        adj = A.holm(p)
        for (nom, v, ref, sens), pv, pa in zip(PREENREGISTREES, p, adj):
            t = r["tests"][nom] if cle is None else r["tests"][nom][cle]
            assert t["somme"] == int((Kx[v] - Kx[ref]).sum()) and t["p"] == pv and abs(t["p_holm"] - pa) < 1e-15, (nom, cle)
            assert t["verdict"] == A.verdict(pa, (Kx[v] - Kx[ref]).sum(), sens), (nom, cle)
            if cle is not None:
                assert abs(t["delta_reussite"] - (Kx[v] - Kx[ref]).mean() / 3) < 1e-12
    for nom, t in r["tests"].items():
        assert t["concordance"] == (t["verdict"] == t["sensibilite_44_46"]["verdict"])
        assert t["verdict_retenu"] == (t["verdict"] if nom.startswith("Q5") or t["concordance"] else "non concordant (exploratoire)")
    i_main = ((K["all:4"] - K["encoder:4"] - K["predictor:4"] - K["autres:4"] + 2 * K["fp32"]) / 5).mean()
    cle = next(k for k in r["additivite"] if k.startswith("I"))
    assert abs(r["additivite"][cle]["moyenne"] - i_main) < 1e-12
    part = (K["encpred:4"] - K["all:4"]).sum() / (K["fp32"] - K["all:4"]).sum()
    pe = r["part_expliquee_autres_4_bits"]
    assert abs(pe["moyenne"] - part) < 1e-12 and pe["bas"] <= part <= pe["haut"]
    idx = A.Bootstrap().idx
    num, den = (K["encpred:4"] - K["all:4"]).astype(float), (K["fp32"] - K["all:4"]).astype(float)
    rr = num[idx].sum(axis=1) / np.maximum(den[idx].sum(axis=1), 1e-12)
    assert abs(pe["bas"] - np.percentile(rr, 2.5)) < 1e-12 and abs(pe["haut"] - np.percentile(rr, 97.5)) < 1e-12
    # d) part non définie quand tout le modèle ne perd rien
    assert A4.part_expliquee(A.Bootstrap(), np.ones(A.N), np.zeros(A.N)) is None
    assert "non définie" in A4.rapport(dict(r, part_expliquee_autres_4_bits=None))
    assert r["controles"]["divergences"] == {v: 0 for v in A4.NOUVELLES}
    txt = A4.rapport(r)
    assert "Sensibilité" in txt and "Part de la chute" in txt and not re.search(r"\bnan\b|\binf\b", txt.lower())
    print("ok bout en bout, effets connus : 4 hypothèses soutenues ; sommes, p, Holm, sensibilité, additivité et part "
          "expliquée recalculées à la main")

    # le vrai chemin : contrôles bloquants d'EXP-003 sur ses 55 fichiers simulés, puis réutilisation de ses tables
    r2 = sans_verif_code(lambda: A4.analyser(d3, d4, san, fin, effectifs=None, verifier_exp003=True))
    assert json.dumps(A.pur(r2["tests"]), sort_keys=True) == json.dumps(A.pur(r["tests"]), sort_keys=True)
    assert json.dumps(A.pur(r2["additivite"]), sort_keys=True) == json.dumps(A.pur(r["additivite"]), sort_keys=True)
    print("ok chemin réel (analyse_exp003.charger puis tables) : mêmes résultats que le chemin de test")

    # b) non concordant : fort effet d'autres:4 aux graines 42 et 43, aucun aux graines 44 à 46
    effets_nc = dict(effets, **{"autres:4": (2.5, 1.0)})
    d3c, d4c, sanc, finc = grilles(tmp, "nonconc", effets=effets_nc, graine=7, divergents=())
    for g in A4.GRAINES_SENSIBILITE:
        x = dict(np.load(d3c / f"fp32_s{g}.npz"))
        x.update(spec="autres:4", n_couches=6)
        np.savez_compressed(d4c / f"autres-4_s{g}.npz", **x)
    r_nc = sans_verif_code(lambda: A4.analyser(d3c, d4c, sanc, finc, effectifs=None, verifier_exp003=False))
    t = r_nc["tests"]["Q3a"]
    assert t["verdict"] == "soutenue" and t["sensibilite_44_46"]["somme"] == 0, t
    assert not t["concordance"] and t["verdict_retenu"] == "non concordant (exploratoire)", t
    assert "non concordant (exploratoire)" in A4.rapport(r_nc)
    print("ok règle 2 : un effet présent aux graines 42 et 43 seulement donne « non concordant (exploratoire) »")

    iden = ("autres:4", "encpred:4", "all:4:g64", "all:4", "autres:3", "all:3:g64", "all:3")
    nul = {"all:4": (1.0, 1.0), "encpred:4": (1.0, 1.0), "all:4:g64": (1.0, 1.0), "all:3": (2.0, 1.0),
           "all:3:g64": (2.0, 1.0)}
    d3n, d4n, sann, finn = grilles(tmp, "nul", effets=nul, graine=6, identiques=iden)
    for v, ref in (("autres:4", "fp32"), ("encpred:4", "all:4"), ("all:4:g64", "all:4"), ("all:3:g64", "all:3")):
        assert np.array_equal(K_fichiers(d3n, d4n, v), K_fichiers(d3n, d4n, ref)), v   # épisode par épisode
    r = sans_verif_code(lambda: A4.analyser(d3n, d4n, sann, finn, effectifs=None, verifier_exp003=False))
    for nom in ("Q3a", "Q3b", "Q5a", "Q5b"):
        t = r["tests"][nom]
        assert t["somme"] == 0 and t["p"] == 1.0 and t["verdict"] == "non soutenue" and t["concordance"], (nom, t)
    print("ok bout en bout, sans effet : issues identiques épisode par épisode, p = 1, rien n'est soutenu")
    return d3, d4, san, fin


def test_refus(tmp, d3, d4, san, fin):
    def doit_echouer(nom, alterer, motif):
        e = [Path(tmp) / f"{nom}-{x}" for x in ("3", "4", "s", "f")]
        for src, dst in zip((d3, d4, san, fin), e):
            shutil.copytree(src, dst)
        alterer(*e)
        try:
            sans_verif_code(lambda: A4.analyser(*e, effectifs=None, verifier_exp003=False))
        except A.ControleEchoue as err:
            assert motif in str(err), (nom, str(err))
            print(f"   refusé comme attendu ({nom}) : {err}")
            return
        raise AssertionError(f"{nom} : contrôle non déclenché")

    def modifier(dossier, f, **champs):
        x = dict(np.load(dossier / f))
        x.update(champs)
        np.savez_compressed(dossier / f, **x)

    def meta(dossier, **champs):
        m = json.loads((dossier / "meta.json").read_text())
        for k, v in champs.items():
            if k == "torch":
                m["torch"] = v
            else:
                m["protocole"][k] = v
        (dossier / "meta.json").write_text(json.dumps(m))

    def inverser(dossier, f):
        x = dict(np.load(dossier / f))
        x["success"][0] = ~x["success"][0]
        np.savez_compressed(dossier / f, **x)

    doit_echouer("fichier manquant", lambda e3, e4, es, ef: (e4 / "all-3-g64_s44.npz").unlink(), "fichiers manquants")
    doit_echouer("couches", lambda e3, e4, es, ef: modifier(e4, "autres-4_s43.npz", n_couches=109), "109 couches")
    doit_echouer("autres épisodes", lambda e3, e4, es, ef: modifier(e4, "encpred-4_s45.npz", start_steps=np.zeros(A.N, int)),
                 "start_steps différent")
    doit_echouer("arbre modifié", lambda e3, e4, es, ef: modifier(e4, "all-4-g64_s46.npz", git_sale=True), "arbre git modifié")
    doit_echouer("sanité d'entrée", lambda e3, e4, es, ef: inverser(es, "all-4_s42.npz"), "sanité d'entrée : success diffère")
    doit_echouer("sanité de clôture", lambda e3, e4, es, ef: inverser(ef, "autres-4_s42.npz"), "sanité de clôture : success diffère")
    doit_echouer("protocole", lambda e3, e4, es, ef: meta(e4, device="cpu"), "protocole différent")
    doit_echouer("clôture sur une autre version",
                 lambda e3, e4, es, ef: shutil.copy(e4 / "encpred-4_s42.npz", ef / "autres-4_s42.npz"),
                 "sanité de clôture : pas la même version")
    doit_echouer("torch", lambda e3, e4, es, ef: meta(ef, torch="2.15.0"), "version de torch différente")
    print("ok contrôles bloquants d'EXP-004 (9 refus, chacun avec son motif)")


if __name__ == "__main__":
    test_verifier_code4()
    with tempfile.TemporaryDirectory() as tmp:
        d3, d4, san, fin = test_bout_en_bout(tmp)
        test_refus(tmp, d3, d4, san, fin)
    print("tous les tests passent")
