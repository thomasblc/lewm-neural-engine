"""Tests de analyse_exp005.py : cas construits à la main (sans simulateur, comptages refaits en numpy pur), puis
grilles simulées de 2 000 épisodes (simulateur d'EXP-003) et refus des fichiers corrompus.

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

SIGNES = np.random.default_rng(0).choice(np.array([-1.0, 1.0]), size=(10_000, A5.N5))   # refaits ici, à part


def p_numpy(x, sens, n=None):
    """Comptage d'inversion de signe écrit indépendamment d'analyse_exp003.Inversion."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x) if n is None else n
    with np.errstate(all="ignore"):                          # faux avertissements de numpy avec Accelerate
        t, obs = SIGNES[:, :n] @ x, x.sum()
    compte = {"plus": np.sum(t >= obs), "moins": np.sum(t <= obs), "bilateral": np.sum(np.abs(t) >= abs(obs))}[sens]
    return (1 + int(compte)) / 10_001


def test_statistique_a_la_main():
    inv = A.Inversion(n=A5.N5)
    rng = np.random.default_rng(99)
    w0 = np.zeros(1333)
    # (i) réussite exactement 2,5 points au-dessus (250 épisodes à +1 sur 2 000) : seule la borne haute échoue
    x = np.zeros(A5.N5, dtype=int); x[:250] = 1
    s = A5.stat_h6(x, w0 + 0.01, inv)
    assert x.mean() / 5 == 0.025 and s["p_haut"] > 0.05 and s["p"] == s["p_haut"], s
    assert s["p_haut"] == p_numpy(10 * x - 1, "moins") and s["p_bas"] == p_numpy(10 * x + 1, "plus")
    # (ii) réussite exactement 1,5 point en dessous (150 épisodes à -1) : dans la marge, p_bas petite
    x = np.zeros(A5.N5, dtype=int); x[:150] = -1
    s = A5.stat_h6(x, w0 + 0.01, inv)
    assert x.mean() / 5 == -0.015
    assert s["p_bas"] == p_numpy(10 * x + 1, "plus") < 0.05 and s["p_haut"] == p_numpy(10 * x - 1, "moins")
    # (ii bis) exactement 2,5 points en dessous : hors de la marge, la borne basse échoue
    x = np.zeros(A5.N5, dtype=int); x[:250] = -1
    s = A5.stat_h6(x, w0 + 0.01, inv)
    assert s["p_bas"] > 0.05 and s["p"] == s["p_bas"] == p_numpy(10 * x + 1, "plus"), s
    # (iii) petit effet sur la pose : p unilatérale vers 0,02, la bilatérale serait vers le double
    w = rng.normal(0, 0.258, 1333)
    w = w - w.mean() + 2.1 * 0.258 / np.sqrt(1333)
    s = A5.stat_h6(np.zeros(A5.N5, dtype=int), w, inv)
    assert s["p_b"] == p_numpy(w, "plus", 1333) and 0.005 < s["p_b"] < 0.04, s["p_b"]
    assert p_numpy(w, "bilateral", 1333) > 1.5 * s["p_b"]
    print("ok statistique à la main : borne haute seule en échec, p_bas dans la marge, p_b unilatérale, "
          "toutes égales au comptage numpy indépendant")


def grille(tmp, nom, **kw):
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


def analyser(d, fin, excl):
    return sans_verif_code(lambda: A5.analyser(d, fin, [excl], effectifs=None))


def charger_K_M3(d, v):
    K = sum(np.load(d / f"{v.replace(':', '-')}_s{g}.npz")["success"].astype(np.int64) for g in A5.GRAINES)
    M3 = np.stack([A.controler_course(dict(np.load(d / f"{v.replace(':', '-')}_s{g}.npz")), v)[1] for g in A5.GRAINES])
    return K, M3


def test_code():
    for f, empreinte in (("run_grid.py", A5.RUN_GRID_SHA256), ("quant.py", A5.QUANT_SHA256)):
        assert hashlib.sha256((A.ICI / f).read_bytes()).hexdigest() == empreinte, f"{f} a changé"
    for sha, motif in (("c5703dbbacb1537be9cc03377a3a922f4a81c1ed", "run_grid.py d'EXP-005"), ("0" * 40, "inconnu")):
        try:
            A5.verifier_code5(sha)
            raise AssertionError(sha)
        except A.ControleEchoue as err:
            assert motif in str(err), err
    try:
        A5.verifier_analyse("c5703dbbacb1537be9cc03377a3a922f4a81c1ed")
        raise AssertionError("code d'analyse de c5703db accepté")
    except A.ControleEchoue as err:
        assert "code d'analyse diffère" in str(err)
    tete = subprocess.run(["git", "-C", str(A.ICI), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    A5.verifier_code5(tete)
    A5.verifier_analyse(tete)
    print(f"ok code : empreintes, code d'analyse identique à {tete[:7]} ; c5703db et commit inconnu refusés")


def test_scenarios(tmp):
    # 1. effet cherché à 6 bits (réussite inchangée, échecs plus loin), rien à 8 bits
    d, fin, excl = grille(tmp, "effet", effets={"all:6": (0.0, 1.5)}, graine=11, divergents=())
    r = analyser(d, fin, excl)
    h = r["H6"]
    assert h["verdict"] == "établie" and r["progression_8_vers_6"], h
    K, M3 = {}, {}
    for v in A5.VERSIONS5:
        K[v], M3[v] = charger_K_M3(d, v)
    bouge = A.depart({("fp32", A.GA[0]): dict(np.load(d / "fp32_s52.npz"))})["bouge"]
    x = K["all:6"] - K["fp32"]
    w = (A.lg(M3["all:6"]).mean(0) - A.lg(M3["fp32"]).mean(0))[bouge]
    assert h["p_bas"] == p_numpy(10 * x + 1, "plus") and h["p_haut"] == p_numpy(10 * x - 1, "moins")
    assert h["p_b"] == p_numpy(w, "plus", int(bouge.sum())) and h["p"] == max(h["p_bas"], h["p_haut"], h["p_b"])
    assert h["delta_reussite"] == x.mean() / 5 and abs(h["delta_log_M3_bouge"] - w.mean()) < 1e-12
    assert h["episodes_bouge"] == int(bouge.sum()) < A5.N5
    moy = (x / 5)[A.Bootstrap(n=A5.N5).idx].mean(axis=1)
    assert h["ecarts"]["reussite"]["bas"] == np.percentile(moy, 2.5)
    wb = (A.lg(M3["all:6"]).mean(0) - A.lg(M3["fp32"]).mean(0))[bouge]
    assert h["ecarts"]["log_M3_bouge"]["haut"] == np.percentile(wb[A.Bootstrap(n=int(bouge.sum()), graine=3).idx].mean(axis=1), 97.5)
    assert abs(sum(e["contribution_log_M3"] for e in h["decomposition_strate"].values()) - w.mean()) < 1e-12
    x8 = K["all:8"] - K["fp32"]
    assert r["controle_8_bits"]["delta_reussite"] == x8.mean() / 5
    assert (h["formulation"] == "la pose du T se dégrade à 6 bits alors que la réussite ne bouge pas") == h["reussite_stable_95"]
    txt = A5.rapport(r)
    assert "établie" in txt and "Contrôle à 8 bits" in txt and not re.search(r"\bnan\b|\binf\b", txt.lower())
    print("ok effet cherché : H6 établie, progression 8 puis 6 bits ; p, écarts, bornes bootstrap et décomposition "
          "recalculés à part")

    # 2. la réussite baisse aussi : H6 ne passe pas, même si la pose se dégrade
    d2, fin2, excl2 = grille(tmp, "baisse", effets={"all:6": (1.2, 1.5)}, graine=12, divergents=())
    h = analyser(d2, fin2, excl2)["H6"]
    assert h["verdict"] == "non établie" and h["p_bas"] > 0.5 and h["p_b"] < 0.05 and h["formulation"] == "H6 non établie", h
    print("ok réussite en baisse : H6 non établie malgré la pose dégradée")

    # 3. aucun effet : versions identiques épisode par épisode
    d3, fin3, excl3 = grille(tmp, "nul", graine=13, divergents=(), identiques=("all:6", "all:8"), paquet_reussi=())
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

    def modifier(dossier, f, **champs):
        x = dict(np.load(dossier / f))
        x.update(champs)
        np.savez_compressed(dossier / f, **x)

    def inverser(dossier, f):
        x = dict(np.load(dossier / f))
        x["success"][0] = ~x["success"][0]
        np.savez_compressed(dossier / f, **x)

    def trajectoire_exclue(e, ef, ex):                        # une trajectoire de la grille dans l'exclusion
        x = json.loads(ex.read_text())
        x["protocole"]["episodes_idx"][0] = 1000
        ex.write_text(json.dumps(x))
        for dd in (e, ef):
            m = json.loads((dd / "meta.json").read_text())
            m["protocole"]["exclus"][0][0] = 1000
            (dd / "meta.json").write_text(json.dumps(m))

    def doublon(e, ef, ex):                                   # deux positions dans la même trajectoire
        ep = list(range(1000, 1000 + A5.N5)); ep[1] = ep[0]
        for dd in (e, ef):
            meta(dd, episodes_idx=ep)
            for f in dd.glob("*.npz"):
                modifier(dd, f.name, episodes_idx=np.array(ep))

    doit_echouer("trajectoire exclue", trajectoire_exclue, "trajectoires en commun")
    doit_echouer("deux positions par trajectoire", doublon, "plusieurs positions dans une même trajectoire")
    doit_echouer("graine de tirage", lambda e, ef, ex: meta(e, graine_episodes=42), "graine de tirage")
    doit_echouer("règle de tirage", lambda e, ef, ex: meta(e, regle_tirage="couples"), "règle de tirage différente")
    doit_echouer("fichier manquant", lambda e, ef, ex: (e / "all-8_s54.npz").unlink(), "fichiers manquants")
    doit_echouer("couches", lambda e, ef, ex: modifier(e, "all-6_s55.npz", n_couches=73), "73 couches")
    doit_echouer("arbre modifié", lambda e, ef, ex: modifier(e, "all-8_s53.npz", git_sale=True), "arbre git modifié")
    doit_echouer("deux commits", lambda e, ef, ex: modifier(e, "fp32_s56.npz", git="1" * 40), "plusieurs commits")
    doit_echouer("état de départ", lambda e, ef, ex: modifier(e, "all-8_s52.npz",
                 init_state=np.load(e / "all-8_s52.npz")["init_state"] + 1), "init_state différent")
    doit_echouer("paquet arrêté tôt", lambda e, ef, ex: modifier(e, "fp32_s55.npz", n_steps=np.r_[30, np.full(39, A.T)].astype(np.int32)),
                 "paquet 0 arrêté à 30 pas")
    doit_echouer("clôture sur une autre version", lambda e, ef, ex: shutil.copy(e / "fp32_s52.npz", ef / "all-6_s52.npz"),
                 "sanité de clôture : pas la même version")
    doit_echouer("sanité de clôture", lambda e, ef, ex: inverser(ef, "all-6_s52.npz"), "sanité de clôture : success diffère")
    print("ok contrôles bloquants d'EXP-005 (12 refus, chacun avec son motif)")


if __name__ == "__main__":
    test_statistique_a_la_main()
    test_code()
    with tempfile.TemporaryDirectory() as tmp:
        d, fin, excl = test_scenarios(tmp)
        test_refus(tmp, d, fin, excl)
    print("tous les tests passent")
