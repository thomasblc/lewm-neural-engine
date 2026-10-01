"""Tests de analyse_exp014.py sur de vrais fichiers de référence (EXP-003, EXP-006, copiés dans un dossier temporaire)
et des fichiers Core ML fabriqués à effet connu. Scénario A : chaque codage = copie de la pleine précision (aucune
perte). Scénario C : effets proches de la marge, p strictement entre le plancher et 1. p par inversion de signe,
bootstrap-t et Holm recomptés en numpy, indépendamment du code testé ; contrôle négatif, sensibilité et effet du codage
seul recomptés ; refus des fichiers incohérents, chacun avec son motif. À lancer sur un arbre commité.

Usage :
    python test_analyse_exp014.py
"""
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

import analyse_exp003 as A
import analyse_exp014 as A14

RES = A14.ICI.parent / "resultats"
SIGNES = np.random.default_rng(0).choice(np.array([-1.0, 1.0]), size=(10_000, 500))
TETE = subprocess.run(["git", "-C", str(A14.ICI), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
PROPRE = subprocess.run(["git", "-C", str(A14.ICI), "status", "--porcelain", "--untracked-files=no"],
                        capture_output=True, text=True).stdout.strip() == ""
SOURCES = {"A": {(t, c): "fp32" for t in A14.TACHES for c in A14.CODAGES},
           "C": {("pusht", "int8"): "all-6", ("pusht", "int4c"): "predictor-3", ("pusht", "pal4"): "fp32",
                 ("tworoom", "int8"): "all-6", ("tworoom", "int4c"): "autres-4", ("tworoom", "pal4"): "fp32"}}
OCTETS = {"int8": 11.7e6, "int4c": 5.9e6, "pal4": 5.9e6}


def ecrire_temps(f, c, g, empreinte):
    appels = [[0.8 * n, n] for n in [50] * 10 + [12] * 10]
    if g == 44:
        appels[3] = [0.8 * 50 + 600.0, 50]                    # un appel allongé par une pause de batterie, écarté
    coreml = {"placement": {"NeuralEngine": 361}, "codage": c, "octets_poids": OCTETS[c], "sha256_poids": empreinte,
              "sha256_modele": f"modele-{empreinte}"}
    f.with_suffix(".temps.json").write_text(json.dumps({"moteur": "coreml-ne", "appels_solve": appels, "coreml": coreml,
                                                        "puissance": {"puissance_moyenne_w": None}, "git": TETE,
                                                        "lectures": [{"t": 2000.0 + g}]}))


def ecrire(f, x, **champs):
    x = dict(x)
    x.update({"git": np.str_(TETE), "git_sale": np.bool_(False), **champs})
    np.savez_compressed(f, **x)


def fabriquer(racine, scenario="A"):
    r = Path(racine)
    for d in A14.TACHES.values():
        shutil.copytree(RES / d, r / d, ignore=shutil.ignore_patterns("*.log"))
    for t, ref_d in A14.TACHES.items():
        for d, g in ((f"exp014-repro-{t}", 42), (f"exp014-fin-{t}", 46)):
            (r / d).mkdir()
            ecrire(r / d / f"fp32_s{g}.npz", np.load(r / ref_d / f"fp32_s{g}.npz"))
        (r / f"exp012-ne-{t}").mkdir()                          # le 16 bits sur le Neural Engine d'EXP-012 : copie du fp32
        for g in A14.GRAINES:
            ecrire(r / f"exp012-ne-{t}" / f"fp32_s{g}.npz", np.load(r / ref_d / f"fp32_s{g}.npz"), git=np.str_("e" * 40),
                   moteur=np.str_("coreml-ne"))
        proto = json.loads((r / ref_d / "meta.json").read_text())["protocole"]
        for c in A14.CODAGES:
            d = r / A14.dossier(t, c)
            d.mkdir()
            pm = dict(proto, moteur={"nom": "coreml-ne", "coremltools": "9.0", "macos": "26.6.2", "precision": "float16", "codage": c})
            (d / "meta.json").write_text(json.dumps({"protocole": pm, "git": TETE, "git_sale": False}))
            for g in A14.GRAINES:
                ecrire(d / f"fp32_s{g}.npz", np.load(r / ref_d / f"{SOURCES[scenario][(t, c)]}_s{g}.npz"), spec=np.str_("fp32"),
                       n_couches=np.int64(0), moteur=np.str_("coreml-ne"), coreml_sha256=np.str_(f"{t}-{c}" * 4))
                ecrire_temps(d / f"fp32_s{g}.npz", c, g, f"{t}-{c}" * 4)
    return r


def K(r, d, v, graines=A14.GRAINES):
    return sum(np.load(Path(r) / d / f"{v}_s{g}.npz")["success"].astype(float) for g in graines)


def p_signe(y):
    return (1 + int((SIGNES @ y >= y.sum()).sum())) / 10_001


def p_boot_t(y):
    """Bootstrap-t écrit à part : t = moyenne / erreur-type, t* = (moyenne* - moyenne) / erreur-type*."""
    n, rng = len(y), np.random.default_rng(4)
    t = y.mean() / (y.std(ddof=1) / np.sqrt(n)) if y.std() > 0 else np.inf
    idx = rng.integers(0, n, size=(10_000, n))
    compte = 0
    for i in range(0, 10_000, 2_000):
        yb = y[idx[i:i + 2_000]]
        se = yb.std(axis=1, ddof=1) / np.sqrt(n)
        tb = np.where(se > 0, (yb.mean(axis=1) - y.mean()) / np.where(se > 0, se, 1), 0.0)
        compte += int((tb >= t).sum())
    return (1 + compte) / 10_001


def holm_numpy(ps):
    o = np.argsort(ps)
    q = np.minimum(1, np.maximum.accumulate(np.array(ps)[o] * (len(ps) - np.arange(len(ps)))))
    out = np.empty(len(ps))
    out[o] = q
    return out


def verifier_p(r, res):
    ps = []
    for nom, t, c in A14.TESTS:
        x = K(r, A14.dossier(t, c), "fp32") - K(r, A14.TACHES[t], "fp32")
        y = 20 * x + 2
        s, b = p_signe(y), p_boot_t(y)
        T = res["tests"][nom]
        assert T["p_signe"] == s and abs(T["p_bootstrap_t"] - b) < 1e-12 and T["p"] == max(s, b), (nom, T, s, b)
        ps.append(max(s, b))
    for (nom, *_), q in zip(A14.TESTS, holm_numpy(ps)):
        assert abs(res["tests"][nom]["p_holm"] - q) < 1e-12, nom
        assert res["tests"][nom]["verdict"] == ("soutenue" if q < 0.05 else "non soutenue"), nom
    for t, ref_d in A14.TACHES.items():                       # contrôle négatif, effet du codage seul
        f, rep = K(r, ref_d, "fp32"), K(r, ref_d, "fp32", A14.REPLIQUE)
        c = res["controle_negatif"][t]
        assert c["p_signe"] == p_signe(20 * (rep - f) + 2) and abs(c["ecart_points"]["moyenne"] - (rep - f).mean() / 5) < 1e-12
        for cod in A14.CODAGES:
            e = (K(r, A14.dossier(t, cod), "fp32") - K(r, f"exp012-ne-{t}", "fp32")) / 5
            assert abs(res["descriptifs"][t][f"{cod}_moins_ne16"]["moyenne"] - e.mean()) < 1e-12, (t, cod)
    for nom, t, c in A14.TESTS:
        x = K(r, A14.dossier(t, c), "fp32") - (K(r, A14.TACHES[t], "fp32") + K(r, A14.TACHES[t], "fp32", A14.REPLIQUE)) / 2
        assert res["sensibilite_10_graines"][nom]["p_signe"] == p_signe(20 * x + 2), nom
    return ps


def test_scenario_a(tmp):
    r = fabriquer(Path(tmp) / "a")
    res = A14.analyser(r, verifier=False)
    verifier_p(r, res)
    assert len(res["tests"]) == 6
    assert all(res["tests"][n]["verdict"] == "soutenue" and res["tests"][n]["somme"] == 0 for n in res["tests"])
    for t in A14.TACHES:
        d = res["descriptifs"][t]
        assert all(abs(d["temps_par_plan_ms"][c] - 800) < 1e-9 and d["octets_poids"][c] == OCTETS[c] for c in A14.CODAGES), d
    assert "Effet du codage seul" in A14.rapport(res)
    print("ok scénario A : 6 hypothèses soutenues, contrôle négatif, sensibilité et effet du codage seul recomptés, "
          "temps par plan avec l'appel allongé écarté")
    return r


def test_scenario_c(tmp):
    r = fabriquer(Path(tmp) / "c", "C")
    res = A14.analyser(r, verifier=False)
    ps = verifier_p(r, res)
    assert all(1 / 10_001 < p < 1 for p, (n, t, c) in zip(ps, A14.TESTS) if SOURCES["C"][(t, c)] != "fp32"), ps
    T = res["tests"]
    print(f"ok scénario C : p intermédiaires, Holm sur 6 recompté ; verdicts {[T[n]['verdict'] for n in T]}")


def test_refus(tmp, r):
    def essai(motif, modifier):
        rr = Path(tmp) / "refus"
        shutil.rmtree(rr, ignore_errors=True)
        shutil.copytree(r, rr)
        modifier(rr)
        try:
            A14.analyser(rr, verifier=False)
        except A.ControleEchoue as e:
            assert motif in str(e), (motif, str(e))
            return
        raise AssertionError(f"accepté à tort : {motif}")

    def champ(d, nom, cle, val):
        def m(rr):
            x = dict(np.load(rr / d / nom))
            x[cle] = val(x)
            np.savez_compressed(rr / d / nom, **x)
        return m

    def meta(d, f):
        def m(rr):
            g = rr / d / "meta.json"
            mm = json.loads(g.read_text())
            f(mm["protocole"])
            g.write_text(json.dumps(mm))
        return m

    def temps(d, nom, f):
        def m(rr):
            g = rr / d / nom
            j = json.loads(g.read_text())
            f(j)
            g.write_text(json.dumps(j))
        return m
    cas = [("reproduction pusht graine 42", lambda rr: shutil.copy(rr / "exp003" / "all-4_s42.npz", rr / "exp014-repro-pusht" / "fp32_s42.npz")),
           ("reproduction tworoom graine 46", champ("exp014-fin-tworoom", "fp32_s46.npz", "success", lambda x: ~x["success"])),
           ("reproduction pusht graine 46 : autre commit", champ("exp014-fin-pusht", "fp32_s46.npz", "git", lambda x: np.str_("c" * 40))),
           ("manquant", lambda rr: (rr / "exp014-int8-pusht" / "fp32_s44.npz").unlink()),
           ("manquant", lambda rr: (rr / "exp012-ne-tworoom" / "fp32_s45.npz").unlink()),
           ("version ou graine", champ("exp014-pal4-tworoom", "fp32_s43.npz", "seed", lambda x: np.int64(42))),
           ("arbre git modifié", champ("exp014-int4c-tworoom", "fp32_s45.npz", "git_sale", lambda x: np.bool_(True))),
           ("couches quantifiées en PyTorch", champ("exp014-int8-pusht", "fp32_s46.npz", "n_couches", lambda x: np.int64(30))),
           ("moteur", champ("exp014-pal4-pusht", "fp32_s42.npz", "moteur", lambda x: np.str_("coreml-gpu"))),
           ("empreintes", champ("exp014-int4c-tworoom", "fp32_s44.npz", "coreml_sha256", lambda x: np.str_("autre"))),
           ("modèle Core ML différent", temps("exp014-int8-pusht", "fp32_s45.temps.json", lambda j: j["coreml"].update(sha256_modele="x"))),
           ("hors du Neural Engine", temps("exp014-int4c-pusht", "fp32_s46.temps.json",
                                           lambda j: j["coreml"].update(placement={"NeuralEngine": 150, "CPU": 211}))),
           ("codage", temps("exp014-pal4-tworoom", "fp32_s42.temps.json", lambda j: j["coreml"].update(codage="int8"))),
           ("protocole différent", meta("exp014-int8-tworoom", lambda p: p.update(chunk=32))),
           ("moteur", meta("exp014-int4c-pusht", lambda p: p["moteur"].update(codage="int4b32"))),
           ("moteur", meta("exp014-pal4-pusht", lambda p: p["moteur"].update(precision="float32"))),
           ("init_state différent", champ("exp014-int8-tworoom", "fp32_s42.npz", "init_state", lambda x: x["init_state"] + 1)),
           ("init_state différent", champ("exp012-ne-pusht", "fp32_s43.npz", "init_state", lambda x: x["init_state"] + 1)),
           ("plusieurs commits", lambda rr: (champ("exp014-pal4-tworoom", "fp32_s45.npz", "git", lambda x: np.str_("b" * 40))(rr),
                                             temps("exp014-pal4-tworoom", "fp32_s45.temps.json", lambda j: j.update(git="b" * 40))(rr))),
           ("fichier de temps", champ("exp014-int8-pusht", "fp32_s46.npz", "git", lambda x: np.str_("d" * 40))),
           ("fichier de temps", temps("exp014-int4c-pusht", "fp32_s43.temps.json", lambda j: j.update(moteur="torch")))]
    for motif, m in cas:
        essai(motif, m)
    print(f"ok contrôles bloquants ({len(cas)} refus, chacun avec son motif)")


def test_code():
    assert PROPRE, "arbre modifié : committer avant de lancer les tests"
    A14.verifier_code(TETE)
    print(f"ok code : fichiers d'analyse et de calcul identiques à {TETE[:7]}")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp, np.errstate(all="ignore"):
        r = test_scenario_a(tmp)
        test_scenario_c(tmp)
        test_refus(tmp, r)
        test_code()
    print("tous les tests passent")
