"""Tests de analyse_exp012.py sur de vrais fichiers de référence (EXP-003, EXP-006, copiés dans un dossier
temporaire) et des fichiers Core ML fabriqués à effet connu. Scénario A : chaque moteur = copie de la pleine
précision (aucune perte). Scénario C : effets proches de la marge, p strictement entre le plancher et 1, qui changent
avec la marge. p et Holm recomptés en numpy (Holm indépendant de A.holm) ; temps et puissance recomptés ; refus des
fichiers incohérents, chacun avec son motif. À lancer sur un arbre commité.

Usage :
    python test_analyse_exp012.py
"""
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

import analyse_exp003 as A
import analyse_exp012 as A12

RES = A12.ICI.parent / "resultats"
SIGNES = np.random.default_rng(0).choice(np.array([-1.0, 1.0]), size=(10_000, 500))
TETE = subprocess.run(["git", "-C", str(A12.ICI), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
PROPRE = subprocess.run(["git", "-C", str(A12.ICI), "status", "--porcelain", "--untracked-files=no"],
                        capture_output=True, text=True).stdout.strip() == ""
SOURCES = {"A": {(t, m): "fp32" for t in A12.TACHES for m in A12.MOTEURS},
           "C": {("pusht", "ne"): "all-6", ("tworoom", "ne"): "all-6", ("pusht", "gpu"): "predictor-3",
                 ("tworoom", "gpu"): "autres-4"}}
TEMPS = {"torch": 0.79, "ne": 0.88, "gpu": 0.40}         # secondes par plan, fabriquées
PUISS = {"torch": 84.0, "ne": 93.0, "gpu": 66.0}


def ecrire_temps(f, moteur, cle, g):
    rng = np.random.default_rng(g)
    tp = list(TEMPS[cle] + 0.01 * rng.standard_normal(100))
    f.with_suffix(".temps.json").write_text(json.dumps({"moteur": moteur, "temps_par_plan_s": tp,
                                                        "puissance_w": [PUISS[cle]] * 20, "git": TETE}))


def fabriquer(racine, scenario="A"):
    r = Path(racine)
    for d in A12.TACHES.values():
        shutil.copytree(RES / d, r / d, ignore=shutil.ignore_patterns("*.log"))
    for t, ref_d in A12.TACHES.items():
        rep = r / f"exp012-repro-{t}"
        rep.mkdir()
        x = dict(np.load(r / ref_d / "fp32_s42.npz"))
        x.update(git=np.str_(TETE), git_sale=np.bool_(False))
        np.savez_compressed(rep / "fp32_s42.npz", **x)
        ecrire_temps(rep / "fp32_s42.npz", "torch", "torch", 0)
        proto = json.loads((r / ref_d / "meta.json").read_text())["protocole"]
        for m, moteur in A12.MOTEURS.items():
            d = r / A12.dossier(t, m)
            d.mkdir()
            pm = dict(proto, moteur={"nom": moteur, "coremltools": "9.0", "macos": "26.6.2", "precision": "float16"})
            (d / "meta.json").write_text(json.dumps({"protocole": pm, "git": TETE, "git_sale": False}))
            for g in A12.GRAINES:
                x = dict(np.load(r / ref_d / f"{SOURCES[scenario][(t, m)]}_s{g}.npz"))
                x.update(spec=np.str_("fp32"), git=np.str_(TETE), n_couches=np.int64(0), moteur=np.str_(moteur),
                         coreml_sha256=np.str_(f"{t}-{m}" * 4))
                np.savez_compressed(d / f"fp32_s{g}.npz", **x)
                ecrire_temps(d / f"fp32_s{g}.npz", moteur, m, g)
    return r


def K(r, d, v):
    return sum(np.load(Path(r) / d / f"{v}_s{g}.npz")["success"].astype(float) for g in A12.GRAINES)


def p_numpy(y):
    return (1 + int((SIGNES @ y >= y.sum()).sum())) / 10_001


def holm_numpy(ps):
    o = np.argsort(ps)
    q = np.minimum(1, np.maximum.accumulate(np.array(ps)[o] * (len(ps) - np.arange(len(ps)))))
    out = np.empty(len(ps))
    out[o] = q
    return out


def verifier_p(r, res):
    ps = []
    for nom, t, m in A12.TESTS:
        x = K(r, A12.dossier(t, m), "fp32") - K(r, A12.TACHES[t], "fp32")
        ps.append(p_numpy(20 * x + 2))
        assert res["tests"][nom]["p"] == ps[-1] and res["tests"][nom]["somme"] == int(x.sum()), nom
    for (nom, *_), q in zip(A12.TESTS, holm_numpy(ps)):
        assert abs(res["tests"][nom]["p_holm"] - q) < 1e-12, nom
        assert res["tests"][nom]["verdict"] == ("soutenue" if q < 0.05 else "non soutenue"), nom
    return ps


def test_scenario_a(tmp):
    r = fabriquer(Path(tmp) / "a")
    res = A12.analyser(r, verifier=False)
    verifier_p(r, res)
    assert all(res["tests"][n]["verdict"] == "soutenue" and res["tests"][n]["somme"] == 0 for n in res["tests"])
    for t in A12.TACHES:
        d = res["descriptifs"][t]
        assert d["ne_moins_gpu"]["moyenne"] == 0 and d["reussite_ne"] == d["reussite_fp32"]
        for m in ("torch", "ne", "gpu"):
            s = d["temps"][m]
            assert abs(s["mediane_ms"] / 1000 - TEMPS[m]) < 0.005 and s["puissance_mediane_w"] == PUISS[m], (t, m, s)
            assert abs(s["energie_par_plan_j"] - PUISS[m] * s["mediane_ms"] / 1000) < 1e-9
        assert d["temps"]["ne"]["plans"] == 500 and d["temps"]["torch"]["plans"] == 100
    assert "Neural Engine moins GPU" in A12.rapport(res)
    print("ok scénario A : 4 hypothèses soutenues, temps, puissance et énergie recomptés")
    return r


def test_scenario_c(tmp):
    r = fabriquer(Path(tmp) / "c", "C")
    res = A12.analyser(r, verifier=False)
    ps = verifier_p(r, res)
    assert all(1 / 10_001 < p < 1 for p in ps), ps
    assert [res["tests"][n]["verdict"] for n in ("N1", "N2", "N3", "N4")] == ["soutenue"] + ["non soutenue"] * 3
    x = K(r, A12.dossier("pusht", "ne"), "fp32") - K(r, "exp003", "fp32")
    assert p_numpy(20 * x + 1) != ps[0] != p_numpy(20 * x + 3)          # la p dépend de la marge
    print("ok scénario C : p intermédiaires, sensibles à la marge, Holm recompté (seule N1 soutenue)")


def test_refus(tmp, r):
    def essai(motif, modifier):
        rr = Path(tmp) / "refus"
        shutil.rmtree(rr, ignore_errors=True)
        shutil.copytree(r, rr)
        modifier(rr)
        try:
            A12.analyser(rr, verifier=False)
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
    cas = [("reproduction pusht", lambda rr: shutil.copy(rr / "exp003" / "all-4_s42.npz", rr / "exp012-repro-pusht" / "fp32_s42.npz")),
           ("reproduction tworoom : autre commit", champ("exp012-repro-tworoom", "fp32_s42.npz", "git", lambda x: np.str_("c" * 40))),
           ("manquant", lambda rr: (rr / "exp012-ne-pusht" / "fp32_s44.npz").unlink()),
           ("version ou graine", champ("exp012-gpu-tworoom", "fp32_s43.npz", "seed", lambda x: np.int64(42))),
           ("arbre git modifié", champ("exp012-ne-tworoom", "fp32_s45.npz", "git_sale", lambda x: np.bool_(True))),
           ("couches quantifiées", champ("exp012-gpu-pusht", "fp32_s46.npz", "n_couches", lambda x: np.int64(30))),
           ("moteur", champ("exp012-ne-pusht", "fp32_s42.npz", "moteur", lambda x: np.str_("coreml-gpu"))),
           ("poids Core ML différents", champ("exp012-ne-tworoom", "fp32_s44.npz", "coreml_sha256", lambda x: np.str_("autre"))),
           ("protocole différent", meta("exp012-gpu-pusht", lambda p: p.update(chunk=32))),
           ("moteur", meta("exp012-ne-tworoom", lambda p: p["moteur"].update(precision="float32"))),
           ("init_state différent", champ("exp012-gpu-tworoom", "fp32_s42.npz", "init_state", lambda x: x["init_state"] + 1)),
           ("plusieurs commits", lambda rr: (champ("exp012-gpu-tworoom", "fp32_s45.npz", "git", lambda x: np.str_("b" * 40))(rr),
                                             temps("exp012-gpu-tworoom", "fp32_s45.temps.json", lambda j: j.update(git="b" * 40))(rr))),
           ("fichier de temps", champ("exp012-ne-pusht", "fp32_s46.npz", "git", lambda x: np.str_("d" * 40))),
           ("fichier de temps", temps("exp012-ne-pusht", "fp32_s43.temps.json", lambda j: j.update(moteur="torch"))),
           ("reproduction pusht : moteur", temps("exp012-repro-pusht", "fp32_s42.temps.json", lambda j: j.update(moteur="coreml-ne")))]
    for motif, m in cas:
        essai(motif, m)
    print(f"ok contrôles bloquants ({len(cas)} refus, chacun avec son motif)")


def test_code():
    assert PROPRE, "arbre modifié : committer avant de lancer les tests"
    A12.verifier_code(TETE)
    print(f"ok code : fichiers d'analyse et de calcul identiques à {TETE[:7]}")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp, np.errstate(all="ignore"):
        r = test_scenario_a(tmp)
        test_scenario_c(tmp)
        test_refus(tmp, r)
        test_code()
    print("tous les tests passent")
