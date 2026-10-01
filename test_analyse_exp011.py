"""Tests de analyse_exp011.py sur de vrais fichiers de référence (EXP-003, EXP-006, copiés dans un dossier
temporaire) et des fichiers à 2 bits fabriqués à effet connu. Scénario A : GPTQ = copie de la pleine précision,
RTN = copie de RTN à 3 bits sur tout le modèle (gain connu, aucune perte). Scénario B : GPTQ = RTN (gain nul, perte
de près de 50 points). Scénario C : effets proches des seuils, p strictement entre le plancher et 1, qui changent
avec la marge et le sens du test, et une hypothèse que seule la correction de Holm rejette. p et Holm recomptés en
numpy ; refus des fichiers incohérents, chacun avec son motif.
À lancer sur un arbre commité.

Usage :
    python test_analyse_exp011.py
"""
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

import analyse_exp003 as A
import analyse_exp011 as A11

RES = A11.ICI.parent / "resultats"
SIGNES = np.random.default_rng(0).choice(np.array([-1.0, 1.0]), size=(10_000, 500))
TETE = subprocess.run(["git", "-C", str(A11.ICI), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
PROPRE = subprocess.run(["git", "-C", str(A11.ICI), "status", "--porcelain", "--untracked-files=no"],
                        capture_output=True, text=True).stdout.strip() == ""


SOURCES = {"A": {t: {"predictor:2:gptq": "fp32", "predictor:2": "all-3"} for t in ("pusht", "tworoom")},
           "B": {t: {"predictor:2:gptq": "all-3", "predictor:2": "all-3"} for t in ("pusht", "tworoom")},
           "C": {"pusht": {"predictor:2:gptq": "predictor-3", "predictor:2": "encoder-4"},
                 "tworoom": {"predictor:2:gptq": "autres-4", "predictor:2": "all-6"}}}


def fabriquer(racine, scenario="A"):
    r = Path(racine)
    for d in ("exp003", "exp006"):
        shutil.copytree(RES / d, r / d, ignore=shutil.ignore_patterns("*.log"))
    for d in ("exp005", "sanite-n50"):
        (r / d).mkdir()
        shutil.copy(RES / d / "meta.json", r / d / "meta.json")
    for t, c in A11.TACHES.items():
        vus = set()
        for e in c["evals"]:
            m = json.loads((r / e / "meta.json").read_text())
            vus |= set(m.get("protocole", m)["episodes_idx"])
        cal = r / c["calib"]
        cal.mkdir()
        libres = [i for i in range(100_000) if i not in vus][:256]
        torch.save({"H": {}, "n": {}, "episodes_idx": libres, "start_steps": [0] * 256, "graine": 0, "git": TETE, "tache": t},
                   cal / "calib.pt")
        (r / f"exp011-repro-{t}").mkdir()
        shutil.copy(r / c["fp32"] / "fp32_s42.npz", r / f"exp011-repro-{t}" / "fp32_s42.npz")
        g = r / c["dossier"]
        g.mkdir()
        proto = json.loads((r / c["fp32"] / "meta.json").read_text())["protocole"]
        proto["gptq_calib"] = {"fichier": "calib.pt", "sha256": hashlib.sha256((cal / "calib.pt").read_bytes()).hexdigest()}
        (g / "meta.json").write_text(json.dumps({"protocole": proto, "git": TETE, "git_sale": False}))
        (r / f"exp009-{t}").mkdir(exist_ok=True)
        (r / f"exp009-{t}" / "meta.json").write_text(json.dumps({"protocole": proto, "git": TETE}))
        x = dict(np.load(r / f"exp011-repro-{t}" / "fp32_s42.npz"))
        x.update(git=np.str_(TETE), git_sale=np.bool_(False))
        np.savez_compressed(r / f"exp011-repro-{t}" / "fp32_s42.npz", **x)
        for v in A11.VERSIONS:
            for s in A11.GRAINES:
                src = SOURCES[scenario][t][v]
                x = dict(np.load(r / c["fp32"] / f"{src}_s{s}.npz"))
                x.update(spec=np.str_(v), git=np.str_(TETE), n_couches=np.int64(A11.COUCHES))
                np.savez_compressed(g / f"{v.replace(':', '-')}_s{s}.npz", **x)
    return r


def K(r, d, v):
    return sum(np.load(Path(r) / d / f"{v.replace(':', '-')}_s{g}.npz")["success"].astype(float) for g in A11.GRAINES)


def p_numpy(y, sens="plus"):
    t = SIGNES @ y
    return (1 + int(((t >= y.sum()) if sens == "plus" else (np.abs(t) >= abs(y.sum()))).sum())) / 10_001


def holm_numpy(ps):
    o = np.argsort(ps)
    q = np.minimum(1, np.maximum.accumulate(np.array(ps)[o] * (len(ps) - np.arange(len(ps)))))
    out = np.empty(len(ps))
    out[o] = q
    return out


def differences(r, t, quoi):
    c = A11.TACHES[t]
    g = K(r, c["dossier"], "predictor:2:gptq")
    return g - (K(r, c["dossier"], "predictor:2") if quoi == "gptq_rtn" else K(r, c["fp32"], "fp32"))


def verifier_p(r, res):
    ps = []
    for nom, t, quoi in A11.TESTS:
        x = differences(r, t, quoi)
        y = x if quoi == "gptq_rtn" else 20 * x + 3
        ps.append(p_numpy(y))
        assert res["tests"][nom]["p"] == ps[-1] and res["tests"][nom]["somme"] == int(x.sum()), nom
    for (nom, *_), q in zip(A11.TESTS, holm_numpy(ps)):
        assert abs(res["tests"][nom]["p_holm"] - q) < 1e-12, nom
        assert res["tests"][nom]["verdict"] == ("soutenue" if q < 0.05 else "non soutenue"), nom


def test_scenario_a(tmp):
    r = fabriquer(Path(tmp) / "a")
    res = A11.analyser(r, verifier=False)
    verifier_p(r, res)
    assert all(res["tests"][n]["verdict"] == "soutenue" for n in ("G1", "G2", "N1", "N2"))
    assert res["tests"]["N1"]["somme"] == 0 and res["tests"]["N1"]["p"] == 1 / 10_001   # y = 3 partout : extrême
    for t in A11.TACHES:
        d = res["descriptifs"][t]
        assert abs(d["part_rattrapee"]["moyenne"] - 1.0) < 1e-12 and abs(d["reussite_gptq"] - d["reussite_fp32"]) < 1e-12
    p3 = K(r, "exp003", "predictor-3") / 5
    assert abs(res["descriptifs"]["pusht"]["gptq2_moins_rtn3"]["moyenne"] - (K(r, "exp003", "fp32") / 5 - p3).mean()) < 1e-12
    assert "N2" in A11.rapport(res)
    print("ok scénario A : p recomptées en numpy, 4 hypothèses soutenues, part rattrapée 100 %")
    return r


def test_scenario_b(tmp):
    r = fabriquer(Path(tmp) / "b", "B")
    res = A11.analyser(r, verifier=False)
    verifier_p(r, res)
    assert [res["tests"][n]["verdict"] for n in ("G1", "G2", "N1", "N2")] == ["non soutenue"] * 4
    assert res["tests"]["G1"]["p"] == 1.0 and res["descriptifs"]["tworoom"]["part_rattrapee"]["moyenne"] == 0
    print("ok scénario B : gain nul et perte de près de 50 points, aucune hypothèse soutenue")


def test_scenario_c(tmp):
    r = fabriquer(Path(tmp) / "c", "C")
    res = A11.analyser(r, verifier=False)
    verifier_p(r, res)
    T = res["tests"]
    assert all(1 / 10_001 < T[n]["p"] < 1 for n in T), {n: T[n]["p"] for n in T}
    assert T["G1"]["p"] < 0.05 <= T["G1"]["p_holm"]            # rejetée sans Holm, gardée avec : Holm appliqué
    assert [T[n]["verdict"] for n in ("G1", "G2", "N1", "N2")] == ["non soutenue", "non soutenue", "non soutenue", "soutenue"]
    for n, t in (("N1", "pusht"), ("N2", "tworoom")):         # la p dépend de la marge : 2 et 4 points donnent autre chose
        x = differences(r, t, "gptq_fp32")
        assert p_numpy(20 * x + 2) != T[n]["p"] != p_numpy(20 * x + 4), n
    x = differences(r, "pusht", "gptq_rtn")
    assert p_numpy(x, "bilateral") != T["G1"]["p"]            # le sens unilatéral est bien celui de l'analyse
    assert res["descriptifs"]["pusht"]["gptq2_moins_rtn3"]["moyenne"] == 0   # GPTQ = RTN à 3 bits du prédicteur
    print("ok scénario C : p intermédiaires, sensibles à la marge et au sens, Holm recompté (G1 rejetée sans Holm seulement)")


def test_refus(tmp, r):
    def essai(motif, modifier):
        rr = Path(tmp) / "refus"
        shutil.rmtree(rr, ignore_errors=True)
        shutil.copytree(r, rr)
        modifier(rr)
        try:
            A11.analyser(rr, verifier=False)
        except A.ControleEchoue as e:
            assert motif in str(e), (motif, str(e))
            return
        raise AssertionError(f"accepté à tort : {motif}")

    def recalib(t, eps):
        def m(rr):
            c = A11.TACHES[t]
            f = rr / c["calib"] / "calib.pt"
            cal = torch.load(f)
            cal["episodes_idx"] = eps(rr)
            torch.save(cal, f)
            g = rr / c["dossier"] / "meta.json"
            meta = json.loads(g.read_text())
            meta["protocole"]["gptq_calib"]["sha256"] = hashlib.sha256(f.read_bytes()).hexdigest()
            g.write_text(json.dumps(meta))
        return m

    def meta(d, f):
        def m(rr):
            g = rr / d / "meta.json"
            mm = json.loads(g.read_text())
            f(mm["protocole"])
            g.write_text(json.dumps(mm))
        return m

    def champ(d, nom, cle, val):
        def m(rr):
            x = dict(np.load(rr / d / nom))
            x[cle] = val(x)
            np.savez_compressed(rr / d / nom, **x)
        return m
    ep3 = lambda rr: json.loads((rr / "exp003" / "meta.json").read_text())["protocole"]["episodes_idx"][:5]
    ep6 = lambda rr: json.loads((rr / "exp006" / "meta.json").read_text())["protocole"]["episodes_idx"][-3:]
    cas = [("partagent des trajectoires", recalib("pusht", ep3)),
           ("partagent des trajectoires", recalib("tworoom", ep6)),
           ("calibration différente", lambda rr: torch.save({"autre": 1}, rr / "exp009-calib-tworoom" / "calib.pt")),
           ("reproduction pusht", lambda rr: shutil.copy(rr / "exp003" / "all-4_s42.npz", rr / "exp011-repro-pusht" / "fp32_s42.npz")),
           ("couches", champ("exp011-tworoom", "predictor-2-gptq_s43.npz", "n_couches", lambda x: np.int64(109))),
           ("couches", champ("exp011-pusht", "predictor-2_s46.npz", "n_couches", lambda x: np.int64(73))),
           ("manquant", lambda rr: (rr / "exp011-pusht" / "predictor-2_s45.npz").unlink()),
           ("plusieurs commits", champ("exp011-pusht", "predictor-2-gptq_s44.npz", "git", lambda x: np.str_("b" * 40))),
           ("arbre git modifié", champ("exp011-tworoom", "predictor-2_s42.npz", "git_sale", lambda x: np.bool_(True))),
           ("version ou graine", champ("exp011-pusht", "predictor-2-gptq_s42.npz", "spec", lambda x: np.str_("predictor:3:gptq"))),
           ("version ou graine", champ("exp011-tworoom", "predictor-2-gptq_s45.npz", "seed", lambda x: np.int64(44))),
           ("init_state différent", champ("exp011-pusht", "predictor-2_s43.npz", "init_state", lambda x: x["init_state"] + 1)),
           ("calibration différente de celle d'EXP-009", meta("exp009-tworoom", lambda p: p["gptq_calib"].update(sha256="0" * 64))),
           ("protocole différent", meta("exp011-pusht", lambda p: p.update(chunk=32))),
           ("protocole différent", meta("exp011-tworoom", lambda p: p.update(planification={"cem_n_steps": 10}))),
           ("reproduction tworoom : autre commit", champ("exp011-repro-tworoom", "fp32_s42.npz", "git", lambda x: np.str_("c" * 40)))]
    for motif, m in cas:
        essai(motif, m)
    print(f"ok contrôles bloquants ({len(cas)} refus, chacun avec son motif)")


def test_code():
    assert PROPRE, "arbre modifié : committer avant de lancer les tests"
    A11.verifier_code(TETE)
    print(f"ok code : fichiers d'analyse et de calcul identiques à {TETE[:7]}")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp, np.errstate(all="ignore"):
        r = test_scenario_a(tmp)
        test_scenario_b(tmp)
        test_scenario_c(tmp)
        test_refus(tmp, r)
        test_code()
    print("tous les tests passent")
