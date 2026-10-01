"""Tests de analyse_exp011.py sur de vrais fichiers de référence (EXP-003, EXP-006, copiés dans un dossier
temporaire) et des fichiers à 2 bits fabriqués à effet connu. Scénario A : GPTQ = copie de la pleine précision,
RTN = copie de RTN à 3 bits sur tout le modèle (gain connu, aucune perte). Scénario B : GPTQ = RTN (gain nul, perte
de près de 50 points). p et Holm recomptés en numpy ; refus des fichiers incohérents, chacun avec son motif.
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


def fabriquer(racine, gptq_comme_fp32=True):
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
        for v in A11.VERSIONS:
            for s in A11.GRAINES:
                src = "fp32" if (v.endswith("gptq") and gptq_comme_fp32) else "all-3"
                x = dict(np.load(r / c["fp32"] / f"{src}_s{s}.npz"))
                x.update(spec=np.str_(v), git=np.str_(TETE), n_couches=np.int64(A11.COUCHES))
                np.savez_compressed(g / f"{v.replace(':', '-')}_s{s}.npz", **x)
    return r


def K(r, d, v):
    return sum(np.load(Path(r) / d / f"{v.replace(':', '-')}_s{g}.npz")["success"].astype(float) for g in A11.GRAINES)


def p_numpy(y):
    return (1 + int((SIGNES @ y >= y.sum()).sum())) / 10_001


def verifier_p(r, res):
    for nom, t, quoi in A11.TESTS:
        c = A11.TACHES[t]
        g = K(r, c["dossier"], "predictor:2:gptq")
        x = g - (K(r, c["dossier"], "predictor:2") if quoi == "gptq_rtn" else K(r, c["fp32"], "fp32"))
        y = x if quoi == "gptq_rtn" else 20 * x + 3
        assert res["tests"][nom]["p"] == p_numpy(y) and res["tests"][nom]["somme"] == int(x.sum()), nom


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
    r = fabriquer(Path(tmp) / "b", gptq_comme_fp32=False)
    res = A11.analyser(r, verifier=False)
    verifier_p(r, res)
    assert [res["tests"][n]["verdict"] for n in ("G1", "G2", "N1", "N2")] == ["non soutenue"] * 4
    assert res["tests"]["G1"]["p"] == 1.0 and res["descriptifs"]["tworoom"]["part_rattrapee"]["moyenne"] == 0
    print("ok scénario B : gain nul et perte de près de 50 points, aucune hypothèse soutenue")


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
           ("init_state différent", champ("exp011-pusht", "predictor-2_s43.npz", "init_state", lambda x: x["init_state"] + 1))]
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
        test_refus(tmp, r)
        test_code()
    print("tous les tests passent")
