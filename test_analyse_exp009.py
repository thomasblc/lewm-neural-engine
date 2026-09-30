"""Tests de analyse_exp009.py sur de vrais fichiers de référence (EXP-003, 004, 006, copiés dans un dossier
temporaire) et des fichiers GPTQ fabriqués à effet connu : all:4:gptq = copie de la pleine précision (gain connu),
les autres = copies de RTN (gain nul). p et Holm recomptés en numpy, refus des fichiers incohérents.
À lancer sur un arbre commité.

Usage :
    python test_analyse_exp009.py
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
import analyse_exp009 as A9

RES = A9.ICI.parent / "resultats"
SIGNES = np.random.default_rng(0).choice(np.array([-1.0, 1.0]), size=(10_000, 500))
TETE = subprocess.run(["git", "-C", str(A9.ICI), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
PROPRE = subprocess.run(["git", "-C", str(A9.ICI), "status", "--porcelain", "--untracked-files=no"],
                        capture_output=True, text=True).stdout.strip() == ""


def fabriquer(racine):
    r = Path(racine)
    for d in ("exp003", "exp004", "exp006"):
        shutil.copytree(RES / d, r / d, ignore=shutil.ignore_patterns("*.log"))
    for d in ("exp005", "sanite-n50"):
        (r / d).mkdir()
        shutil.copy(RES / d / "meta.json", r / d / "meta.json")
    for t, c in A9.TACHES.items():
        vus = set()
        for e in c["evals"]:
            m = json.loads((r / e / "meta.json").read_text())
            vus |= set(m.get("protocole", m)["episodes_idx"])
        cal = r / f"exp009-calib-{t}"
        cal.mkdir()
        libres = [i for i in range(100_000) if i not in vus][:256]
        torch.save({"H": {}, "n": {}, "episodes_idx": libres, "start_steps": [0] * 256, "graine": 0, "git": TETE, "tache": t},
                   cal / "calib.pt")
        (r / f"exp009-repro-{t}").mkdir()
        shutil.copy(r / c["fp32"] / "fp32_s42.npz", r / f"exp009-repro-{t}" / "fp32_s42.npz")
        g = r / c["gptq"]
        g.mkdir()
        proto = json.loads((r / c["fp32"] / "meta.json").read_text())["protocole"]
        proto["gptq_calib"] = {"fichier": "calib.pt", "sha256": hashlib.sha256((cal / "calib.pt").read_bytes()).hexdigest()}
        (g / "meta.json").write_text(json.dumps({"protocole": proto, "git": TETE, "git_sale": False}))
        for v in A9.VERSIONS:
            for s in A9.GRAINES:
                src = r / (c["fp32"] if v == "all:4" else c["rtn"][v]) / f"{'fp32' if v == 'all:4' else v.replace(':', '-')}_s{s}.npz"
                x = dict(np.load(src))
                x.update(spec=np.str_(f"{v}:gptq"), git=np.str_(TETE), n_couches=np.int64(A9.COUCHES[v.split(":")[0]]))
                np.savez_compressed(g / f"{v.replace(':', '-')}-gptq_s{s}.npz", **x)
    return r


def K(r, d, v):
    return sum(np.load(Path(r) / d / f"{v.replace(':', '-')}_s{g}.npz")["success"].astype(float) for g in A9.GRAINES)


def test_tests(tmp):
    r = fabriquer(Path(tmp) / "g")
    res = A9.analyser(r, verifier=False)
    ps = []
    for nom, t, v in A9.TESTS:
        c = A9.TACHES[t]
        x = K(r, c["gptq"], f"{v}:gptq") - K(r, c["rtn"][v], v)
        p = (1 + int((SIGNES @ x >= x.sum()).sum())) / 10_001
        ps.append(p)
        assert res["tests"][nom]["p"] == p and res["tests"][nom]["somme"] == int(x.sum()), nom
    assert [res["tests"][n]["verdict"] for n in ("G1", "G2", "G3", "G4")] == ["soutenue", "soutenue", "non soutenue", "non soutenue"]
    assert res["tests"]["G3"]["somme"] == 0 and res["tests"]["G4"]["p"] == 1.0
    d = res["descriptifs"]["pusht all:4"]
    assert abs(d["part_rattrapee"]["moyenne"] - 1.0) < 1e-12 and abs(d["reussite_gptq"] - d["reussite_fp32"]) < 1e-12
    assert res["descriptifs"]["tworoom encoder:4"]["part_rattrapee"]["moyenne"] == 0
    assert "G4" in A9.rapport(res)
    print("ok tests : p et verdicts recomptés en numpy (G1, G2 soutenues ; G3, G4 nulles), part rattrapée 100 % et 0 %")
    return r


def test_refus(tmp, r):
    def essai(motif, modifier):
        rr = Path(tmp) / "refus"
        shutil.rmtree(rr, ignore_errors=True)
        shutil.copytree(r, rr)
        modifier(rr)
        try:
            A9.analyser(rr, verifier=False)
        except A.ControleEchoue as e:
            assert motif in str(e), (motif, str(e))
            return
        raise AssertionError(f"accepté à tort : {motif}")

    def recalib(t, eps):
        def m(rr):
            f = rr / f"exp009-calib-{t}" / "calib.pt"
            c = torch.load(f)
            c["episodes_idx"] = eps(rr)
            torch.save(c, f)
            g = rr / A9.TACHES[t]["gptq"] / "meta.json"
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
    cas = [("partagent des trajectoires", recalib("pusht", ep3)),
           ("calibration différente", lambda rr: torch.save({"autre": 1}, rr / "exp009-calib-tworoom" / "calib.pt")),
           ("reproduction pusht", lambda rr: shutil.copy(rr / "exp003" / "all-4_s42.npz", rr / "exp009-repro-pusht" / "fp32_s42.npz")),
           ("couches", champ("exp009-tworoom", "encoder-4-gptq_s43.npz", "n_couches", lambda x: np.int64(109))),
           ("manquant", lambda rr: (rr / "exp009-pusht" / "autres-4-gptq_s45.npz").unlink()),
           ("plusieurs commits", champ("exp009-pusht", "all-3-gptq_s44.npz", "git", lambda x: np.str_("b" * 40)))]
    for motif, m in cas:
        essai(motif, m)
    print(f"ok contrôles bloquants ({len(cas)} refus, chacun avec son motif)")


def test_code():
    assert PROPRE, "arbre modifié : committer avant de lancer les tests"
    A9.verifier_code(TETE)
    print(f"ok code : fichiers d'analyse et de calcul identiques à {TETE[:7]}")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp, np.errstate(all="ignore"):
        r = test_tests(tmp)
        test_refus(tmp, r)
        test_code()
    print("tous les tests passent")
