"""Tests de analyse_exp006.py : grille TwoRoom simulée (50 fichiers de 500 épisodes, effets connus), p et Holm
recomptés en numpy pur, refus des fichiers incohérents, vérification du code au commit. À lancer sur un arbre
commité.

Usage :
    python test_analyse_exp006.py
"""
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

import analyse_exp003 as A
import analyse_exp006 as A6

N, T = A6.N6, A6.T6
SIGNES = np.random.default_rng(0).choice(np.array([-1.0, 1.0]), size=(10_000, N))
TETE = subprocess.run(["git", "-C", str(A6.ICI), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
PROPRE = subprocess.run(["git", "-C", str(A6.ICI), "status", "--porcelain", "--untracked-files=no"],
                        capture_output=True, text=True).stdout.strip() == ""
EFFETS = {"fp32": 0.0, "all:8": 0.0, "all:6": 0.0, "all:4": -1.0, "all:3": -2.5, "encoder:4": -0.8,
          "predictor:4": 0.0, "autres:4": -0.6, "all:4:g64": -0.4}


def simuler(dossier, effets=EFFETS, sha="a" * 40):
    """Épisodes : départ à 30-200 du but ; une course réussie passe à 10 du but au pas k puis gèle (NaN)."""
    d = Path(dossier)
    d.mkdir(parents=True)
    rng = np.random.default_rng(3)
    base = rng.normal(1.5, 1.5, N)                       # facilité de l'épisode (logit)
    but = rng.uniform(50, 200, (N, 2))
    ang = rng.uniform(0, 2 * np.pi, N)
    u = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    d0 = rng.uniform(30, 200, N)
    ep, st = rng.permutation(10_000)[:N].tolist(), rng.integers(0, 60, N).tolist()
    meta = {"protocole": {"n": N, "chunk": 50, "episodes_idx": ep, "start_steps": st, "tache": "tworoom"},
            "git": sha, "git_sale": False}
    (d / "meta.json").write_text(json.dumps(meta))
    t = np.arange(T)[:, None]
    for v, g in [(v, g) for v in A6.VERSIONS6 for g in A6.GA] + [("fp32", g) for g in A6.GB]:
        r = np.random.default_rng([g, list(A6.VERSIONS6).index(v)])
        s = r.uniform(size=N) < 1 / (1 + np.exp(-(base + effets[v])))
        k = np.where(s, r.integers(0, 40, N), -1)
        fin = np.where(s, 10.0, r.uniform(17, 60, N))    # distance au pas k (succès) ou minimale (échec)
        kk = np.where(s, k, T - 1)
        frac = np.where(kk[None] == 0, 1.0, t / np.maximum(kk, 1)[None]).clip(0, 1)   # atteint `fin` au pas kk
        dist = d0[None] + (fin - d0)[None] * frac
        etat = but[None] + dist[..., None] * u[None]
        etat[s[None] & (t > k[None])] = np.nan
        n_steps = np.array([k[c * 50:(c + 1) * 50].max() + 1 if s[c * 50:(c + 1) * 50].all() else T for c in range(N // 50)],
                           dtype=np.int32)
        np.savez_compressed(d / f"{v.replace(':', '-')}_s{g}.npz", spec=v, seed=g, n_couches=A6.COUCHES[v.split(":")[0]],
                            git=sha, git_sale=False, positions=np.arange(N), episodes_idx=np.array(ep),
                            start_steps=np.array(st), chunk=50, state=etat.astype(np.float32),
                            env_state_dist=np.zeros((T, N)), success=s, first_success=k.astype(np.int32),
                            n_steps=n_steps, init_state=(but + d0[:, None] * u).astype(np.float32),
                            goal_state=but.astype(np.float32))
    f = Path(str(d) + "-fin")
    f.mkdir()
    shutil.copy(d / "all-4_s42.npz", f / "all-4_s42.npz")
    return d, f


def p_numpy(x, sens):
    obs, perm = x.sum(), SIGNES @ x
    return (1 + int(((perm <= obs) if sens < 0 else (perm >= obs)).sum())) / 10_001


def test_tests(tmp):
    d, f = simuler(Path(tmp) / "effet")
    r = A6.analyser(d, f, verifier=False)
    K = {v: np.stack([np.load(d / f"{v.replace(':', '-')}_s{g}.npz")["success"] for g in A6.GA]).sum(0).astype(float)
         for v in A6.VERSIONS6}
    ps = []
    for nom, v, ref, sens in A6.TESTS:
        x = K[v] - K[ref]
        ps.append(p_numpy(x, sens))
        assert r["tests"][nom]["p"] == ps[-1] and r["tests"][nom]["somme"] == int(x.sum()), nom
    ordre, holm, cour = np.argsort(ps), [0.0] * 4, 0.0
    for rang, i in enumerate(ordre):
        cour = max(cour, min(1.0, (4 - rang) * ps[i]))
        holm[i] = cour
    for (nom, *_), q in zip(A6.TESTS, holm):
        assert abs(r["tests"][nom]["p_holm"] - q) < 1e-12 and r["tests"][nom]["verdict"] == ("soutenue" if q < 0.05 else "non soutenue")
    assert all(r["tests"][n]["verdict"] == "soutenue" for n in ("R1", "R2", "R3")), r["tests"]
    assert abs(r["descriptifs"]["all:4"]["reussite"] - K["all:4"].mean() / 5) < 1e-12
    assert r["controles"]["fichiers"] == 50 and "R4" in A6.rapport(r)
    # sans effet : versions identiques, aucune hypothèse soutenue
    d2, f2 = simuler(Path(tmp) / "nul", effets={v: 0.0 for v in A6.VERSIONS6})
    r2 = A6.analyser(d2, f2, verifier=False)
    assert not any(t["verdict"] == "soutenue" for t in r2["tests"].values()), r2["tests"]
    print("ok tests : p par inversion de signe et Holm égaux au recomptage numpy, verdicts, effets retrouvés, rien sans effet")
    return d, f


def test_refus(tmp, d, f):
    def essai(motif, modifier):
        dd, ff = Path(tmp) / "refus", Path(tmp) / "refus-fin"
        shutil.rmtree(dd, ignore_errors=True)
        shutil.rmtree(ff, ignore_errors=True)
        shutil.copytree(d, dd)
        shutil.copytree(f, ff)
        modifier(dd, ff)
        try:
            A6.analyser(dd, ff, verifier=False)
        except A.ControleEchoue as e:
            assert motif in str(e), (motif, str(e))
            return
        raise AssertionError(f"accepté à tort : {motif}")

    def champ(nom, cle, valeur_de):
        def m(dd, ff):
            x = dict(np.load(dd / nom))
            x[cle] = valeur_de(x)
            np.savez_compressed(dd / nom, **x)
        return m

    def nan_avant(x):
        e = x["state"].copy()
        i = int(np.where(~x["success"])[0][0])
        e[5, i] = np.nan
        return e

    def succes_sans_distance(x):
        s = x["success"].copy()
        i = int(np.where(~s)[0][0])
        s[i] = True
        x["first_success"][i] = 49
        return s

    cas = [
        ("fichiers manquants", lambda dd, ff: (dd / "autres-4_s44.npz").unlink()),
        ("NaN ailleurs", champ("all-3_s43.npz", "state", nan_avant)),
        ("distance et succès diffèrent", champ("encoder-4_s45.npz", "success", succes_sans_distance)),
        ("épisodes différents", champ("all-8_s46.npz", "start_steps", lambda x: x["start_steps"][::-1].copy())),
        ("goal_state différent", champ("all-6_s42.npz", "goal_state", lambda x: x["goal_state"] + 1)),
        ("couches quantifiées", champ("all-4-g64_s43.npz", "n_couches", lambda x: np.int64(73))),
        ("arbre git modifié", champ("predictor-4_s44.npz", "git_sale", lambda x: np.bool_(True))),
        ("plusieurs commits", champ("fp32_s47.npz", "git", lambda x: np.str_("b" * 40))),
        ("sanité de clôture", lambda dd, ff: shutil.copy(dd / "all-3_s42.npz", ff / "all-4_s42.npz")),
        ("pas le protocole TwoRoom", lambda dd, ff: (dd / "meta.json").write_text(
            json.dumps({**json.loads((dd / "meta.json").read_text()), "protocole": {"n": N, "tache": "pusht"}}))),
    ]
    for motif, m in cas:
        essai(motif, m)
    print(f"ok contrôles bloquants ({len(cas)} refus, chacun avec son motif)")


def test_code():
    assert PROPRE, "arbre modifié : committer avant de lancer les tests"
    A6.verifier_code(TETE)
    try:
        A6.verifier_code("0" * 40)
        raise AssertionError("commit inconnu accepté")
    except A.ControleEchoue as e:
        assert "inconnu" in str(e)
    print(f"ok code : fichiers d'analyse et de calcul identiques à {TETE[:7]} ; commit inconnu refusé")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp, np.errstate(all="ignore"):
        d, f = test_tests(tmp)
        test_refus(tmp, d, f)
        test_code()
    print("tous les tests passent")
