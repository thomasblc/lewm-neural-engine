"""Tests de analyse_exp021.py (amendement 1 : graines 52 à 55, 1 000 premières positions, bras B0) sur de vrais fichiers
d'EXP-005 et EXP-010 copiés, en bras fabriqués à effet connu.
Scénario A : A = pleine précision, B et B0 = RTN 8 bits, C et D = GPTQ 3 bits d'EXP-010 (environ 11 points de moins) : H1 à
H3 soutenues. Scénario B, p intermédiaires : A = RTN 6 bits, B = pleine précision (A - B environ -1,5), C = GPTQ 4 bits
(A - C environ +0,8), D = GPTQ 3 bits : H2 seule soutenue. Scénario C : B réglé dégradé (GPTQ 3 bits d'EXP-010) et B0 =
pleine précision : la lecture de H3 le signale. p par inversion de signe, bootstrap-t et Holm recomptés en numpy ; écarts
par réglage recomptés. Essai court (50 épisodes). Refus des fichiers non conformes, chacun avec son motif.

Usage :
    python test_analyse_exp021.py
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
import analyse_exp021 as A21

RES = A21.ICI.parent / "resultats"
TETE = subprocess.run(["git", "-C", str(A21.ICI), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
G = A21.GRAINES
SIGNES = np.random.default_rng(0).choice(np.array([-1.0, 1.0]), size=(10_000, A21.N))
SCENARIO_A = {"A": ("exp005", "fp32"), "B": ("exp005", "all-8"), "B0": ("exp005", "all-8"), "C": ("exp010", "all-3-gptq"),
              "D": ("exp010", "all-3-gptq"), "E": ("exp005", "all-6")}
SCENARIO_B = {"A": ("exp005", "all-6"), "B": ("exp005", "fp32"), "B0": ("exp010", "all-4-gptq"), "C": ("exp010", "all-4-gptq"),
              "D": ("exp010", "all-3-gptq"), "E": ("exp005", "all-8")}
SCENARIO_C = dict(SCENARIO_A, B=("exp010", "all-3-gptq"), B0=("exp005", "fp32"))


def sha(f):
    return hashlib.sha256(Path(f).read_bytes()).hexdigest()


def ecrire(f, x, **champs):
    x = dict(x)
    x.update({"git": np.str_(TETE), "git_sale": np.bool_(False), **champs})
    np.savez_compressed(f, **x)


def premiers(x, k):
    """Les k premières positions d'un fichier, champs scalaires compris (comme run_grid --premiers)."""
    out = A21.tranche(x, k, int(x["chunk"]))
    for c in ("spec", "seed", "n_couches", "chunk"):
        out[c] = x[c]
    out["positions"] = np.arange(k)
    return out


def fabriquer(racine, sources, essai=False, traj_donnees=tuple(range(-32, 0)), pas=None, par_appel=8, apres=2e-4):
    r = Path(racine)
    n = A21.N_ESSAI if essai else A21.N
    pre = "essai-exp021" if essai else "exp021"
    for d in ("exp005", "exp010"):
        (r / d).mkdir()
        shutil.copy(RES / d / "meta.json", r / d / "meta.json")
    for g in G:
        shutil.copy(RES / "exp005" / f"fp32_s{g}.npz", r / "exp005" / f"fp32_s{g}.npz")
    p = json.loads((RES / "exp005" / "meta.json").read_text())["protocole"]
    p = dict(p, gptq_calib=json.loads((RES / "exp010" / "meta.json").read_text())["protocole"]["gptq_calib"],
             affinage_dossier=f"{pre}-collecte", premiers=n)
    d, c = r / f"{pre}-pusht", r / f"{pre}-collecte"
    d.mkdir()
    c.mkdir()
    (d / "meta.json").write_text(json.dumps({"protocole": p, "git": TETE, "git_sale": False}))
    lignes = A21.DONNEES["lignes"]
    torch.save({"X": {T: torch.zeros(m, T, 2) for T, m in lignes.items()}, "C": {T: torch.zeros(m, T, 2) for T, m in lignes.items()},
                "episodes_idx": list(traj_donnees), "graine": 0, "par_appel": par_appel, "tache": "pusht", "git": TETE},
               c / "affinage-donnees.pt")
    sd = sha(c / "affinage-donnees.pt")
    for cle, (spec, gardes) in A21.BRAS.items():
        for g in G:
            sha_aff = ""
            if spec.endswith(":aff"):
                f_aff = c / f"{A21.base(spec).replace(':', '-')}_g{g}.pt"
                torch.save({"spec": A21.base(spec), "graine": g, "donnees_sha256": sd,
                            "pas": pas or (A21.PAS_ESSAI if essai else A21.PAS), "lot": 256, "lr": 1e-3,
                            "controle_avant": 6e-3, "controle_apres": apres, "etat": {}}, f_aff)
                sha_aff = sha(f_aff)
            src_d, src_v = sources[cle]
            x = premiers(dict(np.load(RES / src_d / f"{src_v}_s{g}.npz")), n)
            ecrire(d / A21.nom(spec, g), x, spec=np.str_(spec), seed=np.int64(g), n_couches=np.int64(30),
                   n_gardes=np.int64(gardes), affinage_sha256=np.str_(sha_aff), n_poids_quantifies=np.int64(A21.N_POIDS),
                   n_lignes_quantifiees=np.int64(A21.LIGNES))
    k = A21.K_REPRO_ESSAI if essai else A21.K_REPRO
    for dd, g in ((f"{pre}-repro-debut", G[0]), (f"{pre}-repro-fin", G[-1])):
        (r / dd).mkdir()
        ecrire(r / dd / f"fp32_s{g}.npz", premiers(dict(np.load(RES / "exp005" / f"fp32_s{g}.npz")), k),
               affinage_sha256=np.str_(""), n_gardes=np.int64(0))
    return r


def K(sources, cle, n=A21.N, graines=G):
    d, v = sources[cle]
    return sum(np.load(RES / d / f"{v}_s{g}.npz")["success"][:n].astype(np.int64) for g in graines)


def p_boot_t(y, b=10_000, graine=4):
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    m, se = y.mean(), y.std(ddof=1) / np.sqrt(n)
    yb = y[np.random.default_rng(graine).integers(0, n, size=(b, n))]
    tb = (yb.mean(axis=1) - m) / (yb.std(axis=1, ddof=1) / np.sqrt(n))
    return (1 + int((tb >= m / se).sum())) / (b + 1)


def recompter(sources):
    """p de chaque test et Holm, recomptés sans le code de l'analyse."""
    Kf = sum(np.load(RES / "exp005" / f"fp32_s{g}.npz")["success"][:A21.N].astype(np.int64) for g in G)
    Kx = {c: K(sources, c) for c in A21.BRAS}
    Kx["fp32"] = Kf
    ps = []
    for t, b, ref, s in A21.TESTS:
        x = (Kx[b] - Kx[ref]).astype(np.float64)
        y = 100 * x + (8 if s == "non_inferiorite" else -8)
        with np.errstate(all="ignore"):                  # faux avertissements de matmul (Accelerate), voir analyse_exp003
            p_s = (1 + int((SIGNES @ y >= y.sum()).sum())) / 10_001
        ps.append(max(p_s, p_boot_t(y)))
    ordre = sorted(range(3), key=lambda i: (ps[i], i))
    holm, courant = [0.0] * 3, 0.0
    for rang, i in enumerate(ordre):
        courant = max(courant, min(1.0, (3 - rang) * ps[i]))
        holm[i] = courant
    return ps, holm, Kx


def verifier_scenario(r, sources, attendus):
    res = A21.analyser(r, verifier=False)
    ps, holm, Kx = recompter(sources)
    for (t, *_), p, q in zip(A21.TESTS, ps, holm):
        assert abs(res["tests"][t]["p"] - p) < 1e-12, (t, res["tests"][t]["p"], p)
        assert abs(res["tests"][t]["p_holm"] - q) < 1e-12, (t, res["tests"][t]["p_holm"], q)
        assert res["tests"][t]["verdict_ecrit"] == attendus[t], (t, res["tests"][t]["verdict_ecrit"], attendus[t])
    for c in A21.BRAS:
        assert abs(res["descriptifs"]["reussite"][c] - Kx[c].mean() / 4) < 1e-12
    e = res["descriptifs"]["ecarts"]["A - B"]["moyenne"]
    assert abs(e - 100 * (Kx["A"] - Kx["B"]).mean() / 4) < 1e-9
    for t, b, ref, _ in A21.TESTS:
        attendu = [100 * (K(sources, b, graines=(g,)) - K(sources, ref, graines=(g,))).mean() for g in G]
        assert np.allclose(list(res["tests"][t]["ecart_par_reglage_points"].values()), attendu), t
    return res, ps


def refus(sources, motif, modifier, **kw):
    with tempfile.TemporaryDirectory() as t:
        r = fabriquer(t, sources, **kw)
        modifier(r)
        try:
            A21.analyser(r, verifier=False)
        except A.ControleEchoue as e:
            assert motif in str(e), (motif, str(e))
            return
        raise AssertionError(f"accepté : {motif}")


def modifier_npz(f, **champs):
    x = dict(np.load(f))
    x.update(champs)
    np.savez_compressed(f, **x)


def main():
    with tempfile.TemporaryDirectory() as t:
        res, ps = verifier_scenario(fabriquer(t, SCENARIO_A), SCENARIO_A, {"H1": "soutenue", "H2": "soutenue", "H3": "soutenue"})
        assert "ne perd pas plus de 2 points" in res["lecture"]["H3"]
        print(f"ok scénario A : H1 à H3 soutenues, p {', '.join(f'{p:.4f}' for p in ps)}, Holm et descriptifs recomptés")
    with tempfile.TemporaryDirectory() as t:
        res, ps = verifier_scenario(fabriquer(t, SCENARIO_B), SCENARIO_B,
                                    {"H1": "non soutenue", "H2": "soutenue", "H3": "non soutenue"})
        assert 0.01 < ps[2] < 0.9, ps                      # H3 : p intermédiaire, sensible à la marge
        assert "non établie" in res["lecture"]["H3"] and "non établi" in res["lecture"]["H1"]
        assert res["tests"]["H3"]["borne_points"] < -2 < res["tests"]["H3"]["ecart_points_descriptif"]["moyenne"]
        print(f"ok scénario B : H2 seule soutenue, p {', '.join(f'{p:.4f}' for p in ps)} (H3 intermédiaire), lecture")
    with tempfile.TemporaryDirectory() as t:
        res, ps = verifier_scenario(fabriquer(t, SCENARIO_C), SCENARIO_C, {"H1": "soutenue", "H2": "soutenue", "H3": "soutenue"})
        assert "que le réglage a dégradé" in res["lecture"]["H3"], res["lecture"]["H3"]
        assert "chacun des 4 réglages" in res["lecture"]["H1"]
        print("ok scénario C : B réglé dégradé signalé dans la lecture de H3")
    with tempfile.TemporaryDirectory() as t:
        res = A21.analyser(fabriquer(t, SCENARIO_A, essai=True), essai=True, verifier=False)
        assert res["controles"]["episodes"] == A21.N_ESSAI
        print("ok essai court : 50 épisodes, 50 pas de réglage, contrôles passés")

    sp = lambda r: r / "exp021-pusht"
    cas = [
        ("poids gardés", lambda r: modifier_npz(sp(r) / A21.nom(A21.BRAS["A"][0], 52), n_gardes=np.int64(107_639))),
        ("arbre git modifié", lambda r: modifier_npz(sp(r) / A21.nom(A21.BRAS["C"][0], 53), git_sale=np.bool_(True))),
        ("episodes_idx différent", lambda r: modifier_npz(
            sp(r) / A21.nom(A21.BRAS["B"][0], 52), episodes_idx=np.roll(np.load(RES / "exp005" / "fp32_s52.npz")["episodes_idx"], 1))),
        ("réglage chargé différent", lambda r: modifier_npz(sp(r) / A21.nom(A21.BRAS["E"][0], 53), affinage_sha256=np.str_("0" * 64))),
        ("réglage chargé alors", lambda r: modifier_npz(sp(r) / A21.nom(A21.BRAS["C"][0], 52), affinage_sha256=np.str_("0" * 64))),
        ("protocole différent", lambda r: (sp(r) / "meta.json").write_text(json.dumps(
            {"protocole": dict(json.loads((sp(r) / "meta.json").read_text())["protocole"], chunk=25)}))),
        ("diffère", lambda r: modifier_npz(r / "exp021-repro-fin" / "fp32_s55.npz",
                                           success=~np.load(r / "exp021-repro-fin" / "fp32_s55.npz")["success"])),
        ("couches quantifiées", lambda r: modifier_npz(sp(r) / A21.nom(A21.BRAS["D"][0], 52), n_couches=np.int64(29))),
        ("poids et", lambda r: modifier_npz(sp(r) / A21.nom(A21.BRAS["B0"][0], 54), n_poids_quantifies=np.int64(10))),
        ("premiers", lambda r: (sp(r) / "meta.json").write_text(json.dumps(
            {"protocole": dict(json.loads((sp(r) / "meta.json").read_text())["protocole"], premiers=2000)}))),
    ]
    for motif, f in cas:
        refus(SCENARIO_A, motif, f)
    ev = int(np.load(RES / "exp005" / "fp32_s52.npz")["episodes_idx"][7])
    refus(SCENARIO_A, "trajectoires communes", lambda r: None, traj_donnees=tuple(range(-31, 0)) + (ev,))
    refus(SCENARIO_A, "réglage 4000 pas", lambda r: None, pas=4000)
    refus(SCENARIO_A, "lignes par appel", lambda r: None, par_appel=4)
    refus(SCENARIO_A, "pas de baisse", lambda r: None, apres=7e-3)
    print(f"ok refus : {len(cas) + 4} fichiers non conformes refusés, chacun avec son motif")
    print("tous les tests passent")


if __name__ == "__main__":
    main()
