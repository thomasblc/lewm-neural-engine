"""Tests de analyse_exp018.py sur de vrais fichiers (EXP-005, EXP-010, EXP-003 copiés) et des bras fabriqués à effet connu.
Scénario A : int4c = GPTQ 3 bits d'EXP-010 (perte d'environ 10,8 points), SP = pleine précision, G = RTN 8 bits d'EXP-005 :
T1 à T4 soutenues attendues ; p par inversion de signe, bootstrap-t et Holm recomptés en numpy ; descriptifs recomptés.
Scénario B : int4c = RTN 8 bits (pas de perte) : T1 non soutenue et lecture correspondante. Refus des fichiers non
conformes, chacun avec son motif. À lancer sur un arbre commité.

Usage :
    python test_analyse_exp018.py
"""
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

import analyse_exp003 as A
import analyse_exp018 as A18

RES = A18.ICI.parent / "resultats"
SIGNES = np.random.default_rng(0).choice(np.array([-1.0, 1.0]), size=(10_000, 2000))
TETE = subprocess.run(["git", "-C", str(A18.ICI), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
PROPRE = subprocess.run(["git", "-C", str(A18.ICI), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip() == ""
G, G500 = A18.GRAINES, (42, 43, 44, 45, 46)
CALIB = json.loads((RES / "exp010" / "meta.json").read_text())["protocole"]["gptq_calib"]
SHA = {"int4c": "a" * 64, "int4c-sp": "b" * 64, "int4c-gptq": "c" * 64}
PROG = {"int4c": "1a" * 32, "int4c-sp": "1b" * 32, "int4c-gptq": "1c" * 32}


def ecrire(f, x, git=TETE, **champs):
    x = dict(x)
    x.update({"git": np.str_(git), "git_sale": np.bool_(False), **champs})
    np.savez_compressed(f, **x)


def bras(r, d, source, codage, spec="fp32", couches=0):
    """Un bras de 2 000 épisodes fabriqué : succès copiés de source (dossier/préfixe d'EXP-005 ou d'EXP-010)."""
    d = r / d
    d.mkdir(exist_ok=True)
    p = json.loads((r / "exp005" / "meta.json").read_text())["protocole"]
    p = dict(p, moteur={"nom": "coreml-ne", "coremltools": "9.0", "macos": "26.6.2", "precision": "float16", "codage": codage},
             **({"gptq_calib": CALIB} if spec.endswith(":gptq") else {}))
    (d / "meta.json").write_text(json.dumps({"protocole": p, "git": TETE, "git_sale": False}))
    base = spec.replace(":", "-")
    for g in G:
        sd, sp = source
        ecrire(d / f"{base}_s{g}.npz", np.load(RES / sd / f"{sp}_s{g}.npz"), spec=np.str_(spec), n_couches=np.int64(couches),
               moteur=np.str_("coreml-ne"), coreml_sha256=np.str_(SHA[codage]))
        coreml = {"placement": {"NeuralEngine": 361}, "sha256_poids": SHA[codage], "sha256_modele": PROG[codage], "codage": codage}
        (d / f"{base}_s{g}.temps.json").write_text(json.dumps({"moteur": "coreml-ne", "appels_solve": [[40.0, 50]] * 10, "coreml": coreml, "git": TETE}))


def bras500(r, d, source, codage, spec="fp32"):
    """La même version sur les 500 épisodes : succès copiés d'une version d'EXP-003, empreinte du modèle."""
    (r / d).mkdir(exist_ok=True)
    for g in G500:
        base = spec.replace(":", "-")
        ecrire(r / d / f"{base}_s{g}.npz", np.load(RES / "exp003" / f"{source}_s{g}.npz"), "9" * 40,
               coreml_sha256=np.str_(SHA[codage]))
        (r / d / f"{base}_s{g}.temps.json").write_text(json.dumps({"coreml": {"sha256_poids": SHA[codage], "sha256_modele": PROG[codage]}}))


def fabriquer(racine, int4c=("exp010", "all-3-gptq"), gptq=("exp005", "all-8")):
    r = Path(racine)
    shutil.copytree(RES / "exp005", r / "exp005", ignore=shutil.ignore_patterns("*.log"))
    shutil.copytree(RES / "exp003", r / "exp003", ignore=shutil.ignore_patterns("*.log"))
    (r / "exp010").mkdir()
    shutil.copy(RES / "exp010" / "meta.json", r / "exp010" / "meta.json")
    for d, g, sd in (("exp018-repro-pusht", 52, "exp005"), ("exp018-fin-pusht", 46, "exp003")):
        (r / d).mkdir()
        ecrire(r / d / f"fp32_s{g}.npz", np.load(r / sd / f"fp32_s{g}.npz"))
    bras(r, "exp018-int4c-pusht", int4c, "int4c")
    bras(r, "exp018-int4c-sp-pusht", ("exp005", "fp32"), "int4c-sp")
    bras(r, "exp018-gptq-pusht", gptq, "int4c-gptq", A18.GPTQ, 32)
    bras500(r, "exp014-int4c-pusht", "all-4", "int4c")
    bras500(r, "exp015-int4c-sp-pusht", "predictor-4", "int4c-sp")
    bras500(r, "exp017-gptq-pusht", "fp32", "int4c-gptq", A18.GPTQ)
    return r


def K(r, d, v="fp32"):
    return sum(np.load(Path(r) / d / f"{v.replace(':', '-')}_s{g}.npz")["success"].astype(float) for g in G)


def p_signe(y):
    return (1 + int((SIGNES @ y >= y.sum()).sum())) / 10_001


def p_boot_t(y):
    n = len(y)
    t = y.mean() / (y.std(ddof=1) / np.sqrt(n)) if y.std() > 0 else (np.inf if y.mean() > 0 else -np.inf)
    idx = np.random.default_rng(4).integers(0, n, size=(10_000, n))
    compte = 0
    for i in range(0, 10_000, 1_000):
        yb = y[idx[i:i + 1_000]]
        se = yb.std(axis=1, ddof=1) / np.sqrt(n)
        tb = np.where(se > 0, (yb.mean(axis=1) - y.mean()) / np.where(se > 0, se, 1), 0.0)
        compte += int((tb >= t).sum())
    return (1 + compte) / 10_001


def verifier_p(r, res):
    """p par inversion de signe et bootstrap-t, écarts et Holm recomptés indépendamment."""
    fp, i4, sp, g = K(r, "exp005"), K(r, "exp018-int4c-pusht"), K(r, "exp018-int4c-sp-pusht"), K(r, "exp018-gptq-pusht", A18.GPTQ)
    attendu = {"T1": (i4 - fp, -20 * (i4 - fp) - 2), "T2": (sp - fp, 20 * (sp - fp) + 2), "T3": (g - fp, 20 * (g - fp) + 2),
               "T4": (g - i4, 20 * (g - i4) - 2)}
    ps = {}
    for n, (x, y) in attendu.items():
        T = res["tests"][n]
        s, b = p_signe(y), p_boot_t(y)
        assert T["p_signe"] == s and abs(T["p_bootstrap_t"] - b) < 1e-12, (n, T["p_signe"], s, T["p_bootstrap_t"], b)
        assert abs(T["ecart_points"]["moyenne"] - 100 * x.mean() / 5) < 1e-9 and T["borne_et_verdict_concordent"], (n, T)
        ps[n] = max(s, b)
    ordre = sorted(ps, key=ps.get)
    q = np.minimum(1, np.maximum.accumulate(np.array([ps[n] for n in ordre]) * np.array([4, 3, 2, 1])))
    assert all(abs(res["tests"][n]["p_holm"] - qq) < 1e-12 for n, qq in zip(ordre, q))
    return ps, fp, i4, sp, g


def test_scenario(tmp):
    r = fabriquer(Path(tmp) / "a")
    res = A.pur(A18.analyser(r, verifier=False))
    ps, fp, i4, sp, g = verifier_p(r, res)
    assert all(res["tests"][n]["verdict"] == "soutenue" for n in ps), res["tests"]
    assert res["tests"]["T1"]["borne_points"] < -2 and res["tests"]["T4"]["borne_points"] > 2 and res["tests"]["T3"]["borne_points"] > -2
    E = res["descriptifs"]["ecarts"]
    assert abs(E["g - sp"]["moyenne"] - 100 * (g - sp).mean() / 5) < 1e-9
    p = res["descriptifs"]["parts_rattrapees"]
    assert p["rapportee"] and abs(p["g"]["part"] - (g - i4).sum() / (fp - i4).sum()) < 1e-12 and abs(p["sp"]["part"] - 1.0) < 1e-12
    assert res["lecture"]["localisation"].startswith("la perte d'int4 et sa réparation") and res["lecture"]["gptq"].startswith("fait mieux")
    e5 = res["descriptifs"]["ecarts_500_episodes"]
    fp5 = sum(np.load(r / "exp003" / f"fp32_s{s}.npz")["success"].astype(float) for s in G500)
    i45 = sum(np.load(r / "exp003" / f"all-4_s{s}.npz")["success"].astype(float) for s in G500)
    assert abs(e5["int4c - fp32"]["moyenne"] - 100 * (i45 - fp5).mean() / 5) < 1e-9
    assert res["descriptifs"]["temps_par_plan_ms"] == {"int4c": 800.0, "sp": 800.0, "g": 800.0}
    assert "| T1 (perte) | int4c contre fp32 |" in A18.rapport(res)
    print("ok scénario A : T1 à T4 soutenues, p recomptées (" + ", ".join(f"{n} {ps[n]:.4f}" for n in ps) + "), Holm, descriptifs")
    rb = fabriquer(Path(tmp) / "b", int4c=("exp005", "all-8"), gptq=("exp005", "all-6"))
    resb = A.pur(A18.analyser(rb, verifier=False))
    psb = verifier_p(rb, resb)[0]
    assert any(0.001 < v < 0.5 for v in psb.values()), psb             # au moins une p intermédiaire recomptée
    assert resb["tests"]["T1"]["verdict"] == "non soutenue" and resb["lecture"]["localisation"].startswith("perte de plus de 2 points d'int4c non établie")
    assert "sensibilite_tirages" in resb["tests"]["T3"] and len(resb["tests"]["T3"]["sensibilite_tirages"]["p"]) == 5, resb["tests"]["T3"]
    assert all(resb["tests"][n]["verdict_ecrit"] == resb["tests"][n]["verdict"] for n in resb["tests"])
    assert not resb["descriptifs"]["parts_rattrapees"]["rapportee"]
    print("ok scénario B : sans perte d'int4c, T1 non soutenue et lecture correspondante ; p recomptées (" +
          ", ".join(f"{n} {psb[n]:.4f}" for n in psb) + ")")
    return r


def test_lecture():
    L = A18.lecture
    assert L({"T1": True, "T2": True, "T3": True, "T4": True})["localisation"].startswith("la perte d'int4 et sa réparation")
    assert L({"T1": True, "T2": False, "T3": True, "T4": True})["localisation"].startswith("int4c perd encore")
    assert L({"T1": True, "T2": None, "T3": True, "T4": True})["localisation"].startswith("int4c perd encore")
    assert L({"T1": False, "T2": True, "T3": True, "T4": True})["localisation"].startswith("perte de plus de 2 points d'int4c non établie")
    assert L({"T1": None, "T2": True, "T3": True, "T4": True})["localisation"].startswith("T1 indéterminé")
    g = L({"T1": True, "T2": True, "T3": None, "T4": False})["gptq"]
    assert g.startswith("mieux de plus de 2 points non établi") and g.endswith("indéterminé (borne et verdict discordants)"), g
    assert A18.niveaux_holm([1e-4, 1e-4, 0.02, 0.9]) == [0.0125, 0.05 / 3, 0.025, 0.05]
    print("ok lectures écrites d'avance (6 cas) et niveaux de Holm à ex aequo stables")


def test_refus(tmp, r):
    def essai(motif, m):
        rr = Path(tmp) / "refus"
        shutil.rmtree(rr, ignore_errors=True)
        shutil.copytree(r, rr)
        m(rr)
        try:
            A18.analyser(rr, verifier=False)
        except A.ControleEchoue as e:
            assert motif in str(e), (motif, str(e))
            return
        raise AssertionError(f"accepté à tort : {motif}")

    def champ(d, n, cle, val):
        def m(rr):
            x = dict(np.load(rr / d / n))
            x[cle] = val(x)
            np.savez_compressed(rr / d / n, **x)
        return m

    def js(chemin, f):
        def m(rr):
            g = rr / chemin
            j = json.loads(g.read_text())
            f(j)
            g.write_text(json.dumps(j))
        return m
    GQ, I4, SP = "exp018-gptq-pusht", "exp018-int4c-pusht", "exp018-int4c-sp-pusht"
    cas = [("reproduction de exp005, graine 52", champ("exp018-repro-pusht", "fp32_s52.npz", "success", lambda x: ~x["success"])),
           ("reproduction de exp003, graine 46", lambda rr: shutil.copy(rr / "exp003" / "all-4_s46.npz", rr / "exp018-fin-pusht" / "fp32_s46.npz")),
           ("exp018-fin-pusht : autre commit", champ("exp018-fin-pusht", "fp32_s46.npz", "git", lambda x: np.str_("1" * 40))),
           ("manquant", lambda rr: (rr / GQ / "predict-4-gptq_s54.npz").unlink()),
           ("version ou graine", champ(GQ, "predict-4-gptq_s53.npz", "spec", lambda x: np.str_("predict:4"))),
           ("couches quantifiées", champ(GQ, "predict-4-gptq_s55.npz", "n_couches", lambda x: np.int64(30))),
           ("couches quantifiées", champ(SP, "fp32_s55.npz", "n_couches", lambda x: np.int64(2))),
           ("arbre git modifié", champ(I4, "fp32_s56.npz", "git_sale", lambda x: np.bool_(True))),
           ("moteur", champ(I4, "fp32_s52.npz", "moteur", lambda x: np.str_("coreml-gpu"))),
           ("différent de la référence", champ(SP, "fp32_s53.npz", "start_steps", lambda x: x["start_steps"][::-1].copy())),
           ("calibration GPTQ", js(f"{GQ}/meta.json", lambda j: j["protocole"].update(gptq_calib={"fichier": "x", "sha256": "0"}))),
           ("calibration GPTQ", js(f"{GQ}/meta.json", lambda j: j["protocole"].pop("gptq_calib"))),
           ("calibration GPTQ", js(f"{I4}/meta.json", lambda j: j["protocole"].update(gptq_calib=CALIB))),
           ("hors du Neural Engine", js(f"{GQ}/predict-4-gptq_s55.temps.json", lambda j: j["coreml"].update(placement={"CPU": 361}))),
           ("codage", js(f"{SP}/fp32_s53.temps.json", lambda j: j["coreml"].update(codage="int4c"))),
           ("moteur", js(f"{GQ}/meta.json", lambda j: j["protocole"]["moteur"].update(codage="int4c"))),
           ("protocole différent", js(f"{SP}/meta.json", lambda j: j["protocole"].update(chunk=32))),
           ("protocole différent", js(f"{I4}/meta.json", lambda j: j["protocole"].update(graine_episodes=7))),
           ("empreintes", champ(GQ, "predict-4-gptq_s54.npz", "coreml_sha256", lambda x: np.str_("x"))),
           ("modèle Core ML différent selon les graines", lambda rr: [champ(I4, "fp32_s55.npz", "coreml_sha256", lambda x: np.str_("e" * 64))(rr),
                                                                      js(f"{I4}/fp32_s55.temps.json", lambda j: j["coreml"].update(sha256_poids="e" * 64))(rr)]),
           ("modèle différent de celui de exp015-int4c-sp-pusht", lambda rr: [champ("exp015-int4c-sp-pusht", f"fp32_s{g}.npz", "coreml_sha256", lambda x: np.str_("f" * 64))(rr) for g in G500]),
           ("modèle différent de celui de exp014-int4c-pusht", lambda rr: [champ("exp014-int4c-pusht", f"fp32_s{g}.npz", "coreml_sha256", lambda x: np.str_("f" * 64))(rr) for g in G500]),
           ("modèle différent de celui de exp017-gptq-pusht", lambda rr: [js(f"exp017-gptq-pusht/predict-4-gptq_s{g}.temps.json", lambda j: j["coreml"].update(sha256_modele="0" * 64))(rr) for g in G500]),
           ("modèle Core ML différent selon les graines", js(f"{GQ}/predict-4-gptq_s53.temps.json", lambda j: j["coreml"].update(sha256_modele="0" * 64))),
           ("empreinte du modèle introuvable", js("exp014-int4c-pusht/fp32_s44.temps.json", lambda j: j["coreml"].pop("sha256_modele"))),
           ("plusieurs commits", lambda rr: [champ(SP, "fp32_s54.npz", "git", lambda x: np.str_("2" * 40))(rr),
                                              js(f"{SP}/fp32_s54.temps.json", lambda j: j.update(git="2" * 40))(rr)]),
           ("versions de macOS", js(f"{GQ}/meta.json", lambda j: j["protocole"]["moteur"].update(macos="27.0"))),
           ("fichier de temps", js(f"{I4}/fp32_s52.temps.json", lambda j: j.update(git="3" * 40)))]
    for motif, m in cas:
        essai(motif, m)
    print(f"ok refus ({len(cas)}, chacun avec son motif)")


def test_code():
    assert PROPRE, "arbre modifié : committer avant de lancer les tests"
    A18.verifier_code(TETE)
    print(f"ok code : fichiers identiques à {TETE[:7]}, calcul identique à EXP-017")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp, np.errstate(all="ignore"):
        r = test_scenario(tmp)
        test_lecture()
        test_refus(tmp, r)
        test_code()
    print("tous les tests passent")
