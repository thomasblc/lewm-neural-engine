"""Tests de analyse_exp017.py sur de vrais fichiers de référence (EXP-003 copiés) et des fichiers fabriqués à effet connu.
Scénario A : G = copie de la pleine précision, int4c = RTN 4 bits d'EXP-003 (G1 et G2 soutenues attendues) ; p par
inversion de signe, bootstrap-t et Holm recomptés en numpy ; descriptifs recomptés ; refus des fichiers non conformes,
chacun avec son motif. Constantes codées lues dans constantes.json du paquet fabriqué. À lancer sur un arbre commité.

Usage :
    python test_analyse_exp017.py
"""
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

import analyse_exp003 as A
import analyse_exp017 as A17

RES = A17.ICI.parent / "resultats"
SIGNES = np.random.default_rng(0).choice(np.array([-1.0, 1.0]), size=(10_000, 500))
TETE = subprocess.run(["git", "-C", str(A17.ICI), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
PROPRE = subprocess.run(["git", "-C", str(A17.ICI), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip() == ""
G = A17.GRAINES
GIT12, GIT14, GIT15 = "e" * 40, "d" * 40, "c" * 40
CALIB = {"fichier": "calib.pt", "sha256": "f" * 64}
NOMS_BLOCS = [f"blocs_{i}_{n}_weight_to_fp16" for i in range(6) for n in ("mod", "qkv", "sortie", "fc1", "fc2")]
CONST = {"int4c-gptq": [["c", n, [192, 192, 1, 1]] for n in NOMS_BLOCS] + [["c", "const_7_to_fp16", [2048, 192, 1, 1]],
         ["c", "proj_3_weight_to_fp16", [192, 2048, 1, 1]], ["c", "blocs_0_somme_tetes_to_fp16", [16, 1024, 1, 1]],
         ["c", "blocs_0_etale_tetes_to_fp16", [1024, 16, 1, 1]]]}
OCTETS = {"fp16": 23_232_128, "int4c": 5_945_536, "int4c-sp": 7_120_576, "int4c-gptq": 5_945_536}
A17.constantes_codees = lambda paquet: json.loads((Path(paquet) / "constantes.json").read_text())


def ecrire(f, x, git=TETE, **champs):
    x = dict(x)
    x.update({"git": np.str_(git), "git_sale": np.bool_(False), **champs})
    np.savez_compressed(f, **x)


def proto(r):
    return json.loads((r / "exp003" / "meta.json").read_text())["protocole"]


def bras(r, d, source, git, codage=None, spec="fp32", couches=0, graines=G, gptq=False):
    """Un dossier fabriqué : succès copiés de la version source d'EXP-003 ; codage None = run PyTorch."""
    d = r / d
    d.mkdir(exist_ok=True)
    p = dict(proto(r), **({"gptq_calib": CALIB} if gptq else {}))
    if codage is not None:
        p["moteur"] = {"nom": "coreml-ne", "coremltools": "9.0", "macos": "26.6.2", "precision": "float16", **({} if codage == "fp16" else {"codage": codage})}
    (d / "meta.json").write_text(json.dumps({"protocole": p, "git": git, "git_sale": False}))
    sha = f"sha-{codage}" * 4
    base = spec.replace(":", "-")
    for g in graines:
        champs = {"spec": np.str_(spec), "n_couches": np.int64(couches)}
        if codage is not None:
            champs.update(moteur=np.str_("coreml-ne"), coreml_sha256=np.str_(sha))
        ecrire(d / f"{base}_s{g}.npz", np.load(r / "exp003" / f"{source}_s{g}.npz"), git, **champs)
        if codage is None:
            continue
        coreml = {"placement": {"NeuralEngine": 361}, "sha256_poids": sha, **({} if codage == "fp16" else {"codage": codage})}
        (d / f"{base}_s{g}.temps.json").write_text(json.dumps({"moteur": "coreml-ne", "appels_solve": [[40.0, 50]] * 10, "coreml": coreml, "git": git}))
        w = d / f"{base}_s{g}.mlpackage" / "Data" / "weights"
        w.mkdir(parents=True, exist_ok=True)
        with open(w / "weight.bin", "wb") as f:
            f.truncate(OCTETS[codage])
        if codage in CONST:
            (d / f"{base}_s{g}.mlpackage" / "constantes.json").write_text(json.dumps(CONST[codage]))


def fabriquer(racine):
    r = Path(racine)
    shutil.copytree(RES / "exp003", r / "exp003", ignore=shutil.ignore_patterns("*.log"))
    (r / "exp009-pusht").mkdir()
    (r / "exp009-pusht" / "meta.json").write_text(json.dumps({"protocole": dict(proto(r), gptq_calib=CALIB), "git": "9" * 40}))
    for pre, git in (("exp017", TETE), ("exp012", GIT12), ("exp014", GIT14), ("exp015", GIT15)):
        for d, g in ((f"{pre}-repro-pusht", 42), (f"{pre}-fin-pusht", 46)):
            (r / d).mkdir()
            ecrire(r / d / f"fp32_s{g}.npz", np.load(r / "exp003" / f"fp32_s{g}.npz"), git)
    bras(r, "exp012-ne-pusht", "fp32", GIT12, "fp16")
    bras(r, "exp014-int4c-pusht", "all-4", GIT14, "int4c")
    bras(r, "exp015-int4c-sp-pusht", "predictor-4", GIT15, "int4c-sp")
    bras(r, "exp015-sim-pusht", "all-5", GIT15, spec="predict:4", couches=32)
    bras(r, "exp017-ne16-pusht", "fp32", TETE, "fp16", graines=(42,))
    bras(r, "exp017-int4c-pusht", "all-4", TETE, "int4c", graines=(42,))
    bras(r, "exp017-gptq-pusht", "fp32", TETE, "int4c-gptq", spec="predict:4:gptq", couches=32, gptq=True)
    bras(r, "exp017-sim-pusht", "all-6", TETE, spec="predict:4:gptq", couches=32, gptq=True)
    (r / "exp017-etape0").mkdir()
    (r / "exp017-etape0" / "resultats.json").write_text(json.dumps({"git": TETE, "git_sale": False, "codes_differents": 0, "codes": 11550720,
        "ecart_decode_max_part_ligne": 9e-4, "coremltools_part_codes_changes": 0.0043, "placement": {"NeuralEngine": 361},
        "sha256_poids": "sha-int4c-gptq" * 4}))
    return r


def K(r, d, v="fp32"):
    return sum(np.load(Path(r) / d / f"{v.replace(':', '-')}_s{g}.npz")["success"].astype(float) for g in G)


def p_signe(y):
    return (1 + int((SIGNES @ y >= y.sum()).sum())) / 10_001


def p_boot_t(y):
    n, rng = len(y), np.random.default_rng(4)
    t = y.mean() / (y.std(ddof=1) / np.sqrt(n)) if y.std() > 0 else (np.inf if y.mean() > 0 else -np.inf)
    idx = rng.integers(0, n, size=(10_000, n))
    compte = 0
    for i in range(0, 10_000, 2_000):
        yb = y[idx[i:i + 2_000]]
        se = yb.std(axis=1, ddof=1) / np.sqrt(n)
        tb = np.where(se > 0, (yb.mean(axis=1) - y.mean()) / np.where(se > 0, se, 1), 0.0)
        compte += int((tb >= t).sum())
    return (1 + compte) / 10_001


def test_scenario(tmp):
    r = fabriquer(Path(tmp) / "a")
    res = A.pur(A17.analyser(r, verifier=False))
    g, ne, i4 = K(r, "exp017-gptq-pusht", "predict:4:gptq"), K(r, "exp012-ne-pusht"), K(r, "exp014-int4c-pusht")
    ps = []
    for n, x, y in (("G1", g - ne, 20 * (g - ne) + 2), ("G2", g - i4, 20 * (g - i4) - 2)):
        T = res["tests"][n]
        s, b = p_signe(y), p_boot_t(y)
        assert T["p_signe"] == s and abs(T["p_bootstrap_t"] - b) < 1e-12 and abs(T["ecart_points"]["moyenne"] - 100 * x.mean() / 5) < 1e-9, (n, T)
        assert T["borne_et_verdict_concordent"], T
        ps.append(max(s, b))
    q = np.minimum(1, np.maximum.accumulate(np.sort(ps) * np.array([2, 1])))
    assert all(abs(res["tests"][n]["p_holm"] - qq) < 1e-12 for n, qq in zip(np.array(["G1", "G2"])[np.argsort(ps)], q))
    assert res["tests"]["G1"]["verdict"] == "soutenue" and res["tests"]["G2"]["verdict"] == "soutenue" and res["phrase_visee_soutenue"]
    gs, fp, spr, sp = K(r, "exp017-sim-pusht", "predict:4:gptq"), K(r, "exp003"), K(r, "exp015-sim-pusht", "predict:4"), K(r, "exp015-int4c-sp-pusht")
    E = res["descriptifs"]["ecarts"]
    for k, v in (("gs - fp32", gs - fp), ("gs - s_pr", gs - spr), ("g - sp", g - sp), ("puce contre simulation", g - ne - gs + fp)):
        assert abs(E[k]["moyenne"] - 100 * v.mean() / 5) < 1e-9, k
    assert res["controles"]["ne16_identique"] and res["controles"]["int4c_identique"] and res["descriptifs"]["temps_par_plan_g_ms"] == 800.0
    assert "| G2 (superiorite) | g contre int4c |" in A17.rapport(res)
    print(f"ok scénario : G1 et G2 soutenues, p recomptées ({ps[0]:.4f}, {ps[1]:.4f}), Holm, descriptifs")
    return r


def test_refus(tmp, r):
    def essai(motif, m):
        rr = Path(tmp) / "refus"
        shutil.rmtree(rr, ignore_errors=True)
        shutil.copytree(r, rr)
        m(rr)
        try:
            A17.analyser(rr, verifier=False)
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
    GQ, SIM = "exp017-gptq-pusht", "exp017-sim-pusht"
    cas = [("reproduction de exp003, graine 42", lambda rr: shutil.copy(rr / "exp003" / "all-4_s42.npz", rr / "exp017-repro-pusht" / "fp32_s42.npz")),
           ("exp017-fin-pusht : autre commit", champ("exp017-fin-pusht", "fp32_s46.npz", "git", lambda x: np.str_("1" * 40))),
           ("manquant", lambda rr: (rr / GQ / "predict-4-gptq_s44.npz").unlink()),
           ("version ou graine", champ(GQ, "predict-4-gptq_s43.npz", "spec", lambda x: np.str_("predict:4"))),
           ("couches quantifiées", champ(SIM, "predict-4-gptq_s45.npz", "n_couches", lambda x: np.int64(30))),
           ("arbre git modifié", champ(GQ, "predict-4-gptq_s46.npz", "git_sale", lambda x: np.bool_(True))),
           ("moteur", champ(GQ, "predict-4-gptq_s42.npz", "moteur", lambda x: np.str_("coreml-gpu"))),
           ("calibration GPTQ", js(f"{SIM}/meta.json", lambda j: j["protocole"].update(gptq_calib={"fichier": "x", "sha256": "0"}))),
           ("calibration GPTQ", js(f"{GQ}/meta.json", lambda j: j["protocole"].pop("gptq_calib"))),
           ("hors du Neural Engine", js(f"{GQ}/predict-4-gptq_s45.temps.json", lambda j: j["coreml"].update(placement={"CPU": 361}))),
           ("codage", js(f"{GQ}/predict-4-gptq_s43.temps.json", lambda j: j["coreml"].update(codage="int4c"))),
           ("moteur", js(f"{GQ}/meta.json", lambda j: j["protocole"]["moteur"].update(codage="int4c"))),
           ("protocole différent", js(f"{SIM}/meta.json", lambda j: j["protocole"].update(chunk=32))),
           ("empreintes", champ(GQ, "predict-4-gptq_s44.npz", "coreml_sha256", lambda x: np.str_("x"))),
           ("modèle Core ML différent selon les graines", lambda rr: open(rr / GQ / "predict-4-gptq_s45.mlpackage" / "Data" / "weights" / "weight.bin", "wb").truncate(5_000_001)),
           ("constantes codées", lambda rr: (rr / GQ / "predict-4-gptq_s42.mlpackage" / "constantes.json").write_text(json.dumps(CONST["int4c-gptq"][:32]))),
           ("plusieurs commits", lambda rr: champ(SIM, "predict-4-gptq_s43.npz", "git", lambda x: np.str_("2" * 40))(rr)),
           ("reproduction de exp015, graine 46", champ("exp015-fin-pusht", "fp32_s46.npz", "success", lambda x: ~x["success"])),
           ("commit", champ("exp014-int4c-pusht", "fp32_s44.npz", "git", lambda x: np.str_("3" * 40))),
           ("exp017-ne16-pusht/fp32_s43.npz manquant", champ("exp017-ne16-pusht", "fp32_s42.npz", "positions", lambda x: x["positions"][::-1].copy())),
           ("étape 0 au commit", js("exp017-etape0/resultats.json", lambda j: j.update(git="4" * 40))),
           ("étape 0 : codes", js("exp017-etape0/resultats.json", lambda j: j.update(codes_differents=3))),
           ("étape 0 : modèle différent", js("exp017-etape0/resultats.json", lambda j: j.update(sha256_poids="x"))),
           ("versions de macOS", js(f"{GQ}/meta.json", lambda j: j["protocole"]["moteur"].update(macos="27.0")))]
    for motif, m in cas:
        essai(motif, m)
    print(f"ok refus ({len(cas)}, chacun avec son motif)")


def test_code():
    assert PROPRE, "arbre modifié : committer avant de lancer les tests"
    A17.verifier_code(TETE)
    print(f"ok code : fichiers identiques à {TETE[:7]}")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp, np.errstate(all="ignore"):
        r = test_scenario(tmp)
        test_refus(tmp, r)
        test_code()
    print("tous les tests passent")
