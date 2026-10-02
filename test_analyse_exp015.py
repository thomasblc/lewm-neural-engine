"""Tests de analyse_exp015.py sur de vrais fichiers de référence (EXP-003, copiés dans un dossier temporaire) et des
fichiers fabriqués à effet connu (les bras sont des copies de versions d'EXP-003). Scénario A : aucun effet. Scénario
C : effets connus, p recomptées en numpy indépendamment du code testé (inversion de signe, bootstrap-t, Holm),
descriptifs D1 à D4 recomptés. Scénario S : références refaites dans la session (témoins différents) et bras int4c-sp
demandé par l'étape 0 b. Refus des fichiers incohérents, chacun avec son motif. Les constantes codées sont lues dans un
fichier constantes.json du paquet fabriqué (la vraie lecture est testée par test_analyse_exp014.py). À lancer sur un
arbre commité.

Usage :
    python test_analyse_exp015.py
"""
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

import analyse_exp003 as A
import analyse_exp015 as A15

RES = A15.ICI.parent / "resultats"
SIGNES = np.random.default_rng(0).choice(np.array([-1.0, 1.0]), size=(10_000, 500))
TETE = subprocess.run(["git", "-C", str(A15.ICI), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
PROPRE = subprocess.run(["git", "-C", str(A15.ICI), "status", "--porcelain", "--untracked-files=no"],
                        capture_output=True, text=True).stdout.strip() == ""
G = A15.GRAINES
GIT12, GIT14 = "e" * 40, "d" * 40
OCTETS = {"fp16": 23_232_128, "int4c": 5_945_536, "int4c-bl": 7_100_000, "int4c-pp": 22_050_000, "int4c-sp": 7_120_000}
SOURCES = {"A": {"ne16": "fp32", "int4c": "fp32", "int4c-bl": "fp32", "int4c-pp": "fp32", "int4c-sp": "fp32",
                 "predproj:4": "fp32", "predict:4": "fp32"},
           "C": {"ne16": "fp32", "int4c": "all-4", "int4c-bl": "predictor-3", "int4c-pp": "all-4", "int4c-sp": "predictor-3",
                 "predproj:4": "encoder-4", "predict:4": "all-5"}}
NOMS_BLOCS = [f"blocs_{i}_{n}_weight_to_fp16" for i in range(6) for n in ("mod", "qkv", "sortie", "fc1", "fc2")]
CONST = {"int4c-bl": [["c", n, [192, 192, 1, 1]] for n in NOMS_BLOCS],
         "int4c-pp": [["c", "const_7_to_fp16", [2048, 192, 1, 1]], ["c", "proj_3_weight_to_fp16", [192, 2048, 1, 1]]],
         "tetes": [["c", "blocs_0_somme_tetes_to_fp16", [16, 1024, 1, 1]], ["c", "blocs_0_etale_tetes_to_fp16", [1024, 16, 1, 1]]]}
CONST["int4c-sp"] = CONST["int4c-bl"] + CONST["tetes"]
CONST["int4c"] = CONST["int4c-bl"] + CONST["int4c-pp"] + CONST["tetes"]
A15.constantes_codees = lambda paquet: json.loads((Path(paquet) / "constantes.json").read_text())


def ecrire(f, x, git=TETE, **champs):
    x = dict(x)
    x.update({"git": np.str_(git), "git_sale": np.bool_(False), **champs})
    np.savez_compressed(f, **x)


def proto(r):
    return json.loads((r / "exp003" / "meta.json").read_text())["protocole"]


def bras_ne(r, d, codage, source, git, graines=G):
    """Un bras sur le Neural Engine fabriqué : succès copiés de la version source d'EXP-003."""
    d = r / d
    d.mkdir(exist_ok=True)
    mo = {"nom": "coreml-ne", "coremltools": "9.0", "macos": "26.6.2", "precision": "float16", **({} if codage == "fp16" else {"codage": codage})}
    (d / "meta.json").write_text(json.dumps({"protocole": dict(proto(r), moteur=mo), "git": git, "git_sale": False}))
    sha = f"sha-{codage}" * 4
    for g in graines:
        ecrire(d / f"fp32_s{g}.npz", np.load(r / "exp003" / f"{source.replace(':', '-')}_s{g}.npz"), git, spec=np.str_("fp32"),
               n_couches=np.int64(0), moteur=np.str_("coreml-ne"), coreml_sha256=np.str_(sha))
        coreml = {"placement": {"NeuralEngine": 361}, "sha256_poids": sha, "sha256_modele": f"m-{sha}", **({} if codage == "fp16" else {"codage": codage})}
        (d / f"fp32_s{g}.temps.json").write_text(json.dumps({"moteur": "coreml-ne", "appels_solve": [[40.0, 50]] * 10, "coreml": coreml,
                                                             "git": git}))
        p = d / f"fp32_s{g}.mlpackage" / "Data" / "weights"
        p.mkdir(parents=True, exist_ok=True)
        with open(p / "weight.bin", "wb") as f:
            f.truncate(OCTETS[codage])
        if codage != "fp16":
            (d / f"fp32_s{g}.mlpackage" / "constantes.json").write_text(json.dumps(CONST[codage]))


def fabriquer(racine, scenario="A", session=False, avec_sp=False):
    r = Path(racine)
    shutil.copytree(RES / "exp003", r / "exp003", ignore=shutil.ignore_patterns("*.log"))
    S = SOURCES[scenario]
    for d, g in (("exp015-repro-pusht", 42), ("exp015-fin-pusht", 46)):
        (r / d).mkdir()
        ecrire(r / d / f"fp32_s{g}.npz", np.load(r / "exp003" / f"fp32_s{g}.npz"))
    bras_ne(r, "exp012-ne-pusht", "fp16", S["ne16"], GIT12)
    bras_ne(r, "exp014-int4c-pusht", "int4c", S["int4c"], GIT14)
    # runs refaits dans la session ; session=True : différents des séries réutilisées, séries refaites sur 5 graines
    bras_ne(r, "exp015-ne16-pusht", "fp16", S["ne16"], TETE, G if session else (42,))
    bras_ne(r, "exp015-int4c-pusht", "int4c", S["int4c"], TETE, G if session else (42,))
    for b in ("int4c-bl", "int4c-pp") + (("int4c-sp",) if avec_sp else ()):
        bras_ne(r, f"exp015-{b}-pusht", b, S[b], TETE)
    d = r / "exp015-sim-pusht"
    d.mkdir()
    (d / "meta.json").write_text(json.dumps({"protocole": proto(r), "git": TETE, "git_sale": False}))
    for spec, n in (("predproj:4", 2), ("predict:4", 32)):
        for g in G:
            ecrire(d / A15.nom_fichier(spec, g), np.load(r / "exp003" / f"{S[spec]}_s{g}.npz"), spec=np.str_(spec), n_couches=np.int64(n))
    for g in (G if session else (42,)):
        ecrire(d / A15.nom_fichier("predictor:4", g), np.load(r / "exp003" / f"predictor-4_s{g}.npz"))
    if session:                                       # les runs refaits diffèrent des séries réutilisées
        for dd, f in (("exp015-ne16-pusht", "fp32_s42.npz"), ("exp015-int4c-pusht", "fp32_s42.npz"), ("exp015-sim-pusht", "predictor-4_s42.npz")):
            x = dict(np.load(r / dd / f))
            x["positions"] = x["positions"][::-1].copy()
            np.savez_compressed(r / dd / f, **x)
    (r / "exp015-etape0b").mkdir()
    (r / "exp015-etape0b" / "resultats.json").write_text(json.dumps({"git": TETE, "git_sale": False, "resume": {"fp16": {}},
                                                                     "sp_et_bl_identiques_sur_le_neural_engine": not avec_sp}))
    return r


def K(r, d, v="fp32"):
    return sum(np.load(Path(r) / d / f"{v.replace(':', '-')}_s{g}.npz")["success"].astype(float) for g in G)


def p_signe(y):
    return (1 + int((SIGNES @ y >= y.sum()).sum())) / 10_001


def p_boot_t(y):
    """Bootstrap-t écrit à part : t = moyenne / erreur-type, t* = (moyenne* - moyenne) / erreur-type*."""
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


def holm_numpy(ps):
    o = np.argsort(ps)
    q = np.minimum(1, np.maximum.accumulate(np.array(ps)[o] * (len(ps) - np.arange(len(ps)))))
    out = np.empty(len(ps))
    out[o] = q
    return out


def series(r, session=False):
    s = "exp015-ne16-pusht" if session else "exp012-ne-pusht"
    i = "exp015-int4c-pusht" if session else "exp014-int4c-pusht"
    p4 = K(r, "exp015-sim-pusht", "predictor:4") if session else K(r, "exp003", "predictor:4")
    return {"fp32": K(r, "exp003"), "ne16": K(r, s), "int4c": K(r, i), "predictor4": p4, "bl": K(r, "exp015-int4c-bl-pusht"),
            "pp": K(r, "exp015-int4c-pp-pusht"), "s_pp": K(r, "exp015-sim-pusht", "predproj:4"), "s_pr": K(r, "exp015-sim-pusht", "predict:4")}


def verifier(r, res, session=False):
    S = series(r, session)
    ps = []
    for nom, b, sorte in A15.TESTS:
        ref = "fp32" if b.startswith("s_") else "ne16"
        x = S[b] - S[ref]
        y = 20 * x + 2 if sorte == "non_inferiorite" else -20 * x - 2
        s, bt = p_signe(y), p_boot_t(y)
        T = res["tests"][nom]
        assert T["p_signe"] == s and abs(T["p_bootstrap_t"] - bt) < 1e-12 and T["p"] == max(s, bt), (nom, T, s, bt)
        assert abs(T["ecart_points"]["moyenne"] - 100 * x.mean() / 5) < 1e-9, nom
        ps.append(max(s, bt))
    for (nom, *_), q in zip(A15.TESTS, holm_numpy(ps)):
        assert abs(res["tests"][nom]["p_holm"] - q) < 1e-12, nom
        assert res["tests"][nom]["verdict"] == ("soutenue" if q < 0.05 else "non soutenue"), nom
        assert res["tests"][nom]["borne_et_verdict_concordent"], (nom, res["tests"][nom])
    m = lambda x: 100 * x.mean() / 5
    D1 = res["D1_ecarts"]
    for b, ref in (("bl", "ne16"), ("pp", "ne16"), ("int4c", "ne16"), ("s_pp", "fp32"), ("s_pr", "fp32"), ("predictor4", "fp32")):
        assert abs(D1[f"{b} - {ref}"]["moyenne"] - m(S[b] - S[ref])) < 1e-9, b
    D2, D3 = res["D2_interaction"], res["D3_puce_contre_simulation"]
    assert abs(D2["puce"]["moyenne"] - m(S["int4c"] - S["ne16"] - (S["bl"] - S["ne16"]) - (S["pp"] - S["ne16"]))) < 1e-9
    assert abs(D2["simule"]["moyenne"] - m(S["s_pr"] - S["predictor4"] - S["s_pp"] + S["fp32"])) < 1e-9
    assert abs(D3["pred_proj"]["moyenne"] - m(S["pp"] - S["ne16"] - S["s_pp"] + S["fp32"])) < 1e-9
    assert abs(D3["blocs"]["moyenne"] - m(S["bl"] - S["ne16"] - S["predictor4"] + S["fp32"])) < 1e-9
    assert abs(D3["tout"]["moyenne"] - m(S["int4c"] - S["ne16"] - S["s_pr"] + S["fp32"])) < 1e-9
    D4 = res["D4_part_pred_proj"]
    if D4["rapportee"]:
        assert abs(D4["part"] - (S["pp"] - S["ne16"]).sum() / (S["int4c"] - S["ne16"]).sum()) < 1e-12 and D4["bas"] <= D4["part"] <= D4["haut"]
    return ps


def test_scenario_a(tmp):
    r = fabriquer(Path(tmp) / "a")
    res = A.pur(A15.analyser(r, verifier=False))
    verifier(r, res)
    T = res["tests"]
    assert (T["L1"]["verdict"], T["L2"]["verdict"], T["L3"]["verdict"]) == ("non soutenue", "soutenue", "non soutenue"), T
    c = res["controles"]
    assert c["ne16_identique"] and c["int4c_identique"] and c["predictor4_identique_exp003"] and not c["avec_sp"]
    assert c["source_ne16"] == "exp012-ne-pusht" and c["source_int4c"] == "exp014-int4c-pusht"
    assert c["constantes"]["bl"] == {"blocs": 30, "pred_proj": 0, "reecriture": 0} and c["constantes"]["int4c"]["pred_proj"] == 2
    assert not res["D4_part_pred_proj"]["rapportee"] and res["D5"]["octets_poids"]["ne16"] == OCTETS["fp16"]
    txt = A15.rapport(res)
    assert "| L2 (non_inferiorite) | bl contre ne16 |" in txt and "non lancé" in txt
    print("ok scénario A : aucun effet ; L2 soutenue, L1 et L3 non ; références d'EXP-003, 012 et 014 ancrées")
    return r


def test_scenario_c(tmp):
    r = fabriquer(Path(tmp) / "c", "C")
    res = A.pur(A15.analyser(r, verifier=False))
    ps = verifier(r, res)
    assert any(1 / 10_001 < p < 1 for p in ps), ps
    assert res["D4_part_pred_proj"]["rapportee"]
    T = res["tests"]
    print(f"ok scénario C : p ({', '.join(f'{p:.4f}' for p in ps)}), Holm sur 3, D1 à D4 recomptés ; verdicts "
          f"{[T[n]['verdict'] for n in T]}")


def test_scenario_s(tmp):
    r = fabriquer(Path(tmp) / "s", "C", session=True, avec_sp=True)
    res = A.pur(A15.analyser(r, verifier=False))
    verifier(r, res, session=True)
    c = res["controles"]
    assert not (c["ne16_identique"] or c["int4c_identique"] or c["predictor4_identique_exp003"]) and c["avec_sp"], c
    assert c["source_ne16"] == "exp015-ne16-pusht" and c["source_int4c"] == "exp015-int4c-pusht"
    assert "sp - ne16" in res["D1_ecarts"] and c["constantes"]["sp"] == {"blocs": 30, "pred_proj": 0, "reecriture": 2}
    assert res["D2_interaction"]["matrices_0_1_non_inertes_comprises"]
    print("ok scénario S : témoins différents, séries refaites dans la session utilisées ; bras int4c-sp lu")


def test_refus(tmp, r):
    def essai(motif, modifier, base=r):
        rr = Path(tmp) / "refus"
        shutil.rmtree(rr, ignore_errors=True)
        shutil.copytree(base, rr)
        modifier(rr)
        try:
            A15.analyser(rr, verifier=False)
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

    def json_(chemin, f):
        def m(rr):
            g = rr / chemin
            j = json.loads(g.read_text())
            f(j)
            g.write_text(json.dumps(j))
        return m

    def different(d, nom):                            # run refait différent de sa série réutilisée
        return champ(d, nom, "positions", lambda x: x["positions"][::-1].copy())

    def taille(d, g, n):
        def m(rr):
            with open(rr / d / f"fp32_s{g}.mlpackage" / "Data" / "weights" / "weight.bin", "wb") as f:
                f.truncate(n)
        return m
    BL, PP, SIM = "exp015-int4c-bl-pusht", "exp015-int4c-pp-pusht", "exp015-sim-pusht"
    cas = [("reproduction de exp003, graine 42", lambda rr: shutil.copy(rr / "exp003" / "all-4_s42.npz", rr / "exp015-repro-pusht" / "fp32_s42.npz")),
           ("reproduction de exp003, graine 46", champ("exp015-fin-pusht", "fp32_s46.npz", "success", lambda x: ~x["success"])),
           ("exp015-fin-pusht : autre commit", champ("exp015-fin-pusht", "fp32_s46.npz", "git", lambda x: np.str_("c" * 40))),
           ("exp015-ne16-pusht/fp32_s43.npz manquant", different("exp015-ne16-pusht", "fp32_s42.npz")),
           ("exp015-int4c-pusht/fp32_s43.npz manquant", different("exp015-int4c-pusht", "fp32_s42.npz")),
           ("exp015-sim-pusht/predictor-4_s43.npz manquant", different(SIM, "predictor-4_s42.npz")),
           ("predictor-4_s42.npz manquant", lambda rr: (rr / SIM / "predictor-4_s42.npz").unlink()),
           ("manquant", lambda rr: (rr / BL / "fp32_s44.npz").unlink()),
           ("manquant", lambda rr: (rr / SIM / "predict-4_s45.npz").unlink()),
           ("manquant", lambda rr: (rr / "exp012-ne-pusht" / "fp32_s45.npz").unlink()),
           ("int4c-sp-pusht/meta.json", json_("exp015-etape0b/resultats.json", lambda j: j.update(sp_et_bl_identiques_sur_le_neural_engine=False))),
           ("bras sp présent", lambda rr: shutil.copytree(rr / BL, rr / "exp015-int4c-sp-pusht")),
           ("étape 0 b au commit", json_("exp015-etape0b/resultats.json", lambda j: j.update(git="a" * 40))),
           ("étape 0 b : arbre git modifié", json_("exp015-etape0b/resultats.json", lambda j: j.update(git_sale=True))),
           ("version ou graine", champ(PP, "fp32_s43.npz", "seed", lambda x: np.int64(42))),
           ("version ou graine", champ(SIM, "predproj-4_s44.npz", "spec", lambda x: np.str_("predict:4"))),
           ("arbre git modifié", champ(BL, "fp32_s45.npz", "git_sale", lambda x: np.bool_(True))),
           ("2 couches quantifiées au lieu de 32", champ(SIM, "predict-4_s46.npz", "n_couches", lambda x: np.int64(2))),
           ("couches quantifiées au lieu de 0", champ(PP, "fp32_s46.npz", "n_couches", lambda x: np.int64(2))),
           ("moteur", champ(BL, "fp32_s42.npz", "moteur", lambda x: np.str_("coreml-gpu"))),
           ("moteur", champ(SIM, "predproj-4_s42.npz", "moteur", lambda x: np.str_("coreml-ne"))),
           ("empreintes", champ(PP, "fp32_s44.npz", "coreml_sha256", lambda x: np.str_("autre"))),
           ("modèle Core ML différent selon les graines", taille(BL, 45, 7_000_001)),
           ("hors du Neural Engine", json_(f"{PP}/fp32_s46.temps.json", lambda j: j["coreml"].update(placement={"NeuralEngine": 150, "CPU": 211}))),
           ("hors du Neural Engine", json_("exp015-ne16-pusht/fp32_s42.temps.json", lambda j: j["coreml"].update(placement={"GPU": 197}))),
           ("codage", json_(f"{BL}/fp32_s42.temps.json", lambda j: j["coreml"].update(codage="int4c-sp"))),
           ("protocole différent", json_(f"{BL}/meta.json", lambda j: j["protocole"].update(chunk=32))),
           ("protocole différent", json_(f"{SIM}/meta.json", lambda j: j["protocole"].update(n=400))),
           ("pour un run PyTorch", json_(f"{SIM}/meta.json", lambda j: j["protocole"].update(moteur={"nom": "torch"}))),
           ("moteur", json_(f"{PP}/meta.json", lambda j: j["protocole"]["moteur"].update(codage="int4c"))),
           ("moteur", json_(f"{BL}/meta.json", lambda j: j["protocole"]["moteur"].update(coremltools="9.1"))),
           ("versions de macOS", json_(f"{PP}/meta.json", lambda j: j["protocole"]["moteur"].update(macos="26.7"))),
           ("init_state différent", champ(BL, "fp32_s42.npz", "init_state", lambda x: x["init_state"] + 1)),
           ("episodes_idx différent", champ(SIM, "predict-4_s43.npz", "episodes_idx", lambda x: x["episodes_idx"][::-1].copy())),
           ("plusieurs commits", lambda rr: (champ(PP, "fp32_s45.npz", "git", lambda x: np.str_("b" * 40))(rr),
                                             json_(f"{PP}/fp32_s45.temps.json", lambda j: j.update(git="b" * 40))(rr))),
           ("plusieurs commits", champ(SIM, "predproj-4_s43.npz", "git", lambda x: np.str_("b" * 40))),
           ("fichier de temps", champ(BL, "fp32_s46.npz", "git", lambda x: np.str_("9" * 40))),
           ("commit", champ("exp012-ne-pusht", "fp32_s44.npz", "git", lambda x: np.str_("f" * 40))),
           ("commit", champ("exp014-int4c-pusht", "fp32_s43.npz", "git", lambda x: np.str_("f" * 40))),
           ("constantes codées", lambda rr: (rr / BL / "fp32_s42.mlpackage" / "constantes.json").write_text(json.dumps(CONST["int4c-sp"]))),
           ("constantes codées", lambda rr: (rr / PP / "fp32_s42.mlpackage" / "constantes.json").write_text(json.dumps(CONST["int4c"]))),
           ("constantes codées illisibles", lambda rr: (rr / PP / "fp32_s42.mlpackage" / "constantes.json").write_text(json.dumps({"erreur": "x"}))),
           ("pas plus petit que le 16 bits", lambda rr: [taille(PP, g, 30_000_000)(rr) for g in G]),
           ("même modèle", lambda rr: [champ(BL, f"fp32_s{g}.npz", "coreml_sha256", lambda x: np.str_("sha-int4c-pp" * 4))(rr) or
                                       json_(f"{BL}/fp32_s{g}.temps.json", lambda j: j["coreml"].update(sha256_poids="sha-int4c-pp" * 4))(rr)
                                       for g in G])]
    for motif, m in cas:
        essai(motif, m)
    print(f"ok contrôles bloquants ({len(cas)} refus, chacun avec son motif)")


def test_code():
    assert PROPRE, "arbre modifié : committer avant de lancer les tests"
    A15.verifier_code(TETE)
    print(f"ok code : fichiers d'analyse et de calcul identiques à {TETE[:7]}")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp, np.errstate(all="ignore"):
        r = test_scenario_a(tmp)
        test_scenario_c(tmp)
        test_scenario_s(tmp)
        test_refus(tmp, r)
        test_code()
    print("tous les tests passent")
