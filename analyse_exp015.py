"""Analyse pré-enregistrée d'EXP-015 (PREREG-EXP015.md) : où se loge la perte d'int4 par canal sur le Neural Engine, PushT.

Bras : int4c-bl (blocs seuls), int4c-pp (pred_proj seule), int4c-sp (sans pred_proj, seulement si l'étape 0 b l'a
demandé) sur le Neural Engine ; predproj:4 et predict:4 en simulé (PyTorch float32). Références : fp32 et predictor:4
(EXP-003), 16 bits sur le Neural Engine (EXP-012), int4c (EXP-014), chacune ancrée par un run refait dans la session
(graine 42) ; une référence dont le run refait diffère est remplacée par sa série refaite dans la session (5 graines).
Tests L1 à L4, plus L5 si le bras int4c-sp est lancé (Holm sur 4 ou 5, amendement 1) ; descriptifs D1 à D5 ; règle R1
de l'étape 0 b reprise.

Usage :
    python analyse_exp015.py --racine ../resultats --sortie ../resultats/analyse-exp015
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

import analyse_exp003 as A
import analyse_exp009 as A9
from analyse_exp012 import borne_bootstrap_t, niveaux_holm, p_bootstrap_t
from analyse_exp014 import categories, constantes_codees, identiques, lire_temps

GRAINES = (42, 43, 44, 45, 46)
MARGE = 2                                           # points
MOTEUR = "coreml-ne"
COREMLTOOLS = "9.0"
REF = "exp003"
BRAS_NE = {"bl": "int4c-bl", "pp": "int4c-pp", "sp": "int4c-sp"}
CONSTANTES = {"int4c-bl": (30, 0, 0), "int4c-pp": (0, 2, 0), "int4c-sp": (30, 0, 2), "int4c": (30, 2, 2)}   # blocs, pred_proj, réécriture
SIMULES = {"predproj:4": 2, "predict:4": 32, "predictor:4": 30}                                          # couches quantifiées
# (nom, bras, référence, sorte) ; sorte : « perte » (perd plus de 2 points), « non_inferiorite » (ne perd pas plus de 2
# points), « pas_mieux » (ne fait pas mieux de plus de 2 points)
TESTS = (("L1", "pp", "ne16", "perte"), ("L2", "bl", "ne16", "non_inferiorite"), ("L3", "s_pp", "fp32", "perte"),
         ("L4", "pp", "int4c", "pas_mieux"))
L5 = ("L5", "sp", "ne16", "non_inferiorite")         # seulement si le bras int4c-sp est lancé
FICHIERS_ANALYSE = ("analyse_exp015.py", "analyse_exp014.py", "analyse_exp012.py", "analyse_exp009.py", "analyse_exp003.py",
                    "metriques.py", "quant.py", "run_grid.py", "coreml_pred.py", "predicteur_ane.py", "etape0b_exp015.py",
                    "etape0_exp015.py", "PREREG-EXP015.md")
ICI = Path(__file__).resolve().parent


def git(*args):
    return subprocess.run(["git", "-C", str(ICI), *args], capture_output=True, text=True)


def verifier_code(sha):
    A.exiger(git("cat-file", "-e", f"{sha}^{{commit}}").returncode == 0, f"commit {sha[:7]} inconnu du dépôt")
    A.exiger(git("diff", "--quiet", sha, "--", *FICHIERS_ANALYSE).returncode == 0,
             f"code d'analyse ou de calcul différent de celui du commit {sha[:7]}")


def nom_fichier(spec, g):
    return f"{spec.replace(':', '-')}_s{g}.npz"


def charger(racine, verifier=True):
    r = Path(racine)
    ref = dict(np.load(r / REF / "fp32_s42.npz"))
    proto_ref = json.loads((r / REF / "meta.json").read_text())["protocole"]
    for d, g in (("exp015-repro-pusht", 42), ("exp015-fin-pusht", 46)):
        A9.identiques(r / d / f"fp32_s{g}.npz", r / REF / f"fp32_s{g}.npz", f"reproduction de {REF}, graine {g}")
    shas, info = set(), {}

    def lire(d, g, spec="fp32", moteur=None, du_commit=None):
        """Un fichier de résultats, contrôlé ; du_commit : commit exigé (fichiers réutilisés d'une autre expérience)."""
        f = r / d / nom_fichier(spec, g)
        A.exiger(f.exists(), f"{d}/{f.name} manquant")
        x = dict(np.load(f))
        A.exiger((str(x["spec"]), int(x["seed"])) == (spec, g), f"{d}/{f.name} : version ou graine")
        A.exiger(not bool(x["git_sale"]), f"{d}/{f.name} : arbre git modifié")
        attendu = 0 if spec == "fp32" else SIMULES[spec]
        A.exiger(int(x["n_couches"]) == attendu, f"{d}/{f.name} : {int(x['n_couches'])} couches quantifiées au lieu de {attendu}")
        A.exiger(str(x.get("moteur", "torch")) == (moteur or "torch"), f"{d}/{f.name} : moteur {x.get('moteur')}")
        for k in ("episodes_idx", "start_steps", "init_state", "goal_state"):
            A.exiger(np.array_equal(x[k], ref[k]), f"{d}/{f.name} : {k} différent de la référence")
        A.controler_course(x, f"{d}/{f.name}")
        if du_commit is not None:
            A.exiger(str(x["git"]) == du_commit, f"{d}/{f.name} : commit {str(x['git'])[:7]} au lieu de {du_commit[:7]}")
        else:
            shas.add(str(x["git"]))
        return x

    def meta(d, codage=None, du_commit=None):
        A.exiger((r / d / "meta.json").exists(), f"{d}/meta.json manquant")
        m = json.loads((r / d / "meta.json").read_text())
        p = m["protocole"]
        A.exiger({k: v for k, v in p.items() if k != "moteur"} == proto_ref, f"{d} : protocole différent de {REF}")
        if codage is None:
            A.exiger("moteur" not in p, f"{d} : moteur {p.get('moteur')} pour un run PyTorch")
        else:
            mo = p.get("moteur", {})
            A.exiger(mo.get("nom") == MOTEUR and mo.get("precision") == "float16" and mo.get("codage", "fp16") == codage
                     and mo.get("coremltools") == COREMLTOOLS, f"{d} : moteur {mo}")
            info.setdefault("macos", set()).add(mo.get("macos"))
        if du_commit is not None:
            A.exiger(m["git"] == du_commit, f"{d} : meta d'un autre commit")
        return m

    def run_ne(d, g, codage, du_commit=None):
        """Un run sur le Neural Engine : fichier contrôlé, tout sur le Neural Engine, codage, empreinte ; taille du
        fichier des poids lue sur le paquet enregistré (EXP-012 ne la notait pas)."""
        x = lire(d, g, moteur=MOTEUR, du_commit=du_commit)
        tf = lire_temps(r / d / f"fp32_s{g}.temps.json", str(x["git"]))
        A.exiger(set(tf["coreml"].get("placement", {})) == {"NeuralEngine"}, f"{d} graine {g} : opérations hors du Neural Engine")
        A.exiger(tf["coreml"].get("codage", "fp16") == codage, f"{d} graine {g} : codage {tf['coreml'].get('codage')}")
        A.exiger(tf["coreml"].get("sha256_poids") == str(x["coreml_sha256"]), f"{d} graine {g} : empreintes")
        octets = sum(f.stat().st_size for f in (r / d / f"fp32_s{g}.mlpackage").rglob("weight.bin"))
        A.exiger(octets > 0, f"{d} graine {g} : fichier des poids absent")
        info.setdefault(f"temps_{d}", {})[g] = tf["par_plan_s"]
        return x, str(x["coreml_sha256"]), octets

    def serie_ne(d, codage, du_commit=None):
        """Somme des succès sur les 5 graines d'un bras sur le Neural Engine ; un seul modèle."""
        meta(d, codage, du_commit)
        S, emp = 0, set()
        for g in GRAINES:
            x, sha_m, octets = run_ne(d, g, codage, du_commit)
            S = S + x["success"].astype(np.int64)
            emp.add((sha_m, octets))
        A.exiger(len(emp) == 1, f"{d} : modèle Core ML différent selon les graines")
        sha_m, octets = emp.pop()
        return S, sha_m, octets

    def serie_torch(d, spec, du_commit=None):
        meta(d, None, du_commit)
        return sum(lire(d, g, spec, du_commit=du_commit)["success"].astype(np.int64) for g in GRAINES)

    def git_meta(d):
        A.exiger((r / d / "meta.json").exists(), f"{d}/meta.json manquant")
        return json.loads((r / d / "meta.json").read_text())["git"]

    def reproductions(prefixe, c):
        """Une série réutilisée garde ses propres contrôles : reproductions de la graine 42 et de la graine 46 identiques
        à EXP-003, du commit de la série (amendement 1, point 7)."""
        for d, g in ((f"{prefixe}-repro-pusht", 42), (f"{prefixe}-fin-pusht", 46)):
            f = r / d / f"fp32_s{g}.npz"
            A.exiger(f.exists(), f"{d}/{f.name} manquant")
            A9.identiques(f, r / REF / f"fp32_s{g}.npz", f"reproduction de {prefixe}, graine {g}")
            x = np.load(f)
            A.exiger(str(x["git"]) == c and not bool(x["git_sale"]), f"{d} : reproduction d'un autre commit que la série")

    def ancre(run_session, run_ref, nom):
        """Run refait dans la session (graine 42) contre celui de la série réutilisée : identique octet pour octet ?"""
        A.exiger(run_session.exists(), f"{nom} : run refait manquant")
        return bool(identiques(run_session, run_ref))

    K, rep = {}, {}
    # fp32 et predictor:4 d'EXP-003 (commit d'EXP-003) ; predictor:4 ancré par sa réplique de la graine 42
    c3 = git_meta(REF)
    K["fp32"] = serie_torch(REF, "fp32", c3)
    meta("exp015-sim-pusht")
    lire("exp015-sim-pusht", 42, "predictor:4")
    A.exiger((r / REF / nom_fichier("predictor:4", 42)).exists(), f"{REF}/{nom_fichier('predictor:4', 42)} manquant")
    rep["predictor4_identique_exp003"] = ancre(r / "exp015-sim-pusht" / nom_fichier("predictor:4", 42), r / REF / nom_fichier("predictor:4", 42), "predictor:4")
    K["predictor4"] = serie_torch(REF, "predictor:4", c3) if rep["predictor4_identique_exp003"] else serie_torch("exp015-sim-pusht", "predictor:4")
    # 16 bits sur le Neural Engine (EXP-012) et int4c (EXP-014), ancrés de même
    c12, c14 = git_meta("exp012-ne-pusht"), git_meta("exp014-int4c-pusht")
    for cle, d_ref, d_ses, codage, c in (("ne16", "exp012-ne-pusht", "exp015-ne16-pusht", "fp16", c12),
                                         ("int4c", "exp014-int4c-pusht", "exp015-int4c-pusht", "int4c", c14)):
        meta(d_ses, codage)
        _, sha_ses, _ = run_ne(d_ses, 42, codage)
        A.exiger((r / d_ref / "fp32_s42.npz").exists(), f"{d_ref}/fp32_s42.npz manquant")
        rep[f"{cle}_identique"] = ancre(r / d_ses / "fp32_s42.npz", r / d_ref / "fp32_s42.npz", cle)
        if rep[f"{cle}_identique"]:
            reproductions(d_ref.split("-")[0], c)
        K[cle], info[f"sha_{cle}"], info[f"octets_{cle}"] = (serie_ne(d_ref, codage, c) if rep[f"{cle}_identique"]
                                                             else serie_ne(d_ses, codage))
        info[f"source_{cle}"] = d_ref if rep[f"{cle}_identique"] else d_ses
        A.exiger(sha_ses == info[f"sha_{cle}"], f"{d_ses} : modèle Core ML différent de celui de la série {info[f'source_{cle}']}")
    # étape 0 b : produite au commit d'EXP-015, décide du bras sp
    A.exiger((r / "exp015-etape0b" / "resultats.json").exists(), "étape 0 b : sortie manquante")
    e0b = json.loads((r / "exp015-etape0b" / "resultats.json").read_text())
    A.exiger(not e0b["git_sale"], "étape 0 b : arbre git modifié")
    A.exiger(e0b.get("valide") is True, "étape 0 b invalide (plancher de validité) : EXP-015 ne devait pas être lancée")
    avec_sp = not e0b["sp_et_bl_identiques_sur_le_neural_engine"]
    for b in (("bl", "pp", "sp") if avec_sp else ("bl", "pp")):
        d = f"exp015-{BRAS_NE[b]}-pusht"
        K[b], info[f"sha_{b}"], info[f"octets_{b}"] = serie_ne(d, BRAS_NE[b])
    A.exiger(avec_sp or not (r / "exp015-int4c-sp-pusht").exists(), "bras sp présent alors que l'étape 0 b ne le demandait pas")
    K["s_pp"] = serie_torch("exp015-sim-pusht", "predproj:4")
    K["s_pr"] = serie_torch("exp015-sim-pusht", "predict:4")
    # le codage a été appliqué, à la bonne portée ; modèles tous différents
    bras = [b for b in ("bl", "pp", "sp", "int4c") if b in K]
    A.exiger(len({info[f"sha_{b}"] for b in bras + ["ne16"]}) == len(bras) + 1, "deux bras ont le même modèle Core ML")
    info["constantes"] = {}
    for b in bras:
        A.exiger(info[f"octets_{b}"] < info["octets_ne16"], f"{b} : fichier des poids pas plus petit que le 16 bits")
        codage = "int4c" if b == "int4c" else BRAS_NE[b]
        d = info["source_int4c"] if b == "int4c" else f"exp015-{codage}-pusht"
        lst = constantes_codees(r / d / "fp32_s42.mlpackage")
        A.exiger(isinstance(lst, list), f"{b} : constantes codées illisibles {lst}")
        k = categories(lst)
        A.exiger((k["blocs"], k["pred_proj"], k["reecriture"]) == CONSTANTES[codage] and k["autres"] == 0,
                 f"{b} : constantes codées {k} au lieu de {CONSTANTES[codage]}")
        info["constantes"][b] = {c: k[c] for c in ("blocs", "pred_proj", "reecriture")}
    A.exiger(len(info["macos"]) == 1, f"versions de macOS différentes {info['macos']}")
    # l'étape 0 b a mesuré les modèles des runs, avec les mêmes versions
    A.exiger(e0b.get("coremltools") == COREMLTOOLS and e0b.get("macos") in info["macos"],
             f"étape 0 b : versions {e0b.get('coremltools')}, macOS {e0b.get('macos')}")
    for b, codage in (("ne16", "fp16"), ("int4c", "int4c"), ("bl", "int4c-bl"), ("pp", "int4c-pp"), ("sp", "int4c-sp")):
        if b in K:
            A.exiger(e0b["codages"][codage]["sha256_poids"] == info[f"sha_{b}"], f"étape 0 b : modèle {codage} différent de celui des runs")
    A.exiger(len(shas) == 1, f"EXP-015 : fichiers de plusieurs commits {sorted(shas)}")
    sha = shas.pop()
    A.exiger(e0b["git"] == sha, f"étape 0 b au commit {e0b['git'][:7]}, runs au commit {sha[:7]}")
    for d, g in (("exp015-repro-pusht", 42), ("exp015-fin-pusht", 46)):
        x = np.load(r / d / f"fp32_s{g}.npz")
        A.exiger(str(x["git"]) == sha and not bool(x["git_sale"]), f"{d} : autre commit ou arbre modifié")
    if verifier:
        verifier_code(sha)
    info.update(rep)
    info["avec_sp"] = avec_sp
    info["etape0b_resume"] = e0b.get("resume")
    info["R1"] = e0b.get("R1")
    info["temps_ms"] = {b: 1000 * float(np.median(list(info[f"temps_{d}"].values()))) for b, d in
                        (("ne16", info["source_ne16"]), ("int4c", info["source_int4c"]), *((b, f"exp015-{BRAS_NE[b]}-pusht") for b in BRAS_NE if b in K))}
    info["macos"] = sorted(info["macos"])
    return K, info, sha


def analyser(racine, verifier=True):
    K, info, sha = charger(racine, verifier)
    inv, boot = A.Inversion(n=500), A.Bootstrap(n=500)
    ecart = lambda a, b: K[a] - K[b]                  # somme sur les graines, par épisode
    famille = TESTS + ((L5,) if info["avec_sp"] else ())

    def test(x, sorte):
        y = {"non_inferiorite": 20 * x + MARGE, "perte": -20 * x - MARGE, "pas_mieux": -20 * x + MARGE}[sorte]
        ps, pb = inv.p(y, "plus"), p_bootstrap_t(y)
        return {"sorte": sorte, "ecart_points": {k: 100 * v for k, v in boot.ic(x / 5).items()}, "p_signe": ps,
                "p_bootstrap_t": pb, "erreur_mc_p_signe": float(np.sqrt(ps * (1 - ps) / A.B)),
                "erreur_mc_p_bootstrap_t": float(np.sqrt(pb * (1 - pb) / 10_000)), "p": max(ps, pb)}
    tests = {nom: {"bras": b, "reference": ref, **test(ecart(b, ref), s)} for nom, b, ref, s in famille}
    ps = [tests[n]["p"] for n, *_ in famille]
    for (nom, b, ref, s), q, niv in zip(famille, A.holm(ps), niveaux_holm(ps)):
        t = tests[nom]
        t["p_holm"], t["niveau_holm"] = float(q), float(niv)
        t["verdict"] = "soutenue" if q < A.ALPHA else "non soutenue"
        x = ecart(b, ref) / 5                          # borne unilatérale bootstrap-t au niveau de Holm, en points
        if s == "non_inferiorite":                     # borne inférieure de l'écart, au-dessus de -2 ?
            t["borne_points"] = 100 * borne_bootstrap_t(x, niv)
            t["borne_et_verdict_concordent"] = (t["borne_points"] > -MARGE) == (t["verdict"] == "soutenue")
        else:                                          # borne supérieure de l'écart : - borne inférieure de -x
            t["borne_points"] = -100 * borne_bootstrap_t(-x, niv)
            seuil = -MARGE if s == "perte" else MARGE
            t["borne_et_verdict_concordent"] = (t["borne_points"] < seuil) == (t["verdict"] == "soutenue")
    ic = lambda x: {k: 100 * v for k, v in boot.ic(x / 5).items()}
    paires = [("bl", "ne16"), ("pp", "ne16"), ("sp", "ne16"), ("int4c", "ne16"), ("bl", "fp32"), ("pp", "fp32"), ("sp", "fp32"),
              ("int4c", "fp32"), ("ne16", "fp32"), ("s_pp", "fp32"), ("s_pr", "fp32"), ("predictor4", "fp32")]
    d1 = {f"{b} - {ref}": ic(ecart(b, ref)) for b, ref in paires if b in K}
    perte_etablie = {b: d1[f"{b} - ne16"]["haut"] < 0 for b in ("bl", "pp", "sp") if b in K}
    d2 = {"puce": ic(ecart("int4c", "ne16") - ecart("bl", "ne16") - ecart("pp", "ne16")),
          "simule": ic(ecart("s_pr", "fp32") - ecart("predictor4", "fp32") - ecart("s_pp", "fp32")),
          "matrices_0_1_non_inertes_comprises": info["avec_sp"]}
    if info["avec_sp"]:                                # plan complet ne16, SP, PP, int4c (amendement 1, point 8)
        d2["puce_avec_sp"] = ic(ecart("int4c", "ne16") - ecart("sp", "ne16") - ecart("pp", "ne16"))
    d3 = {"pred_proj": ic(ecart("pp", "ne16") - ecart("s_pp", "fp32")),
          "blocs": ic(ecart("bl", "ne16") - ecart("predictor4", "fp32")),
          "tout": ic(ecart("int4c", "ne16") - ecart("s_pr", "fp32"))}
    d3_exclut_0 = {k: v["haut"] < 0 or v["bas"] > 0 for k, v in d3.items()}
    perte = ecart("int4c", "ne16").astype(np.float64)
    d4 = {"rapportee": d1["int4c - ne16"]["haut"] < 0}
    if d4["rapportee"]:                                # rapport des moyennes, épisodes rééchantillonnés ensemble
        num = ecart("pp", "ne16").astype(np.float64)
        ratios = num[boot.idx].mean(axis=1) / perte[boot.idx].mean(axis=1)
        d4.update({"part": float(num.mean() / perte.mean()), "bas": float(np.percentile(ratios, 2.5)),
                   "haut": float(np.percentile(ratios, 97.5))})
    d5 = {"reussite": {b: float(K[b].mean() / 5) for b in K},
          "octets_poids": {b: info.get(f"octets_{b}") for b in ("ne16", "int4c", "bl", "pp", "sp") if f"octets_{b}" in info},
          "temps_par_plan_ms": info["temps_ms"]}
    soutenue = lambda n: tests[n]["verdict"] == "soutenue"
    lecture = {"perte_vient_de_pred_proj": soutenue("L1") and soutenue("L2") and soutenue("L4"),
               "deploiement_sans_pred_proj_appuye_sur": ("L5" if info["avec_sp"] else "L2")
               if (soutenue("L5") if info["avec_sp"] else soutenue("L2")) else None,
               "perte_etablie": perte_etablie, "D3_exclut_0": d3_exclut_0}
    controles = {"commit": sha, **{k: info[k] for k in ("predictor4_identique_exp003", "ne16_identique", "int4c_identique",
                                                        "source_ne16", "source_int4c", "avec_sp", "constantes", "macos", "R1")}}
    return {"controles": controles, "tests": tests, "lecture": lecture, "D1_ecarts": d1, "D2_interaction": d2,
            "D3_puce_contre_simulation": d3, "D4_part_pred_proj": d4, "D5": d5, "etape0b_resume": info["etape0b_resume"]}


def rapport(r):
    L = ["# EXP-015 : rapport de l'analyse pré-enregistrée", "", f"Commit des runs : `{r['controles']['commit'][:7]}`.", "",
         "Contrôles : reproductions de la pleine précision identiques (bloquant) ; références réutilisées : "
         f"ne16 {r['controles']['source_ne16']}, int4c {r['controles']['source_int4c']}, predictor:4 "
         f"{'EXP-003' if r['controles']['predictor4_identique_exp003'] else 'refait dans la session'} ; bras int4c-sp "
         f"{'lancé' if r['controles']['avec_sp'] else 'non lancé (matrices 0/1 inertes, étape 0 b)'}.", "",
         "| Test | Bras contre référence | Écart (points) [IC 95 %] | p Holm | Borne au niveau de Holm | Verdict |",
         "| --- | --- | --- | --- | --- | --- |"]
    f = lambda e: f"{e['moyenne']:+.1f} [{e['bas']:+.1f} ; {e['haut']:+.1f}]"
    for n, t in r["tests"].items():
        L.append(f"| {n} ({t['sorte']}) | {t['bras']} contre {t['reference']} | {f(t['ecart_points'])} | {t['p_holm']:.4f} | "
                 f"{t['borne_points']:+.2f} | {t['verdict']} |")
    for n, x in r["tests"].items():
        if not x["borne_et_verdict_concordent"]:
            L.append(f"- Attention, {n} : la borne au niveau de Holm et le verdict ne concordent pas.")
    lec = r["lecture"]
    L += ["", f"Lecture pré-enregistrée : la perte d'int4c vient de pred_proj (L1, L2 et L4) : "
          f"{'soutenue' if lec['perte_vient_de_pred_proj'] else 'non établie'} ; coder tout sauf pred_proj : "
          f"{'appuyé sur ' + lec['deploiement_sans_pred_proj_appuye_sur'] if lec['deploiement_sans_pred_proj_appuye_sur'] else 'non établi'} ; "
          "perte établie (intervalle sous 0) : " + ", ".join(f"{b} {'oui' if v else 'non'}" for b, v in lec["perte_etablie"].items())
          + " ; D3 exclut 0 : " + ", ".join(f"{k} {'oui' if v else 'non'}" for k, v in lec["D3_exclut_0"].items()) + ".",
          "", "Étape 0 b, règle R1 : " + ", ".join(f"{k} {v}" for k, v in (r["controles"]["R1"] or {}).items()) + "."]
    L += ["", "D1, écarts :", ""] + [f"- {k} : {f(v)}" for k, v in r["D1_ecarts"].items()]
    L += ["", "D2, interaction :", "", f"- puce : {f(r['D2_interaction']['puce'])}", f"- simulé : {f(r['D2_interaction']['simule'])}"]
    if "puce_avec_sp" in r["D2_interaction"]:
        L.append(f"- puce, plan complet avec int4c-sp : {f(r['D2_interaction']['puce_avec_sp'])}")
    L += ["", "D3, puce contre simulation (différence de différences) :", ""] + [f"- {k} : {f(v)}" for k, v in r["D3_puce_contre_simulation"].items()]
    d4 = r["D4_part_pred_proj"]
    L += ["", "D4, part de la perte d'int4c portée par pred_proj : " +
          (f"{d4['part']:.2f} [{d4['bas']:.2f} ; {d4['haut']:.2f}]" if d4["rapportee"] else "non rapportée (perte d'int4c non établie)")]
    L += ["", "D5, réussite : " + ", ".join(f"{k} {100 * v:.1f} %" for k, v in r["D5"]["reussite"].items()),
          "", "Fichier des poids (octets) : " + ", ".join(f"{k} {v}" for k, v in r["D5"]["octets_poids"].items()),
          "", "Temps par plan médian (ms, descriptif) : " + ", ".join(f"{k} {v:.0f}" for k, v in r["D5"]["temps_par_plan_ms"].items())]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--racine", required=True)
    ap.add_argument("--sortie", required=True)
    a = ap.parse_args()
    r = A.pur(analyser(a.racine))
    out = Path(a.sortie)
    out.mkdir(parents=True, exist_ok=True)
    (out / "resultats.json").write_text(json.dumps(r, indent=1, ensure_ascii=False))
    (out / "rapport.md").write_text(rapport(r))
    print(rapport(r))


if __name__ == "__main__":
    main()
