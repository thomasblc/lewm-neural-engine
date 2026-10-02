"""Analyse pré-enregistrée d'EXP-015 (PREREG-EXP015.md) : où se loge la perte d'int4 par canal sur le Neural Engine, PushT.

Bras : int4c-bl (blocs seuls), int4c-pp (pred_proj seule), int4c-sp (sans pred_proj, seulement si l'étape 0 b l'a
demandé) sur le Neural Engine ; predproj:4 et predict:4 en simulé (PyTorch float32). Références : fp32 et predictor:4
(EXP-003), 16 bits sur le Neural Engine (EXP-012), int4c (EXP-014), chacune ancrée par un run refait dans la session
(graine 42) ; une référence dont le run refait diffère est remplacée par sa série refaite dans la session (5 graines).
Tests L1 à L3 (Holm sur 3) ; descriptifs D1 à D5.

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
TESTS = (("L1", "pp", "perte"), ("L2", "bl", "non_inferiorite"), ("L3", "s_pp", "perte"))
FICHIERS_ANALYSE = ("analyse_exp015.py", "analyse_exp014.py", "analyse_exp012.py", "analyse_exp009.py", "analyse_exp003.py",
                    "metriques.py", "quant.py", "run_grid.py", "coreml_pred.py", "predicteur_ane.py", "etape0b_exp015.py",
                    "PREREG-EXP015.md")
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

    def ancre(run_session, run_ref, nom):
        """Run refait dans la session (graine 42) contre celui de la série réutilisée : identique octet pour octet ?"""
        A.exiger(run_session.exists(), f"{nom} : run refait manquant")
        return bool(identiques(run_session, run_ref))

    K, rep = {}, {}
    # fp32 et predictor:4 d'EXP-003 (commit d'EXP-003) ; predictor:4 ancré par sa réplique de la graine 42
    c3 = json.loads((r / REF / "meta.json").read_text())["git"]
    K["fp32"] = serie_torch(REF, "fp32", c3)
    meta("exp015-sim-pusht")
    lire("exp015-sim-pusht", 42, "predictor:4")
    rep["predictor4_identique_exp003"] = ancre(r / "exp015-sim-pusht" / nom_fichier("predictor:4", 42), r / REF / nom_fichier("predictor:4", 42), "predictor:4")
    K["predictor4"] = serie_torch(REF, "predictor:4", c3) if rep["predictor4_identique_exp003"] else serie_torch("exp015-sim-pusht", "predictor:4")
    # 16 bits sur le Neural Engine (EXP-012) et int4c (EXP-014), ancrés de même
    c12 = json.loads((r / "exp012-ne-pusht" / "meta.json").read_text())["git"]
    c14 = json.loads((r / "exp014-int4c-pusht" / "meta.json").read_text())["git"]
    for cle, d_ref, d_ses, codage, c in (("ne16", "exp012-ne-pusht", "exp015-ne16-pusht", "fp16", c12),
                                         ("int4c", "exp014-int4c-pusht", "exp015-int4c-pusht", "int4c", c14)):
        meta(d_ses, codage)
        _, sha_ses, _ = run_ne(d_ses, 42, codage)
        rep[f"{cle}_identique"] = ancre(r / d_ses / "fp32_s42.npz", r / d_ref / "fp32_s42.npz", cle)
        K[cle], info[f"sha_{cle}"], info[f"octets_{cle}"] = (serie_ne(d_ref, codage, c) if rep[f"{cle}_identique"]
                                                             else serie_ne(d_ses, codage))
        info[f"source_{cle}"] = d_ref if rep[f"{cle}_identique"] else d_ses
        A.exiger(sha_ses == info[f"sha_{cle}"], f"{d_ses} : modèle Core ML différent de celui de la série {info[f'source_{cle}']}")
    # étape 0 b : produite au commit d'EXP-015, décide du bras sp
    e0b = json.loads((r / "exp015-etape0b" / "resultats.json").read_text())
    A.exiger(not e0b["git_sale"], "étape 0 b : arbre git modifié")
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
    info["macos"] = sorted(info["macos"])
    return K, info, sha


def analyser(racine, verifier=True):
    K, info, sha = charger(racine, verifier)
    inv, boot = A.Inversion(n=500), A.Bootstrap(n=500)
    ecart = lambda a, b: K[a] - K[b]                  # somme sur les graines, par épisode
    reference = {"bl": "ne16", "pp": "ne16", "sp": "ne16", "int4c": "ne16", "s_pp": "fp32", "s_pr": "fp32", "predictor4": "fp32"}

    def test(x, sorte):
        y = 20 * x + MARGE if sorte == "non_inferiorite" else -20 * x - MARGE
        ps, pb = inv.p(y, "plus"), p_bootstrap_t(y)
        return {"sorte": sorte, "ecart_points": {k: 100 * v for k, v in boot.ic(x / 5).items()}, "p_signe": ps,
                "p_bootstrap_t": pb, "erreur_mc_p_signe": float(np.sqrt(ps * (1 - ps) / A.B)),
                "erreur_mc_p_bootstrap_t": float(np.sqrt(pb * (1 - pb) / 10_000)), "p": max(ps, pb)}
    tests = {nom: {"bras": b, "reference": reference[b], **test(ecart(b, reference[b]), s)} for nom, b, s in TESTS}
    ps = [tests[n]["p"] for n, *_ in TESTS]
    for (nom, b, s), q, niv in zip(TESTS, A.holm(ps), niveaux_holm(ps)):
        t = tests[nom]
        t["p_holm"], t["niveau_holm"] = float(q), float(niv)
        t["verdict"] = "soutenue" if q < A.ALPHA else "non soutenue"
        x = ecart(b, reference[b]) / 5                 # borne unilatérale bootstrap-t au niveau de Holm, en points
        if s == "non_inferiorite":
            t["borne_points"] = 100 * borne_bootstrap_t(x, niv)
            t["borne_et_verdict_concordent"] = (t["borne_points"] > -MARGE) == (t["verdict"] == "soutenue")
        else:                                          # borne supérieure de l'écart : - borne inférieure de -x
            t["borne_points"] = -100 * borne_bootstrap_t(-x, niv)
            t["borne_et_verdict_concordent"] = (t["borne_points"] < -MARGE) == (t["verdict"] == "soutenue")
    ic = lambda x: {k: 100 * v for k, v in boot.ic(x / 5).items()}
    d1 = {f"{b} - {reference[b]}": ic(ecart(b, reference[b])) for b in reference if b in K}
    d2 = {"puce": ic(ecart("int4c", "ne16") - ecart("bl", "ne16") - ecart("pp", "ne16")),
          "simule": ic(ecart("s_pr", "fp32") - ecart("predictor4", "fp32") - ecart("s_pp", "fp32")),
          "matrices_0_1_non_inertes_comprises": info["avec_sp"]}
    d3 = {"pred_proj": ic(ecart("pp", "ne16") - ecart("s_pp", "fp32")),
          "blocs": ic(ecart("bl", "ne16") - ecart("predictor4", "fp32")),
          "tout": ic(ecart("int4c", "ne16") - ecart("s_pr", "fp32"))}
    perte = ecart("int4c", "ne16").astype(np.float64)
    d4 = {"rapportee": d1["int4c - ne16"]["haut"] < 0}
    if d4["rapportee"]:                                # rapport des moyennes, épisodes rééchantillonnés ensemble
        num = ecart("pp", "ne16").astype(np.float64)
        ratios = num[boot.idx].mean(axis=1) / perte[boot.idx].mean(axis=1)
        d4.update({"part": float(num.mean() / perte.mean()), "bas": float(np.percentile(ratios, 2.5)),
                   "haut": float(np.percentile(ratios, 97.5))})
    d5 = {"reussite": {b: float(K[b].mean() / 5) for b in K},
          "octets_poids": {b: info.get(f"octets_{b}") for b in ("ne16", "int4c", "bl", "pp", "sp") if f"octets_{b}" in info}}
    controles = {"commit": sha, **{k: info[k] for k in ("predictor4_identique_exp003", "ne16_identique", "int4c_identique",
                                                        "source_ne16", "source_int4c", "avec_sp", "constantes", "macos")}}
    return {"controles": controles, "tests": tests, "D1_ecarts": d1, "D2_interaction": d2, "D3_puce_contre_simulation": d3,
            "D4_part_pred_proj": d4, "D5": d5, "etape0b_resume": info["etape0b_resume"]}


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
    L += ["", "D1, écarts :", ""] + [f"- {k} : {f(v)}" for k, v in r["D1_ecarts"].items()]
    L += ["", "D2, interaction :", "", f"- puce : {f(r['D2_interaction']['puce'])}", f"- simulé : {f(r['D2_interaction']['simule'])}"]
    L += ["", "D3, puce contre simulation (différence de différences) :", ""] + [f"- {k} : {f(v)}" for k, v in r["D3_puce_contre_simulation"].items()]
    d4 = r["D4_part_pred_proj"]
    L += ["", "D4, part de la perte d'int4c portée par pred_proj : " +
          (f"{d4['part']:.2f} [{d4['bas']:.2f} ; {d4['haut']:.2f}]" if d4["rapportee"] else "non rapportée (perte d'int4c non établie)")]
    L += ["", "D5, réussite : " + ", ".join(f"{k} {100 * v:.1f} %" for k, v in r["D5"]["reussite"].items()),
          "", "Fichier des poids (octets) : " + ", ".join(f"{k} {v}" for k, v in r["D5"]["octets_poids"].items())]
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
