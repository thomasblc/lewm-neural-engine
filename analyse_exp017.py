"""Analyse pré-enregistrée d'EXP-017 (PREREG-EXP017.md) : les codes de GPTQ écrits dans Core ML, sur le Neural Engine,
PushT. Bras G (puce) et GS (simulé) ; références réutilisées fp32 (EXP-003), ne16 (EXP-012), int4c (EXP-014), int4c-sp
et predict:4 RTN (EXP-015), chacune contrôlée ; ancres ne16 et int4c refaites dans la session. Tests G1 et G2 (Holm sur
2) ; descriptifs.

Usage :
    python analyse_exp017.py --racine ../resultats --sortie ../resultats/analyse-exp017
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
MARGE = 2
MOTEUR = "coreml-ne"
COREMLTOOLS = "9.0"
REF = "exp003"
GPTQ = "predict:4:gptq"
COUCHES = {"fp32": 0, "predict:4": 32, GPTQ: 32}
TESTS = (("G1", "g", "ne16", "non_inferiorite"), ("G2", "g", "int4c", "superiorite"))
# amendement 1 : fichiers de calcul identiques à ceux du commit des runs ; analyse au commit de l'analyse, qui en descend
FICHIERS_CALCUL = ("metriques.py", "quant.py", "gptq.py", "run_grid.py", "coreml_pred.py", "predicteur_ane.py", "etape0_exp017.py",
                   "etape0_exp015.py", "etape0b_exp015.py")
FICHIERS_ANALYSE = ("analyse_exp017.py", "analyse_exp014.py", "analyse_exp012.py", "analyse_exp009.py", "analyse_exp003.py",
                    "PREREG-EXP017.md")
ICI = Path(__file__).resolve().parent


def git(*args):
    return subprocess.run(["git", "-C", str(ICI), *args], capture_output=True, text=True)


def verifier_code(sha):
    """Amendement 1 : calcul identique au commit des runs ; analyse propre et commitée, à un commit qui descend des runs."""
    A.exiger(git("cat-file", "-e", f"{sha}^{{commit}}").returncode == 0, f"commit {sha[:7]} inconnu du dépôt")
    A.exiger(git("merge-base", "--is-ancestor", sha, "HEAD").returncode == 0, f"le commit de l'analyse ne descend pas du commit des runs {sha[:7]}")
    A.exiger(git("diff", "--quiet", sha, "HEAD", "--", *FICHIERS_CALCUL).returncode == 0, f"code de calcul différent du commit des runs {sha[:7]}")
    A.exiger(git("diff", "--quiet", "HEAD", "--", *FICHIERS_CALCUL, *FICHIERS_ANALYSE).returncode == 0, "fichiers modifiés depuis le commit de l'analyse")


def nom(spec, g):
    return f"{spec.replace(':', '-')}_s{g}.npz"


def charger(racine, verifier=True):
    r = Path(racine)
    ref = dict(np.load(r / REF / "fp32_s42.npz"))
    proto_ref = json.loads((r / REF / "meta.json").read_text())["protocole"]
    calib9 = json.loads((r / "exp009-pusht" / "meta.json").read_text())["protocole"]["gptq_calib"]
    for d, g in (("exp017-repro-pusht", 42), ("exp017-fin-pusht", 46)):
        A.exiger((r / d / f"fp32_s{g}.npz").exists(), f"{d}/fp32_s{g}.npz manquant")
        A9.identiques(r / d / f"fp32_s{g}.npz", r / REF / f"fp32_s{g}.npz", f"reproduction de {REF}, graine {g}")
    shas, info = set(), {}

    def git_meta(d):
        A.exiger((r / d / "meta.json").exists(), f"{d}/meta.json manquant")
        return json.loads((r / d / "meta.json").read_text())["git"]

    def lire(d, g, spec="fp32", moteur=None, du_commit=None):
        f = r / d / nom(spec, g)
        A.exiger(f.exists(), f"{d}/{f.name} manquant")
        x = dict(np.load(f))
        A.exiger((str(x["spec"]), int(x["seed"])) == (spec, g), f"{d}/{f.name} : version ou graine")
        A.exiger(not bool(x["git_sale"]), f"{d}/{f.name} : arbre git modifié")
        A.exiger(int(x["n_couches"]) == COUCHES[spec], f"{d}/{f.name} : {int(x['n_couches'])} couches quantifiées au lieu de {COUCHES[spec]}")
        A.exiger(str(x.get("moteur", "torch")) == (moteur or "torch"), f"{d}/{f.name} : moteur {x.get('moteur')}")
        for k in ("episodes_idx", "start_steps", "init_state", "goal_state"):
            A.exiger(np.array_equal(x[k], ref[k]), f"{d}/{f.name} : {k} différent de la référence")
        A.controler_course(x, f"{d}/{f.name}")
        if du_commit is not None:
            A.exiger(str(x["git"]) == du_commit, f"{d}/{f.name} : commit {str(x['git'])[:7]} au lieu de {du_commit[:7]}")
        else:
            shas.add(str(x["git"]))
        return x

    def meta(d, codage=None, gptq=False, du_commit=None):
        A.exiger((r / d / "meta.json").exists(), f"{d}/meta.json manquant")
        m = json.loads((r / d / "meta.json").read_text())
        p = m["protocole"]
        A.exiger({k: v for k, v in p.items() if k not in ("moteur", "gptq_calib")} == proto_ref, f"{d} : protocole différent de {REF}")
        A.exiger((p.get("gptq_calib") == calib9) if gptq else ("gptq_calib" not in p), f"{d} : calibration GPTQ {p.get('gptq_calib')}")
        if codage is None:
            A.exiger("moteur" not in p, f"{d} : moteur {p.get('moteur')} pour un run PyTorch")
        else:
            mo = p.get("moteur", {})
            A.exiger(mo.get("nom") == MOTEUR and mo.get("precision") == "float16" and mo.get("codage", "fp16") == codage
                     and mo.get("coremltools") == COREMLTOOLS, f"{d} : moteur {mo}")
            info.setdefault("macos", set()).add(mo.get("macos"))
        if du_commit is not None:
            A.exiger(m["git"] == du_commit, f"{d} : meta d'un autre commit")

    def serie_ne(d, codage, spec="fp32", du_commit=None):
        meta(d, codage, gptq=spec.endswith(":gptq"), du_commit=du_commit)
        S, emp, temps = 0, set(), []
        for g in GRAINES:
            x = lire(d, g, spec, MOTEUR, du_commit)
            tf = lire_temps(r / d / f"{nom(spec, g)[:-4]}.temps.json", str(x["git"]))
            A.exiger(set(tf["coreml"].get("placement", {})) == {"NeuralEngine"}, f"{d} graine {g} : opérations hors du Neural Engine")
            A.exiger(tf["coreml"].get("codage", "fp16") == codage, f"{d} graine {g} : codage {tf['coreml'].get('codage')}")
            A.exiger(tf["coreml"].get("sha256_poids") == str(x["coreml_sha256"]), f"{d} graine {g} : empreintes")
            octets = sum(f.stat().st_size for f in (r / d / f"{nom(spec, g)[:-4]}.mlpackage").rglob("weight.bin"))
            emp.add((str(x["coreml_sha256"]), octets))
            temps.append(tf["par_plan_s"])
            S = S + x["success"].astype(np.int64)
        A.exiger(len(emp) == 1, f"{d} : modèle Core ML différent selon les graines")
        sha_m, octets = emp.pop()
        return S, sha_m, octets, 1000 * float(np.median(temps))

    def serie_torch(d, spec, du_commit=None):
        meta(d, None, gptq=spec.endswith(":gptq"), du_commit=du_commit)
        return sum(lire(d, g, spec, du_commit=du_commit)["success"].astype(np.int64) for g in GRAINES)

    def reproductions(prefixe, c):
        for d, g in ((f"{prefixe}-repro-pusht", 42), (f"{prefixe}-fin-pusht", 46)):
            f = r / d / f"fp32_s{g}.npz"
            A.exiger(f.exists(), f"{d}/{f.name} manquant")
            A9.identiques(f, r / REF / f"fp32_s{g}.npz", f"reproduction de {prefixe}, graine {g}")
            x = np.load(f)
            A.exiger(str(x["git"]) == c and not bool(x["git_sale"]), f"{d} : reproduction d'un autre commit que la série")

    K = {}
    K["fp32"] = serie_torch(REF, "fp32", git_meta(REF))
    c15 = git_meta("exp015-int4c-sp-pusht")
    reproductions("exp015", c15)
    K["sp"], info["sha_sp"], info["octets_sp"], _ = serie_ne("exp015-int4c-sp-pusht", "int4c-sp", du_commit=c15)
    K["s_pr"] = serie_torch("exp015-sim-pusht", "predict:4", c15)
    for cle, d_ref, d_ses, codage in (("ne16", "exp012-ne-pusht", "exp017-ne16-pusht", "fp16"), ("int4c", "exp014-int4c-pusht", "exp017-int4c-pusht", "int4c")):
        c = git_meta(d_ref)
        meta(d_ses, codage)
        x = lire(d_ses, 42, moteur=MOTEUR)
        tf = lire_temps(r / d_ses / "fp32_s42.temps.json", str(x["git"]))
        info[f"temps_ancre_{cle}_ms"] = 1000 * tf["par_plan_s"]
        A.exiger(set(tf["coreml"].get("placement", {})) == {"NeuralEngine"}, f"{d_ses} : opérations hors du Neural Engine")
        A.exiger((r / d_ref / "fp32_s42.npz").exists(), f"{d_ref}/fp32_s42.npz manquant")
        info[f"{cle}_identique"] = bool(identiques(r / d_ses / "fp32_s42.npz", r / d_ref / "fp32_s42.npz"))
        if info[f"{cle}_identique"]:
            reproductions(d_ref.split("-")[0], c)
            K[cle], info[f"sha_{cle}"], info[f"octets_{cle}"], _ = serie_ne(d_ref, codage, du_commit=c)
        else:
            K[cle], info[f"sha_{cle}"], info[f"octets_{cle}"], _ = serie_ne(d_ses, codage)
        info[f"source_{cle}"] = d_ref if info[f"{cle}_identique"] else d_ses
        A.exiger(str(x["coreml_sha256"]) == info[f"sha_{cle}"], f"{d_ses} : modèle différent de celui de la série")
    K["g"], info["sha_g"], info["octets_g"], info["temps_g_ms"] = serie_ne("exp017-gptq-pusht", "int4c-gptq", GPTQ)
    K["gs"] = serie_torch("exp017-sim-pusht", GPTQ)
    lst = constantes_codees(r / "exp017-gptq-pusht" / f"{nom(GPTQ, 42)[:-4]}.mlpackage")
    A.exiger(isinstance(lst, list), f"G : constantes codées illisibles {lst}")
    k = categories(lst)
    A.exiger((k["blocs"], k["pred_proj"], k["reecriture"], k["autres"]) == (30, 2, 2, 0), f"G : constantes codées {k}")
    A.exiger(len({info["sha_g"], info["sha_ne16"], info["sha_int4c"], info["sha_sp"]}) == 4, "deux bras ont le même modèle Core ML")
    A.exiger(info["octets_g"] < info["octets_ne16"], "G : fichier des poids pas plus petit que le 16 bits")
    A.exiger(len(info["macos"]) == 1, f"versions de macOS différentes {info['macos']}")
    A.exiger(len(shas) == 1, f"EXP-017 : fichiers de plusieurs commits {sorted(shas)}")
    sha = shas.pop()
    for d, g in (("exp017-repro-pusht", 42), ("exp017-fin-pusht", 46)):
        x = np.load(r / d / f"fp32_s{g}.npz")
        A.exiger(str(x["git"]) == sha and not bool(x["git_sale"]), f"{d} : autre commit ou arbre modifié")
    A.exiger((r / "exp017-etape0" / "resultats.json").exists(), "étape 0 : sortie manquante")
    e0 = json.loads((r / "exp017-etape0" / "resultats.json").read_text())
    A.exiger(e0["git"] == sha and not e0["git_sale"], f"étape 0 au commit {e0['git'][:7]}, runs au commit {sha[:7]}")
    A.exiger(e0["codes_differents"] == 0 and set(e0["placement"]) == {"NeuralEngine"} and e0.get("couches_gptq") == 32,
             "étape 0 : codes, placement ou couches non conformes")
    rapports = [a["relative"] / b["relative"] for a, b in zip(e0["ne_contre_gptq_simule"], e0["repere_ne16_contre_fp32"])]
    A.exiger(len(rapports) == 3 and all(x <= 2 for x in rapports), f"étape 0 : sortie de la puce trop loin de la simulation {rapports}")
    info["etape0_rapports_ne_sur_repere"] = rapports
    A.exiger(info["octets_g"] == info["octets_int4c"], f"G : fichier des poids de {info['octets_g']} octets au lieu de {info['octets_int4c']} (int4c)")
    A.exiger(e0["sha256_poids"] == info["sha_g"], "étape 0 : modèle différent de celui des runs")
    if verifier:
        verifier_code(sha)
    info["constantes_g"] = {c: k[c] for c in ("blocs", "pred_proj", "reecriture")}
    info["macos"] = sorted(info["macos"])
    info["etape0"] = {x: e0[x] for x in ("codes_differents", "codes", "ecart_decode_max_part_ligne", "coremltools_part_codes_changes")}
    info["etape0"]["rapports_ne_sur_repere"] = info["etape0_rapports_ne_sur_repere"]
    return K, info, sha


def analyser(racine, verifier=True):
    K, info, sha = charger(racine, verifier)
    inv, boot = A.Inversion(n=500), A.Bootstrap(n=500)
    ecart = lambda a, b: K[a] - K[b]
    tests = {}
    for n, b, ref, s in TESTS:
        x = ecart(b, ref)
        y = 20 * x + MARGE if s == "non_inferiorite" else 20 * x - MARGE
        ps, pb = inv.p(y, "plus"), p_bootstrap_t(y)
        tests[n] = {"bras": b, "reference": ref, "sorte": s, "ecart_points": {k: 100 * v for k, v in boot.ic(x / 5).items()},
                    "p_signe": ps, "p_bootstrap_t": pb, "p": max(ps, pb)}
    ps = [tests[n]["p"] for n, *_ in TESTS]
    for (n, b, ref, s), q, niv in zip(TESTS, A.holm(ps), niveaux_holm(ps)):
        t = tests[n]
        t["p_holm"], t["niveau_holm"] = float(q), float(niv)
        t["verdict"] = "soutenue" if q < A.ALPHA else "non soutenue"
        t["borne_inferieure_points"] = 100 * borne_bootstrap_t(ecart(b, ref) / 5, niv)
        seuil = -MARGE if s == "non_inferiorite" else MARGE
        t["borne_et_verdict_concordent"] = (t["borne_inferieure_points"] > seuil) == (t["verdict"] == "soutenue")
    ic = lambda x: {k: 100 * v for k, v in boot.ic(x / 5).items()}
    d = {"g - fp32": ic(ecart("g", "fp32")), "gs - fp32": ic(ecart("gs", "fp32")), "gs - s_pr": ic(ecart("gs", "s_pr")),
         "puce contre simulation": ic(ecart("g", "ne16") - ecart("gs", "fp32")), "g - sp": ic(ecart("g", "sp")),
         "g - ne16": ic(ecart("g", "ne16")), "g - int4c": ic(ecart("g", "int4c"))}
    perte = ecart("ne16", "int4c").astype(np.float64)
    part = {"rapportee": ic(ecart("int4c", "ne16"))["haut"] < 0}      # perte d'int4c contre ne16 établie
    if part["rapportee"]:                              # part de la perte d'int4c rattrapée (amendement 1, point 3)
        num = ecart("g", "int4c").astype(np.float64)
        ratios = num[boot.idx].mean(axis=1) / perte[boot.idx].mean(axis=1)
        part.update({"part": float(num.mean() / perte.mean()), "bas": float(np.percentile(ratios, 2.5)), "haut": float(np.percentile(ratios, 97.5))})
    pc = d["puce contre simulation"]
    desc = {"ecarts": d, "perte_etablie_g_contre_ne16": d["g - ne16"]["haut"] < 0, "part_rattrapee": part,
            "puce_contre_simulation_lecture": "compatible" if pc["bas"] <= 0 <= pc["haut"] else "la simulation ne prédit pas la puce",
            "g_moins_sp_lecture": "descriptif" if info["ne16_identique"] and info["int4c_identique"] else "non comparable",
            "temps_ancres_session_ms": {"ne16": info["temps_ancre_ne16_ms"], "int4c": info["temps_ancre_int4c_ms"]},
            "reussite": {b: float(K[b].mean() / 5) for b in K},
            "octets_poids": {b: info[f"octets_{b}"] for b in ("ne16", "int4c", "sp", "g")}, "temps_par_plan_g_ms": info["temps_g_ms"]}
    concord = all(tests[n]["borne_et_verdict_concordent"] for n in ("G1", "G2"))
    lecture = tests["G1"]["verdict"] == "soutenue" and tests["G2"]["verdict"] == "soutenue" and concord
    if lecture:
        phrase = "phrase visée soutenue (G1 et G2, bornes concordantes)"
    elif tests["G2"]["verdict"] == "soutenue" and tests["G1"]["verdict"] != "soutenue":
        phrase = "fait mieux que l'arrondi au plus proche ; non-infériorité au 16 bits non établie"
    elif not concord:
        phrase = "non établie (borne et verdict discordants)"
    else:
        phrase = "non établie"
    controles = {"commit": sha, **{k: info[k] for k in ("ne16_identique", "int4c_identique", "source_ne16", "source_int4c",
                                                        "constantes_g", "macos", "etape0")}}
    return {"controles": controles, "tests": tests, "phrase_visee_soutenue": lecture, "lecture": phrase, "descriptifs": desc}


def rapport(r):
    f = lambda e: f"{e['moyenne']:+.1f} [{e['bas']:+.1f} ; {e['haut']:+.1f}]"
    L = ["# EXP-017 : rapport de l'analyse pré-enregistrée (codes de GPTQ dans Core ML, Neural Engine, PushT)", "",
         f"Commit des runs : `{r['controles']['commit'][:7]}` ; ne16 {r['controles']['source_ne16']}, int4c {r['controles']['source_int4c']} ; "
         f"étape 0 : {r['controles']['etape0']['codes_differents']} code différent sur {r['controles']['etape0']['codes']}.", "",
         "| Test | Bras contre référence | Écart (points) [IC 95 %] | Borne inférieure au niveau de Holm | p Holm | Verdict |",
         "| --- | --- | --- | --- | --- | --- |"]
    for n, t in r["tests"].items():
        L.append(f"| {n} ({t['sorte']}) | {t['bras']} contre {t['reference']} | {f(t['ecart_points'])} | {t['borne_inferieure_points']:+.2f} | "
                 f"{t['p_holm']:.4f} | {t['verdict']} |")
        if not t["borne_et_verdict_concordent"]:
            L.append(f"- Attention, {n} : borne et verdict ne concordent pas.")
    L += ["", f"Lecture (amendement 1) : {r['lecture']}.", "", "Descriptifs :", ""]
    L += [f"- {k} : {f(v)}" for k, v in r["descriptifs"]["ecarts"].items()]
    L += ["", "Réussite : " + ", ".join(f"{k} {100 * v:.1f} %" for k, v in r["descriptifs"]["reussite"].items()),
          "Fichier des poids (octets) : " + ", ".join(f"{k} {v}" for k, v in r["descriptifs"]["octets_poids"].items()),
          f"Temps par plan (ms, descriptif, sans comparaison affirmée) : G {r['descriptifs']['temps_par_plan_g_ms']:.0f} ; ancres de la session, "
          f"graine 42 : ne16 {r['descriptifs']['temps_ancres_session_ms']['ne16']:.0f}, int4c {r['descriptifs']['temps_ancres_session_ms']['int4c']:.0f}",
          f"Puce contre simulation : {r['descriptifs']['puce_contre_simulation_lecture']} ; G - int4c-sp : {r['descriptifs']['g_moins_sp_lecture']}",
          "Part de la perte d'int4c rattrapée : " + (f"{r['descriptifs']['part_rattrapee']['part']:.2f} [{r['descriptifs']['part_rattrapee']['bas']:.2f} ; "
          f"{r['descriptifs']['part_rattrapee']['haut']:.2f}]" if r["descriptifs"]["part_rattrapee"]["rapportee"] else "non rapportée"),
          "Étape 0, sortie de la puce contre la simulation, rapport au repère 16 bits par lot : " + ", ".join(f"{x:.3f}" for x in r["controles"]["etape0"]["rapports_ne_sur_repere"])]
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
