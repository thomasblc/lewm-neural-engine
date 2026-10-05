"""Analyse pré-enregistrée d'EXP-018 (PREREG-EXP018.md) : int4c, tout sauf pred_proj (SP) et les codes de GPTQ (G), sur le
Neural Engine, sur les 2 000 épisodes d'EXP-005 (graines 52 à 56), contre la pleine précision PyTorch d'EXP-005. Tests T1
à T4 (Holm sur 4) ; descriptifs ; rappel des écarts sur les 500 épisodes d'EXP-014, EXP-015 et EXP-017.

Usage :
    python analyse_exp018.py --racine ../resultats --sortie ../resultats/analyse-exp018
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

import analyse_exp003 as A
import analyse_exp009 as A9
from analyse_exp012 import borne_bootstrap_t, niveaux_holm, p_bootstrap_t
from analyse_exp014 import lire_temps

GRAINES = (52, 53, 54, 55, 56)
N = 2000
MARGE = 2
MOTEUR = "coreml-ne"
COREMLTOOLS = "9.0"
REF = "exp005"
GPTQ = "predict:4:gptq"
COMMIT_EXP017 = "8d2463192645777caa8100505ba8275e13e18465"     # commit des runs d'EXP-017 : modèles passés aux étapes 0
# (clé, dossier, spec, codage, couches, dossier de la même version sur 500 épisodes, spec de ce dossier)
BRAS = (("int4c", "exp018-int4c-pusht", "fp32", "int4c", 0, "exp014-int4c-pusht", "fp32"),
        ("sp", "exp018-int4c-sp-pusht", "fp32", "int4c-sp", 0, "exp015-int4c-sp-pusht", "fp32"),
        ("g", "exp018-gptq-pusht", GPTQ, "int4c-gptq", 32, "exp017-gptq-pusht", GPTQ))
TESTS = (("T1", "int4c", "fp32", "perte"), ("T2", "sp", "fp32", "non_inferiorite"), ("T3", "g", "fp32", "non_inferiorite"),
         ("T4", "g", "int4c", "superiorite"))
FICHIERS_CALCUL = ("metriques.py", "quant.py", "gptq.py", "run_grid.py", "coreml_pred.py", "predicteur_ane.py")
FICHIERS_ANALYSE = ("analyse_exp018.py", "analyse_exp014.py", "analyse_exp012.py", "analyse_exp009.py", "analyse_exp003.py",
                    "PREREG-EXP018.md")
ICI = Path(__file__).resolve().parent


def git(*args):
    return subprocess.run(["git", "-C", str(ICI), *args], capture_output=True, text=True)


def verifier_code(sha):
    """Code du commit des runs, propre ; calcul identique à celui d'EXP-017, dont les modèles ont passé les étapes 0."""
    A.exiger(git("cat-file", "-e", f"{sha}^{{commit}}").returncode == 0, f"commit {sha[:7]} inconnu du dépôt")
    A.exiger(git("diff", "--quiet", sha, "--", *FICHIERS_CALCUL, *FICHIERS_ANALYSE).returncode == 0,
             f"code d'analyse ou de calcul différent de celui du commit {sha[:7]}")
    A.exiger(git("diff", "--quiet", COMMIT_EXP017, sha, "--", *FICHIERS_CALCUL).returncode == 0,
             "code de calcul différent de celui d'EXP-017")


def nom(spec, g):
    return f"{spec.replace(':', '-')}_s{g}.npz"


def charger(racine, verifier=True):
    r = Path(racine)
    ref = dict(np.load(r / REF / "fp32_s52.npz"))
    A.exiger(len(ref["success"]) == N, f"{REF} : {len(ref['success'])} épisodes au lieu de {N}")
    proto_ref = json.loads((r / REF / "meta.json").read_text())["protocole"]
    calib10 = json.loads((r / "exp010" / "meta.json").read_text())["protocole"]["gptq_calib"]
    shas, info = set(), {"macos": set()}

    def lire(d, g, spec, couches, moteur=None, du_commit=None):
        f = r / d / nom(spec, g)
        A.exiger(f.exists(), f"{d}/{f.name} manquant")
        x = dict(np.load(f))
        A.exiger((str(x["spec"]), int(x["seed"])) == (spec, g), f"{d}/{f.name} : version ou graine")
        A.exiger(not bool(x["git_sale"]), f"{d}/{f.name} : arbre git modifié")
        A.exiger(int(x["n_couches"]) == couches, f"{d}/{f.name} : {int(x['n_couches'])} couches quantifiées au lieu de {couches}")
        A.exiger(str(x.get("moteur", "torch")) == (moteur or "torch"), f"{d}/{f.name} : moteur {x.get('moteur')}")
        for k in ("episodes_idx", "start_steps", "init_state", "goal_state"):
            A.exiger(np.array_equal(x[k], ref[k]), f"{d}/{f.name} : {k} différent de la référence")
        A.controler_course(x, f"{d}/{f.name}")
        if du_commit is not None:
            A.exiger(str(x["git"]) == du_commit, f"{d}/{f.name} : commit {str(x['git'])[:7]} au lieu de {du_commit[:7]}")
        else:
            shas.add(str(x["git"]))
        return x

    def meta(d, codage, gptq):
        A.exiger((r / d / "meta.json").exists(), f"{d}/meta.json manquant")
        p = json.loads((r / d / "meta.json").read_text())["protocole"]
        A.exiger({k: v for k, v in p.items() if k not in ("moteur", "gptq_calib")} == proto_ref, f"{d} : protocole différent de {REF}")
        A.exiger((p.get("gptq_calib") == calib10) if gptq else ("gptq_calib" not in p), f"{d} : calibration GPTQ {p.get('gptq_calib')}")
        mo = p.get("moteur", {})
        A.exiger(mo.get("nom") == MOTEUR and mo.get("precision") == "float16" and mo.get("codage", "fp16") == codage
                 and mo.get("coremltools") == COREMLTOOLS, f"{d} : moteur {mo}")
        info["macos"].add(mo.get("macos"))

    def empreinte_500(d, spec):
        """Empreinte du modèle de la même version sur les 500 épisodes (une seule sur ses graines)."""
        emp = {str(np.load(f)["coreml_sha256"]) for f in sorted((r / d).glob(f"{spec.replace(':', '-')}_s4[2-6].npz"))}
        A.exiger(len(emp) == 1, f"{d} : empreinte du modèle introuvable ou multiple")
        return emp.pop()

    K = {"fp32": sum(lire(REF, g, "fp32", 0, du_commit=json.loads((r / REF / "meta.json").read_text())["git"])["success"]
                     .astype(np.int64) for g in GRAINES)}
    for cle, d, spec, codage, couches, d500, spec500 in BRAS:
        meta(d, codage, spec.endswith(":gptq"))
        S, emp, temps = 0, set(), []
        for g in GRAINES:
            x = lire(d, g, spec, couches, MOTEUR)
            tf = lire_temps(r / d / f"{nom(spec, g)[:-4]}.temps.json", str(x["git"]))
            A.exiger(set(tf["coreml"].get("placement", {})) == {"NeuralEngine"}, f"{d} graine {g} : opérations hors du Neural Engine")
            A.exiger(tf["coreml"].get("codage", "fp16") == codage, f"{d} graine {g} : codage {tf['coreml'].get('codage')}")
            A.exiger(tf["coreml"].get("sha256_poids") == str(x["coreml_sha256"]), f"{d} graine {g} : empreintes")
            emp.add(str(x["coreml_sha256"]))
            temps.append(tf["par_plan_s"])
            S = S + x["success"].astype(np.int64)
        A.exiger(len(emp) == 1, f"{d} : modèle Core ML différent selon les graines")
        info[f"sha_{cle}"] = emp.pop()
        A.exiger(info[f"sha_{cle}"] == empreinte_500(d500, spec500), f"{d} : modèle différent de celui de {d500}")
        info[f"temps_{cle}_ms"] = 1000 * float(np.median(temps))
        K[cle] = S
    A.exiger(len(info["macos"]) == 1, f"versions de macOS différentes {info['macos']}")
    A.exiger(len(shas) == 1, f"EXP-018 : fichiers de plusieurs commits {sorted(shas)}")
    sha = shas.pop()
    for d, g, refd in (("exp018-repro-pusht", 52, REF), ("exp018-fin-pusht", 46, "exp003")):
        f = r / d / f"fp32_s{g}.npz"
        A.exiger(f.exists(), f"{d}/{f.name} manquant")
        A9.identiques(f, r / refd / f"fp32_s{g}.npz", f"reproduction de {refd}, graine {g}")
        x = np.load(f)
        A.exiger(str(x["git"]) == sha and not bool(x["git_sale"]), f"{d} : autre commit ou arbre modifié")
    if verifier:
        verifier_code(sha)
    info["macos"] = sorted(info["macos"])
    return K, info, sha


def ecarts_500(racine):
    """Rappel descriptif : les mêmes écarts sur les 500 épisodes d'EXP-003 (fichiers d'EXP-014, EXP-015, EXP-017)."""
    r = Path(racine)
    K = {"fp32": sum(np.load(r / "exp003" / f"fp32_s{g}.npz")["success"].astype(np.int64) for g in (42, 43, 44, 45, 46))}
    for cle, _, _, _, _, d500, spec500 in BRAS:
        K[cle] = sum(np.load(r / d500 / nom(spec500, g))["success"].astype(np.int64) for g in (42, 43, 44, 45, 46))
    boot = A.Bootstrap(n=500)
    ic = lambda x: {k: 100 * v for k, v in boot.ic(x / 5).items()}
    return {f"{b} - {ref}": ic(K[b] - K[ref]) for b, ref in (("int4c", "fp32"), ("sp", "fp32"), ("g", "fp32"), ("g", "int4c"))}


def analyser(racine, verifier=True):
    K, info, sha = charger(racine, verifier)
    inv, boot = A.Inversion(n=N), A.Bootstrap(n=N)
    ecart = lambda a, b: K[a] - K[b]
    y_de = {"non_inferiorite": lambda x: 20 * x + MARGE, "superiorite": lambda x: 20 * x - MARGE, "perte": lambda x: -20 * x - MARGE}
    tests = {}
    for n, b, ref, s in TESTS:
        x = ecart(b, ref)
        y = y_de[s](x)
        ps, pb = inv.p(y, "plus"), p_bootstrap_t(y)
        tests[n] = {"bras": b, "reference": ref, "sorte": s, "ecart_points": {k: 100 * v for k, v in boot.ic(x / 5).items()},
                    "p_signe": ps, "p_bootstrap_t": pb, "erreur_mc_p_signe": float(np.sqrt(ps * (1 - ps) / A.B)),
                    "erreur_mc_p_bootstrap_t": float(np.sqrt(pb * (1 - pb) / 10_000)), "p": max(ps, pb)}
    ps = [tests[n]["p"] for n, *_ in TESTS]
    for (n, b, ref, s), q, niv in zip(TESTS, A.holm(ps), niveaux_holm(ps)):
        t = tests[n]
        t["p_holm"], t["niveau_holm"] = float(q), float(niv)
        t["verdict"] = "soutenue" if q < A.ALPHA else "non soutenue"
        x = ecart(b, ref) / 5
        if s == "perte":                               # borne supérieure de l'écart, sous -2 ?
            t["borne_points"] = -100 * borne_bootstrap_t(-x, niv)
            t["borne_et_verdict_concordent"] = (t["borne_points"] < -MARGE) == (t["verdict"] == "soutenue")
        else:                                          # borne inférieure de l'écart, au-dessus de -2 ou de +2 ?
            t["borne_points"] = 100 * borne_bootstrap_t(x, niv)
            seuil = -MARGE if s == "non_inferiorite" else MARGE
            t["borne_et_verdict_concordent"] = (t["borne_points"] > seuil) == (t["verdict"] == "soutenue")
    ic = lambda x: {k: 100 * v for k, v in boot.ic(x / 5).items()}
    d = {f"{b} - {ref}": ic(ecart(b, ref)) for b, ref in (("int4c", "fp32"), ("sp", "fp32"), ("g", "fp32"), ("g", "int4c"), ("g", "sp"))}
    perte = ecart("fp32", "int4c").astype(np.float64)
    parts = {"rapportee": d["int4c - fp32"]["haut"] < 0}
    if parts["rapportee"]:
        for b in ("g", "sp"):
            num = ecart(b, "int4c").astype(np.float64)
            ratios = num[boot.idx].mean(axis=1) / perte[boot.idx].mean(axis=1)
            parts[b] = {"part": float(num.mean() / perte.mean()), "bas": float(np.percentile(ratios, 2.5)),
                        "haut": float(np.percentile(ratios, 97.5))}
    v = {n: tests[n]["verdict"] == "soutenue" and tests[n]["borne_et_verdict_concordent"] for n, *_ in TESTS}
    if v["T1"] and v["T2"]:
        loc = "la perte d'int4 et sa réparation par pred_proj en 16 bits se retrouvent sur des épisodes neufs"
    elif v["T1"]:
        loc = "int4c perd encore ; la recette n'est pas établie à 2 points sur ces épisodes"
    else:
        loc = "la perte d'int4c ne dépasse pas 2 points sur ces épisodes ; rien sur sa localisation ici"
    gptq = ("fait mieux que l'arrondi simple de plus de 2 points" if v["T4"] else "pas mieux de plus de 2 points établi") + " ; " + \
           ("ne perd pas plus de 2 points face à PyTorch float32" if v["T3"] else "non-infériorité face à PyTorch float32 non établie")
    desc = {"ecarts": d, "parts_rattrapees": parts, "reussite": {b: float(K[b].mean() / 5) for b in K},
            "temps_par_plan_ms": {b: info[f"temps_{b}_ms"] for b in ("int4c", "sp", "g")}, "ecarts_500_episodes": ecarts_500(racine)}
    return {"controles": {"commit": sha, "macos": info["macos"], **{f"sha_{b}": info[f"sha_{b}"] for b in ("int4c", "sp", "g")}},
            "tests": tests, "lecture": {"localisation": loc, "gptq": gptq}, "descriptifs": desc}


def rapport(r):
    f = lambda e: f"{e['moyenne']:+.1f} [{e['bas']:+.1f} ; {e['haut']:+.1f}]"
    L = ["# EXP-018 : rapport de l'analyse pré-enregistrée (2 000 épisodes neufs pour la puce, PushT)", "",
         f"Commit des runs : `{r['controles']['commit'][:7]}` ; modèles identiques à ceux d'EXP-014, EXP-015 et EXP-017.", "",
         "| Test | Bras contre référence | Écart (points) [IC 95 %] | Borne au niveau de Holm | p Holm | Verdict |",
         "| --- | --- | --- | --- | --- | --- |"]
    for n, t in r["tests"].items():
        L.append(f"| {n} ({t['sorte']}) | {t['bras']} contre {t['reference']} | {f(t['ecart_points'])} | {t['borne_points']:+.2f} | "
                 f"{t['p_holm']:.4f} | {t['verdict']} |")
        if not t["borne_et_verdict_concordent"]:
            L.append(f"- Attention, {n} : borne et verdict ne concordent pas.")
    L += ["", f"Lecture : {r['lecture']['localisation']} ; GPTQ : {r['lecture']['gptq']}.", "", "Descriptifs (2 000 épisodes) :", ""]
    L += [f"- {k} : {f(e)}" for k, e in r["descriptifs"]["ecarts"].items()]
    p = r["descriptifs"]["parts_rattrapees"]
    if p["rapportee"]:
        L += [f"- part de la perte d'int4c rattrapée par {b} : {p[b]['part']:.2f} [{p[b]['bas']:.2f} ; {p[b]['haut']:.2f}]" for b in ("g", "sp")]
    L += ["", "Rappel sur les 500 épisodes d'EXP-003 (EXP-014, EXP-015, EXP-017) :", ""]
    L += [f"- {k} : {f(e)}" for k, e in r["descriptifs"]["ecarts_500_episodes"].items()]
    L += ["", "Réussite : " + ", ".join(f"{k} {100 * v:.1f} %" for k, v in r["descriptifs"]["reussite"].items()),
          "Temps par plan (ms, descriptif) : " + ", ".join(f"{k} {v:.0f}" for k, v in r["descriptifs"]["temps_par_plan_ms"].items())]
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
