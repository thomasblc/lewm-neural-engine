"""Analyse pré-enregistrée d'EXP-013 (PREREG-EXP013.md, amendement 2) : vitesse, énergie de la puce et mémoire de la
planification selon le moteur, le codage des poids et une charge de vision. Descriptive, avec une règle de décision
fixée d'avance pour dire qu'une différence est établie. Contrôles bloquants.

Usage :
    python analyse_exp013.py --racine ../resultats --sortie ../resultats/analyse-exp013
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

import analyse_exp003 as A
import vitesse as V

# Canaux de la puce (compteurs IOReport « Energy Model »), niveau haut seulement : « CPU Energy » contient PCPU et
# MCPU ; « GPU » double « GPU Energy » et n'est pas compté
SOC = ("CPU Energy", "GPU Energy", "ANE", "DRAM", "AMCC", "DCS", "FAB", "AFR", "ISP", "AVE", "MSR", "DISP", "DISPEXT")
GROUPES = {"calcul": ("CPU Energy", "GPU Energy", "ANE"), "memoire": ("DRAM", "AMCC", "DCS", "FAB"), "puce": SOC}
PRINCIPAUX = {"CPU": "CPU Energy", "GPU": "GPU Energy", "Neural Engine": "ANE"}
PLACEMENT = {"coreml-ne": "NeuralEngine", "coreml-gpu": "GPU", "ne-complet": "NeuralEngine", "coreml-cpu": "CPU"}
SUR_ANE = ("coreml-ne", "ne-complet")
MARGE_TEMPS, MARGE_ENERGIE = 0.05, 0.10
FICHIERS_ANALYSE = ("analyse_exp013.py", "vitesse.py", "run_grid.py", "coreml_pred.py", "predicteur_ane.py", "encodeur_ane.py",
                    "energie_puce.py", "repos_puce.py", "charge_vision.py", "balayage_lot.py", "quant.py", "PREREG-EXP013.md")
ICI = Path(__file__).resolve().parent


def git(*args):
    return subprocess.run(["git", "-C", str(ICI), *args], capture_output=True, text=True)


def verifier_code(sha):
    A.exiger(git("cat-file", "-e", f"{sha}^{{commit}}").returncode == 0, f"commit {sha[:7]} inconnu du dépôt")
    A.exiger(git("diff", "--quiet", sha, "--", *FICHIERS_ANALYSE).returncode == 0,
             f"code d'analyse ou de calcul différent de celui du commit {sha[:7]}")


def somme(j, canaux):
    return sum(j.get(k, 0.0) for k in canaux)


def lire_jsonl(f):
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip()] if f.exists() else []


def vision_fenetre(lignes, debut, fin):
    """Images par seconde, latences et échéances manquées par seconde, sur les mesures entièrement dans [debut, fin]."""
    d = [x for x in lignes if x["t0"] >= debut and x["t1"] <= fin]
    if not d:
        return None
    duree = sum(x["t1"] - x["t0"] for x in d)
    return {"ips": sum(x["images"] for x in d) / duree, "latence_p50_ms": float(np.median([x["latence_p50_ms"] for x in d])),
            "latence_p99_ms": float(np.max([x["latence_p99_ms"] for x in d])),
            "manquees_par_s": sum(x["echeances_manquees"] for x in d) / duree}


def repos_puissance(r, etiquette, sha):
    fr = sorted(r.glob(f"exp013-repos-{etiquette}-*.json"))
    A.exiger(fr, f"mesure au repos {etiquette} manquante")
    j = json.loads(fr[0].read_text())
    A.exiger(j["calculs_run_grid_vus"] == 0, f"mesure au repos {etiquette} : un calcul tournait")
    A.exiger(j["git"] == sha, f"mesure au repos {etiquette} : autre commit")
    return {k: v / j["duree_s"] for k, v in j["joules"].items()}


def charger(racine, verifier=True):
    r = Path(racine)
    ordre = json.loads((r / "exp013-ordre.json").read_text())
    configs = V.configurations()
    A.exiger(ordre["configs"] == [V.nom(*c) for c in configs], "exp013 : configurations différentes du plan")
    attendus = [V.nom(*c) for c in configs if not c[0].startswith("seule")]
    runs, shas = {}, set()
    for rep in range(1, V.REPETITIONS + 1):
        for nom in attendus:
            d = r / f"exp013-{nom}-r{rep}"
            f = d / "fp32_s42.npz"
            A.exiger(f.exists(), f"{d.name}/{f.name} manquant")
            x = dict(np.load(f))
            A.exiger(not bool(x["git_sale"]) and int(x["n_couches"]) == 0, f"{d.name} : arbre modifié ou couches quantifiées")
            t = json.loads(f.with_suffix(".temps.json").read_text())
            A.exiger(t["git"] == str(x["git"]) and t.get("energie_puce"), f"{d.name} : fichier de temps ou énergie manquants")
            ts = [y["t"] for y in t.get("lectures", []) if "t" in y]
            A.exiger(len(ts) >= 2 and not any(b - a > 12.5 for a, b in zip(ts, ts[1:])), f"{d.name} : pause de la garde pendant le run")
            moteur = t["moteur"]
            if moteur in PLACEMENT:
                A.exiger(set(t["coreml"].get("placement", {})) == {PLACEMENT[moteur]}, f"{d.name} : opérations hors du processeur demandé")
            shas.add(str(x["git"]))
            runs[(nom, rep)] = {"temps": t, "succes": float(x["success"].mean()), "fichier": f, "moteur": moteur}
    A.exiger(len(shas) == 1, f"EXP-013 : fichiers de plusieurs commits {sorted(shas)}")
    sha = shas.pop()
    etiquettes = ["r1-avant"] + [f"r{k}-r{k + 1}" for k in range(1, V.REPETITIONS)] + [f"r{V.REPETITIONS}-apres"]
    repos = {k: repos_puissance(r, e, sha) for k, e in enumerate(etiquettes)}
    for nom in attendus:                               # déterminisme : identique entre répétitions et avec ou sans vision
        base = nom.rsplit("-vision", 1)[0]
        for rep in range(1, V.REPETITIONS + 1):
            A.exiger(V.identiques(runs[(nom, rep)]["fichier"], runs[(base, 1)]["fichier"]),
                     f"{nom} r{rep} : trajectoires différentes de {base} r1 (calcul non déterministe ou perturbé)")
    ref = r / "pilote-exp012" / "torch" / "fp32_s42.npz"
    for f in [runs[("torch", rep)]["fichier"] for rep in range(1, V.REPETITIONS + 1)] + [r / "exp013-fin" / "fp32_s42.npz"]:
        A.exiger(V.identiques(f, ref), f"{f.parent.name} : run torch différent du pilote d'EXP-012")
    if verifier:
        verifier_code(sha)
    return runs, repos, sha, attendus


def par_run(run, repos_avant, repos_apres, vision, vision_seule30):
    t = run["temps"]
    e = t["energie_puce"]
    d = np.array([a for a, _ in t["appels_solve"]], dtype=np.float64)
    n = np.array([b for _, b in t["appels_solve"]], dtype=np.float64)
    plans, duree_solve = n.sum(), d.sum()
    idle = {k: (repos_avant.get(k, 0) + repos_apres.get(k, 0)) / 2 for k in SOC}

    def net(joules, duree, w):
        return {g: (somme(joules, c) - somme(w, c) * duree) / plans for g, c in GROUPES.items()}
    s = {"temps_par_plan_ms": 1000 * duree_solve / plans, "plans_par_s": plans / duree_solve,
         "energie_solve": net(e["joules_solve"], duree_solve, idle),
         "energie_solve_repos_avant": net(e["joules_solve"], duree_solve, repos_avant)["puce"],
         "energie_solve_repos_apres": net(e["joules_solve"], duree_solve, repos_apres)["puce"],
         "energie_run": net(e["joules"], e["duree_s"], idle),
         "composants_solve_w": {nom: e["joules_solve"].get(k, 0) / duree_solve - idle.get(k, 0) for nom, k in PRINCIPAUX.items()},
         "puissance_puce_w": somme(e["joules"], SOC) / e["duree_s"], "cpu_s_par_plan": e["cpu_s"] / plans,
         "maxrss_mo": e["maxrss_octets"] / 1e6,
         "part_coreml": e["appels_predict"]["coreml_s"] / duree_solve if e["appels_predict"]["n"] else None,
         "succes": run["succes"], "octets_poids": (t.get("coreml") or {}).get("octets_poids") or e.get("octets_modele_torch")}
    if vision is not None:
        s["vision"] = vision_fenetre(vision, e["debut"], e["debut"] + e["duree_s"])
        s["vision_avant"] = vision_fenetre(vision, e["debut"] - 60, e["debut"])
        s["energie_solve"] = None                      # avec charge : pas d'énergie par plan (travail de la vision élastique)
    if vision_seule30 is not None:                     # cadence fixe : énergie de planification = total moins vision seule
        p_v = somme(vision_seule30["joules"], SOC) / vision_seule30["duree_s"]
        s["energie_planification_30_j"] = (somme(e["joules_solve"], SOC) - p_v * duree_solve) / plans
    return s


def resume(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return None
    return {"mediane": float(np.median(v)), "min": float(min(v)), "max": float(max(v)), "valeurs": [float(x) for x in v]}


def decision(rapports, marge):
    """Différence établie si les 3 rapports appariés (même répétition) sont du même côté de 1, au-delà de la marge."""
    if len(rapports) < V.REPETITIONS:
        return "non calculable"
    if all(x < 1 - marge for x in rapports):
        return "plus bas, établi"
    if all(x > 1 + marge for x in rapports):
        return "plus haut, établi"
    return "non établi"


def vision_seule_ips(r, mode, k):
    l = lire_jsonl(r / f"exp013-vision-seule-{mode}-r{k}.jsonl")
    v = vision_fenetre(l, l[0]["t0"] + 10, 1e12) if l else None   # sans les 10 premières secondes (chauffe)
    return v["ips"] if v else None


def analyser(racine, verifier=True):
    runs, repos, sha, attendus = charger(racine, verifier)
    r = Path(racine)
    lignes = {}
    for nom in attendus:
        mode = "vision30" if nom.endswith("-vision30") else ("vision" if nom.endswith("-vision") else None)
        for rep in range(1, V.REPETITIONS + 1):
            vis = lire_jsonl(r / f"exp013-vision-{nom}-r{rep}.jsonl") if mode else None
            v30 = json.loads((r / f"exp013-vision-seule-vision30-r{rep}.json").read_text()) if mode == "vision30" else None
            x = par_run(runs[(nom, rep)], repos[rep - 1], repos[rep], vis, v30)
            if mode:
                A.exiger(x["vision"] is not None, f"{nom} r{rep} : aucune mesure de la charge de vision pendant le run")
            else:                                      # repli à l'exécution : le Neural Engine doit vraiment calculer
                w = x["composants_solve_w"]["Neural Engine"]
                if runs[(nom, rep)]["moteur"] in SUR_ANE:
                    A.exiger(w > 1.0, f"{nom} r{rep} : Neural Engine presque inactif ({w:.2f} W)")
                else:
                    A.exiger(w < 0.3, f"{nom} r{rep} : Neural Engine actif ({w:.2f} W) pour un moteur qui ne l'utilise pas")
            lignes[(nom, rep)] = x
    res = {}
    for nom in attendus:
        L = [lignes[(nom, rep)] for rep in range(1, V.REPETITIONS + 1)]
        c = {k: resume([x[k] for x in L]) for k in ("temps_par_plan_ms", "plans_par_s", "puissance_puce_w", "cpu_s_par_plan",
                                                    "maxrss_mo", "part_coreml", "succes")}
        c["octets_poids"] = L[0]["octets_poids"]
        if L[0]["energie_solve"] is not None:
            for g in GROUPES:
                c[f"energie_solve_{g}_j"] = resume([x["energie_solve"][g] for x in L])
            c["energie_solve_puce_sensibilite_j"] = {"repos_avant": resume([x["energie_solve_repos_avant"] for x in L]),
                                                     "repos_apres": resume([x["energie_solve_repos_apres"] for x in L])}
            c["energie_run_puce_j"] = resume([x["energie_run"]["puce"] for x in L])
            c["composants_solve_w"] = {k: resume([x["composants_solve_w"][k] for x in L]) for k in PRINCIPAUX}
        if "vision" in L[0]:
            for k in ("ips", "latence_p50_ms", "latence_p99_ms", "manquees_par_s"):
                c[f"vision_{k}"] = resume([x["vision"][k] for x in L])
                c[f"vision_avant_{k}"] = resume([x["vision_avant"][k] if x["vision_avant"] else None for x in L])
        if "energie_planification_30_j" in L[0]:
            c["energie_planification_30_j"] = resume([x["energie_planification_30_j"] for x in L])
        res[nom] = c
    comparaisons = {}
    R = range(1, V.REPETITIONS + 1)
    for nom in attendus:
        if nom.endswith(("-vision", "-vision30")) or nom == "torch":
            continue
        rt = [lignes[(nom, k)]["temps_par_plan_ms"] / lignes[("torch", k)]["temps_par_plan_ms"] for k in R]
        comparaisons[nom] = {"temps_contre_torch": {"rapports": rt, "verdict": decision(rt, MARGE_TEMPS)}}
        for ref in ("torch", "coreml-gpu"):
            if nom == ref:
                continue
            verdicts = []
            for cle in ("puce", "repos_avant", "repos_apres"):
                val = (lambda x: x["energie_solve"]["puce"]) if cle == "puce" else (lambda x, c=cle: x[f"energie_solve_{c}"])
                re = [val(lignes[(nom, k)]) / val(lignes[(ref, k)]) for k in R]
                comparaisons[nom][f"energie_contre_{ref}" + ("" if cle == "puce" else f"_{cle}")] = {"rapports": re, "verdict": decision(re, MARGE_ENERGIE)}
                verdicts.append(decision(re, MARGE_ENERGIE))
            comparaisons[nom][f"energie_contre_{ref}"]["robuste_au_repos"] = len(set(verdicts)) == 1
    seules = {}
    for mode in V.MODES_VISION:
        p = []
        for k in R:
            j = json.loads((r / f"exp013-vision-seule-{mode}-r{k}.json").read_text())
            p.append(somme(j["joules"], SOC) / j["duree_s"])
        seules[mode] = {"puissance_puce_w": resume(p), "ips": resume([vision_seule_ips(r, mode, k) for k in R])}
    f_b = r / "exp013-balayage-lot.json"
    return {"controles": {"commit": sha}, "configurations": res, "comparaisons": comparaisons, "vision_seule": seules,
            "repos_puce_w": {k: somme(w, SOC) for k, w in repos.items()}, "reussite_ne_complet": reussite(r, sha),
            "balayage_lot": json.loads(f_b.read_text()) if f_b.exists() else None}


def reussite(r, sha):
    """Contrôle du planificateur sur le Neural Engine sur les 500 épisodes d'EXP-003 : moyenne sur ses graines contre la
    pleine précision d'EXP-003 (5 graines), par épisode (mêmes épisodes ; tirages de la CEM différents, sur le CPU)."""
    m, graines = V.REUSSITE
    d = r / f"exp013-{m}-reussite"
    ref = dict(np.load(r / "exp003" / "fp32_s42.npz"))
    S = []
    for g in graines:
        x = dict(np.load(d / f"fp32_s{g}.npz"))
        A.exiger(str(x["git"]) == sha and not bool(x["git_sale"]), f"{d.name} graine {g} : autre commit ou arbre modifié")
        for k in ("episodes_idx", "start_steps", "init_state", "goal_state"):
            A.exiger(np.array_equal(x[k], ref[k]), f"{d.name} graine {g} : {k} différent d'EXP-003")
        A.controler_course(x, f"{d.name} graine {g}")
        S.append(x["success"].astype(np.float64))
    f = np.mean([np.load(r / "exp003" / f"fp32_s{g}.npz")["success"].astype(np.float64) for g in (42, 43, 44, 45, 46)], axis=0)
    return {"reussite": float(np.mean(S)), "reussite_fp32_exp003": float(f.mean()), "ecart": A.Bootstrap(n=500).ic(np.mean(S, axis=0) - f)}


def rapport(r):
    f = lambda s, n=0, k=1: "" if s is None else f"{k * s['mediane']:.{n}f} ({k * s['min']:.{n}f} à {k * s['max']:.{n}f})"
    L = ["# EXP-013 : vitesse, énergie et mémoire de la planification (analyse pré-enregistrée, descriptive)", "",
         f"Contrôles passés, commit {r['controles']['commit'][:7]} ; runs torch identiques au pilote d'EXP-012 ; chaque "
         "configuration identique d'une répétition à l'autre et avec ou sans vision. Médiane des 3 répétitions (min à max). "
         "Énergie de la puce (compteurs IOReport, canaux de niveau haut) pendant la planification seule, au-dessus du repos.", "",
         "## Sans charge", "",
         "| Configuration | Temps par plan (ms) | Énergie par plan, puce (J) | calcul (J) | mémoire (J) | Poids (Mo) | Part de l'appel Core ML | Réussite (50 épisodes) |",
         "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for nom, c in r["configurations"].items():
        if "vision" in nom:
            continue
        mo = f"{c['octets_poids'] / 1e6:.1f}" if c["octets_poids"] else ""
        L.append(f"| {nom} | {f(c['temps_par_plan_ms'])} | {f(c.get('energie_solve_puce_j'), 2)} | {f(c.get('energie_solve_calcul_j'), 2)} | "
                 f"{f(c.get('energie_solve_memoire_j'), 2)} | {mo} | {f(c['part_coreml'], 0, 100)} | {f(c['succes'], 0, 100)} |")
    L += ["", "| Comparaison | Temps contre torch | Énergie contre torch | Énergie contre coreml-gpu |", "| --- | --- | --- | --- |"]

    def fr(x):
        if not x:
            return ""
        return (f"{x['verdict']} ({', '.join(f'{v:.2f}' for v in x['rapports'])})"
                + ("" if x.get("robuste_au_repos", True) else " ; pas robuste au repos"))
    for nom, c in r["comparaisons"].items():
        L.append(f"| {nom} | {fr(c['temps_contre_torch'])} | {fr(c.get('energie_contre_torch'))} | {fr(c.get('energie_contre_coreml-gpu'))} |")
    L += ["", "## Avec charge de vision sur le GPU", "",
          "| Configuration | Temps par plan (ms) | Puissance de la puce (W) | Vision (images/s) | Vision avant le run | Latence p99 (ms) | Échéances manquées par s | Énergie de planification estimée (J par plan, cadence fixe) |",
          "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for nom, c in r["configurations"].items():
        if "vision" not in nom:
            continue
        L.append(f"| {nom} | {f(c['temps_par_plan_ms'])} | {f(c['puissance_puce_w'], 1)} | {f(c['vision_ips'], 1)} | "
                 f"{f(c.get('vision_avant_ips'), 1)} | {f(c['vision_latence_p99_ms'], 1)} | {f(c['vision_manquees_par_s'], 2)} | "
                 f"{f(c.get('energie_planification_30_j'), 2)} |")
    L += ["", "Vision seule : " + " ; ".join(f"{m} {f(v['ips'], 1)} images/s, puce {f(v['puissance_puce_w'], 1)} W"
                                           for m, v in r["vision_seule"].items())]
    rn = r["reussite_ne_complet"]
    L += ["", f"Contrôle de réussite du planificateur sur le Neural Engine (500 épisodes d'EXP-003, 2 graines) : "
          f"{100 * rn['reussite']:.1f} % contre {100 * rn['reussite_fp32_exp003']:.1f} % en pleine précision PyTorch, écart "
          f"{A.ic_txt(rn['ecart'], 100, 1)} points (tirages de la CEM différents : sur le CPU)."]
    if r["balayage_lot"]:
        lots = sorted({x["lot"] for x in r["balayage_lot"]})
        L += ["", "## Balayage de la taille du lot (appels isolés du prédicteur, ms, médiane)", "",
              "| Processeur | Codage | " + " | ".join(f"lot {b}" for b in lots) + " |", "| --- | --- | " + " | ".join("---" for _ in lots) + " |"]
        for u in ("ne", "gpu"):
            for cod in sorted({x["codage"] for x in r["balayage_lot"]}):
                v = {x["lot"]: x["mediane_ms"] for x in r["balayage_lot"] if x["unite"] == u and x["codage"] == cod}
                L.append(f"| {u} | {cod} | " + " | ".join(f"{v[b]:.2f}" for b in sorted(v)) + " |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--racine", required=True)
    ap.add_argument("--sortie", required=True)
    a = ap.parse_args()
    r = analyser(a.racine)
    out = Path(a.sortie)
    out.mkdir(parents=True, exist_ok=True)
    (out / "resultats.json").write_text(json.dumps(A.pur(r), indent=1, ensure_ascii=False))
    (out / "rapport.md").write_text(rapport(r))
    print((out / "rapport.md").read_text())


if __name__ == "__main__":
    main()
