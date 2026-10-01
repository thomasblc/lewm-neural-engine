"""Analyse pré-enregistrée d'EXP-013 (PREREG-EXP013.md) : vitesse, énergie par composant et mémoire de la planification
selon le moteur, le codage des poids et une charge de vision. Descriptive : par configuration, médiane des 3 répétitions
(minimum et maximum) du temps par plan, de l'énergie par plan au-dessus du repos (CPU, GPU, Neural Engine, mémoire), des
images par seconde de la vision et de la taille des poids ; réussite des 50 épisodes en contrôle. Contrôles bloquants.

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

COMPOSANTS = ("CPU Energy", "GPU Energy", "ANE", "DRAM")
NOMS = {"CPU Energy": "CPU", "GPU Energy": "GPU", "ANE": "Neural Engine", "DRAM": "mémoire"}
PLACEMENT = {"coreml-ne": "NeuralEngine", "coreml-gpu": "GPU"}
FICHIERS_ANALYSE = ("analyse_exp013.py", "vitesse.py", "run_grid.py", "coreml_pred.py", "predicteur_ane.py", "energie_puce.py",
                    "repos_puce.py", "charge_vision.py", "quant.py", "PREREG-EXP013.md")
ICI = Path(__file__).resolve().parent


def git(*args):
    return subprocess.run(["git", "-C", str(ICI), *args], capture_output=True, text=True)


def verifier_code(sha):
    A.exiger(git("cat-file", "-e", f"{sha}^{{commit}}").returncode == 0, f"commit {sha[:7]} inconnu du dépôt")
    A.exiger(git("diff", "--quiet", sha, "--", *FICHIERS_ANALYSE).returncode == 0,
             f"code d'analyse ou de calcul différent de celui du commit {sha[:7]}")


def vision_pendant(lignes, debut, fin):
    """Images par seconde de la charge sur les mesures entièrement comprises dans [debut, fin]."""
    dedans = [x for x in lignes if x["t0"] >= debut and x["t1"] <= fin]
    if not dedans:
        return None
    return sum(x["images"] for x in dedans) / sum(x["t1"] - x["t0"] for x in dedans)


def lire_jsonl(f):
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip()] if f.exists() else []


def charger(racine, verifier=True):
    r = Path(racine)
    ordre = json.loads((r / "exp013-ordre.json").read_text())
    configs = ordre["configs"]
    attendus = [V.nom(m, c, ch) for m, c, ch in [(m, c, False) for m, c in V.SANS_CHARGE] + [(m, c, True) for m, c in V.AVEC_CHARGE]]
    A.exiger(set(attendus) <= set(configs), "exp013 : configurations différentes du plan")
    runs, shas, repos = {}, set(), {}
    for rep in range(1, V.REPETITIONS + 1):
        fr = sorted(r.glob(f"exp013-repos-r{rep}-*.json"))
        A.exiger(fr, f"répétition {rep} : mesure au repos manquante")
        j = json.loads(fr[-1].read_text())
        A.exiger(j["calculs_run_grid_vus"] == 0, f"répétition {rep} : un calcul tournait pendant la mesure au repos")
        repos[rep] = {k: j["joules"].get(k, 0.0) / j["duree_s"] for k in COMPOSANTS}
        shas.add(j["git"])
        for nom in attendus:
            d = r / f"exp013-{nom}-r{rep}"
            f = d / "fp32_s42.npz"
            A.exiger(f.exists(), f"{d.name}/{f.name} manquant")
            x = dict(np.load(f))
            A.exiger(not bool(x["git_sale"]) and int(x["n_couches"]) == 0, f"{d.name} : arbre modifié ou couches quantifiées")
            t = json.loads(f.with_suffix(".temps.json").read_text())
            A.exiger(t["git"] == str(x["git"]) and t.get("energie_puce"), f"{d.name} : fichier de temps ou énergie manquants")
            moteur = t["moteur"]
            if moteur in PLACEMENT:
                A.exiger(set(t["coreml"].get("placement", {})) == {PLACEMENT[moteur]}, f"{d.name} : opérations hors du processeur demandé")
            shas.add(str(x["git"]))
            runs[(nom, rep)] = {"temps": t, "succes": float(x["success"].mean()), "fichier": f}
    A.exiger(len(shas) == 1, f"EXP-013 : fichiers de plusieurs commits {sorted(shas)}")
    sha = shas.pop()
    ref = r / "pilote-exp012" / "torch" / "fp32_s42.npz"
    for f in [runs[("torch", rep)]["fichier"] for rep in range(1, V.REPETITIONS + 1)] + [r / "exp013-fin" / "fp32_s42.npz"]:
        A.exiger(V.identiques(f, ref), f"{f.parent.name} : run torch différent du pilote d'EXP-012")
    if verifier:
        verifier_code(sha)
    return runs, repos, sha, attendus


def par_run(run, repos_w, vision):
    t = run["temps"]
    d = np.array([a for a, _ in t["appels_solve"]], dtype=np.float64)
    n = np.array([b for _, b in t["appels_solve"]], dtype=np.float64)
    pp = d / np.maximum(n, 1)
    garde = pp <= 3 * np.median(pp)
    plans = n[garde].sum()
    e = t["energie_puce"]
    marg = {k: (e["joules"].get(k, 0.0) - repos_w[k] * e["duree_s"]) / t["plans"] for k in COMPOSANTS}
    v = vision_pendant(vision, e["debut"], e["debut"] + e["duree_s"]) if vision is not None else None
    return {"temps_par_plan_ms": 1000 * d[garde].sum() / plans, "energie_par_plan_j": marg,
            "energie_puce_par_plan_j": sum(marg.values()), "vision_ips": v, "succes": run["succes"],
            "octets_poids": (t.get("coreml") or {}).get("octets_poids")}


def resume(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return None
    return {"mediane": float(np.median(v)), "min": float(min(v)), "max": float(max(v))}


def analyser(racine, verifier=True):
    runs, repos, sha, attendus = charger(racine, verifier)
    r = Path(racine)
    res = {}
    for nom in attendus:
        lignes = []
        for rep in range(1, V.REPETITIONS + 1):
            vis = lire_jsonl(r / f"exp013-vision-{nom}-r{rep}.jsonl") if nom.endswith("-vision") else None
            x = par_run(runs[(nom, rep)], repos[rep], vis)
            if nom.endswith("-vision"):
                A.exiger(x["vision_ips"] is not None, f"{nom} r{rep} : aucune mesure de la charge de vision pendant le run")
            lignes.append(x)
        res[nom] = {"temps_par_plan_ms": resume([x["temps_par_plan_ms"] for x in lignes]),
                    "energie_puce_par_plan_j": resume([x["energie_puce_par_plan_j"] for x in lignes]),
                    **{f"energie_{NOMS[k]}_j": resume([x["energie_par_plan_j"][k] for x in lignes]) for k in COMPOSANTS},
                    "vision_ips": resume([x["vision_ips"] for x in lignes]), "succes": resume([x["succes"] for x in lignes]),
                    "octets_poids": lignes[0]["octets_poids"]}
    seule = resume([vision_pendant(lire_jsonl(r / f"exp013-vision-seule-r{rep}.jsonl"), 0, 1e12) for rep in range(1, V.REPETITIONS + 1)])
    return {"controles": {"commit": sha}, "configurations": res, "vision_seule_ips": seule,
            "repos_w": {f"r{k}": v for k, v in repos.items()}}


def rapport(r):
    f = lambda s, n=0, k=1: "?" if s is None else f"{k * s['mediane']:.{n}f} ({k * s['min']:.{n}f} à {k * s['max']:.{n}f})"
    L = ["# EXP-013 : vitesse, énergie et mémoire de la planification (analyse pré-enregistrée, descriptive)", "",
         f"Contrôles passés, commit {r['controles']['commit'][:7]} ; runs torch identiques au pilote d'EXP-012. Médiane des 3 "
         "répétitions (minimum à maximum).", "",
         "| Configuration | Temps par plan (ms) | Énergie de la puce par plan (J) | dont GPU | dont Neural Engine | dont CPU | Vision (images/s) | Poids (Mo) | Réussite (contrôle) |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for nom, c in r["configurations"].items():
        mo = f"{c['octets_poids'] / 1e6:.1f}" if c["octets_poids"] else ""
        L.append(f"| {nom} | {f(c['temps_par_plan_ms'])} | {f(c['energie_puce_par_plan_j'], 2)} | {f(c['energie_GPU_j'], 2)} | "
                 f"{f(c['energie_Neural Engine_j'], 2)} | {f(c['energie_CPU_j'], 2)} | {f(c['vision_ips'], 1)} | {mo} | {f(c['succes'], 0, 100)} % |")
    L += ["", f"Vision seule : {f(r['vision_seule_ips'], 1)} images/s."]
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
