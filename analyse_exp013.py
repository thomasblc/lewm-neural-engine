"""Analyse pré-enregistrée d'EXP-013 (PREREG-EXP013.md, amendements 2 à 6) : vitesse, énergie de la puce et mémoire de
la planification selon le moteur, le codage des poids et une charge de vision. Descriptive, avec une règle de constance
fixée d'avance (les 3 répétitions du même côté, au-delà d'une marge) ; ce n'est pas un test. Contrôles bloquants (données
incohérentes) et signalements (mesures valides mais à lire avec prudence).

Usage :
    python analyse_exp013.py --racine ../resultats --sortie ../resultats/analyse-exp013
    python analyse_exp013.py --racine DOSSIER_DE_TRAVAIL --sortie DOSSIER --essai     (chemin court de vitesse.py --essai)
"""
import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np

import analyse_exp003 as A
import coreml_pred as C
import vitesse as V

# Canaux de la puce (compteurs IOReport « Energy Model »), niveau haut seulement : « CPU Energy » contient PCPU et
# MCPU ; « GPU » double « GPU Energy » et n'est pas compté
SOC = ("CPU Energy", "GPU Energy", "ANE", "DRAM", "AMCC", "DCS", "FAB", "AFR", "ISP", "AVE", "MSR", "DISP", "DISPEXT")
PUCE = tuple(k for k in SOC if k not in ("DISP", "DISPEXT"))   # sans l'affichage : il ne calcule pas le plan
GROUPES = {"calcul": ("CPU Energy", "GPU Energy", "ANE"), "memoire": ("DRAM", "AMCC", "DCS", "FAB"), "puce": PUCE}
PRINCIPAUX = {"CPU": "CPU Energy", "GPU": "GPU Energy", "Neural Engine": "ANE"}
PLACEMENT = {"coreml-ne": "NeuralEngine", "coreml-gpu": "GPU", "ne-complet": "NeuralEngine", "coreml-cpu": "CPU"}
SUR_ANE = ("coreml-ne", "ne-complet")
HORS_GPU = ("ne-complet", "coreml-cpu")
MARGE_TEMPS, MARGE_ENERGIE = 0.05, 0.10
SEUIL_ANE_ACTIF, SEUIL_ANE_INACTIF, SEUIL_GPU_HORS = 1.0, 0.3, 0.3     # W au-dessus du repos
GARDE_30 = (29.7, 0.3)                    # 30 images/s gardées : au moins 29,7 images/s, au plus 0,3 échéance manquée/s
BRUIT_TORCH = 0.025                       # écart des 3 runs torch au-delà duquel la règle du temps est peu informative
SEUIL_FOND = 20.0                         # % d'un cœur : processus de fond pendant une mesure au repos
# placement attendu de chaque codage (essai de placement, resultats/preparation-exp013/placement-codages-70ecada.json)
PLACEMENT_CODAGE = {("coreml-ne", "int4b32"): {"CPU": 211, "NeuralEngine": 150}}
REPLI_CPU = (("coreml-ne", "int4b32"),)   # le seul cas où le Neural Engine peut calculer peu : signalé, pas bloquant
THERMIQUE_NORMALE = ("No thermal warning level", "No performance warning level")
FICHIERS_ANALYSE = ("analyse_exp013.py", "analyse_exp003.py", "vitesse.py", "run_grid.py", "coreml_pred.py",
                    "predicteur_ane.py", "encodeur_ane.py", "energie_puce.py", "repos_puce.py", "charge_vision.py",
                    "balayage_lot.py", "quant.py", "PREREG-EXP013.md")
ICI = Path(__file__).resolve().parent


def git(*args):
    return subprocess.run(["git", "-C", str(ICI), *args], capture_output=True, text=True)


def verifier_code(sha):
    A.exiger(git("cat-file", "-e", f"{sha}^{{commit}}").returncode == 0, f"commit {sha[:7]} inconnu du dépôt")
    A.exiger(git("diff", "--quiet", sha, "--", *FICHIERS_ANALYSE).returncode == 0,
             f"code d'analyse ou de calcul différent de celui du commit {sha[:7]}")


def somme(j, canaux):
    return sum(j.get(k, 0.0) for k in canaux)


def canaux_ok(j, quoi):
    A.exiger(all(k in j and math.isfinite(j[k]) for k in PUCE), f"{quoi} : canal IOReport manquant ou non fini")


def lire_jsonl(f):
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip()] if f.exists() else []


def images(lignes):
    """Fin (s, horloge time.time), latence (ms) et retard de chaque image d'une charge de vision."""
    f, l, r = [], [], []
    for x in lignes:
        if "fins" not in x:
            continue
        retard = set(x["retard"])
        f += x["fins"]
        l += x["lat_ms"]
        r += [i in retard for i in range(len(x["fins"]))]
    return np.array(f, dtype=np.float64), np.array(l, dtype=np.float64), np.array(r, dtype=bool)


couverture = V.couverture                 # source unique : vitesse.py refait un run que l'analyse refuserait


def p99_tranches(lignes, fenetres):
    """Maximum des 99e centiles par tranche de 5 s, sur les tranches qui touchent une fenêtre (amendement 3, point 2)."""
    v = [x["latence_p99_ms"] for x in lignes if "fins" in x and x["latence_p99_ms"] is not None
         and any(x["t0"] < h1 and x["t1"] > h0 for h0, h1 in fenetres)]
    return max(v) if v else None


def vision_fenetres(img, fenetres):
    """Images par seconde, latences médiane et 99e centile (sur toutes les images), échéances manquées par seconde et
    plus grand trou sans image, sur les images finies dans les fenêtres [h0, h1]."""
    f, l, r = img
    duree = sum(h1 - h0 for h0, h1 in fenetres)
    m = np.zeros(len(f), dtype=bool)
    trou = 0.0
    for h0, h1 in fenetres:
        dans = (f >= h0) & (f <= h1)
        m |= dans
        trou = max(trou, float(np.diff(np.concatenate([[h0], np.sort(f[dans]), [h1]])).max()))
    if duree <= 0:
        return None
    if not m.any():                                    # aucune image finie dans les fenêtres : vision affamée, un résultat
        return {"ips": 0.0, "images": 0, "latence_p50_ms": None, "latence_p99_ms": None, "manquees_par_s": 0.0, "trou_max_s": trou}
    return {"ips": float(m.sum() / duree), "images": int(m.sum()), "latence_p50_ms": float(np.median(l[m])),
            "latence_p99_ms": float(np.percentile(l[m], 99)), "manquees_par_s": float(r[m].sum() / duree), "trou_max_s": trou}


def garde_30(v):
    return v is not None and v["ips"] >= GARDE_30[0] and v["manquees_par_s"] <= GARDE_30[1]


def lire_repos(r, etiquette, sha, essai=False):
    fr = sorted(r.glob(f"exp013-repos-{etiquette}-*.json"))
    A.exiger(fr, f"mesure au repos {etiquette} manquante")
    A.exiger(len(fr) == 1, f"mesure au repos {etiquette} : plusieurs fichiers")
    j = json.loads(fr[0].read_text())
    A.exiger(essai or j["calculs_run_grid_vus"] == 0, f"mesure au repos {etiquette} : un calcul tournait")
    A.exiger(j["git"] == sha, f"mesure au repos {etiquette} : autre commit")
    canaux_ok(j["joules"], f"mesure au repos {etiquette}")
    fond = [x for x in j.get("processus_debut", []) + j.get("processus_fin", []) if x[0] > SEUIL_FOND and x[1] not in C.NON_COMPTES]
    return {"w": {k: v / j["duree_s"] for k, v in j["joules"].items()}, "debut": j["debut"], "fin": j["fin"], "fond": fond}


def lire_run(d, moteur, codage, sha_attendu=None, essai=False):
    f = d / "fp32_s42.npz"
    A.exiger(f.exists(), f"{d.name}/{f.name} manquant")
    x = dict(np.load(f))
    A.exiger(not bool(x["git_sale"]) and int(x["n_couches"]) == 0, f"{d.name} : arbre modifié ou couches quantifiées")
    t = json.loads(f.with_suffix(".temps.json").read_text())
    e = t.get("energie_puce")
    A.exiger(t["git"] == str(x["git"]) and e, f"{d.name} : fichier de temps ou énergie manquants")
    A.exiger(t["moteur"] == moteur and (moteur.startswith("torch") or t["coreml"].get("codage", "fp16") == codage),
             f"{d.name} : moteur ou codage différents du nom du dossier")
    ts = [y["t"] for y in t.get("lectures", []) if "t" in y]
    A.exiger(len(ts) >= 2 and not any(b - a > V.SEUIL_TROU_S for a, b in zip(ts, ts[1:])), f"{d.name} : pause pendant le run")
    A.exiger(essai or max(y.get("processus_run_grid", 0) for y in t["lectures"]) == 0, f"{d.name} : autre calcul run_grid pendant le run")
    if moteur in PLACEMENT:                            # placement attendu : celui de l'essai de placement (amendement 5)
        pl = t["coreml"].get("placement", {})
        attendu = PLACEMENT_CODAGE.get((moteur, codage))
        A.exiger(pl == attendu if attendu else set(pl) == {PLACEMENT[moteur]}, f"{d.name} : opérations hors du processeur demandé ({pl})")
    canaux_ok(e["joules_solve"], d.name)
    canaux_ok(e["joules"], d.name)
    A.exiger(len(e.get("fenetres_solve") or []) == len(t["appels_solve"]) > 0, f"{d.name} : fenêtres de planification manquantes")
    if sha_attendu:
        A.exiger(str(x["git"]) == sha_attendu, f"{d.name} : autre commit")
    thermique = [y.get("thermique", "") for y in t["lectures"]]
    autres = [p for y in t["lectures"] for p in y.get("processus", []) if p[1] not in C.NON_COMPTES]
    return {"temps": t, "succes": float(x["success"].mean()), "n_episodes": int(len(x["success"])), "fichier": f,
            "moteur": moteur, "codage": codage, "debut": e["debut"], "fin": e["debut"] + e["duree_s"], "git": str(x["git"]),
            "alerte_thermique": any(not all(k in s for k in THERMIQUE_NORMALE) for s in thermique),
            "batterie_min": min((y["CurrentCapacity"] for y in t["lectures"] if "CurrentCapacity" in y), default=None),
            "secteur": all(y.get("ExternalConnected", True) for y in t["lectures"]),
            "autre_processus_max": max(autres, key=lambda p: p[0]) if autres else None}


def charger(racine, verifier=True, essai=False):
    r = Path(racine)
    ordre = json.loads((r / "exp013-ordre.json").read_text())
    configs = V.configurations()
    A.exiger(ordre["configs"] == [V.nom(*c) for c in configs], "exp013 : configurations différentes du plan")
    A.exiger(ordre["ordres"] == V.ordres_contrebalances(len(configs), np.random.default_rng(13), configs),
             "exp013 : ordre différent de l'ordre tiré d'avance")
    attendus = {V.nom(*c): c for c in configs if not c[0].startswith("seule")}
    runs = {}
    for rep in range(1, V.REPETITIONS + 1):
        for nom, (m, c, _) in attendus.items():
            runs[(nom, rep)] = lire_run(r / f"exp013-{nom}-r{rep}", m, c, essai=essai)
    shas = {x["git"] for x in runs.values()}
    A.exiger(len(shas) == 1, f"EXP-013 : fichiers de plusieurs commits {sorted(shas)}")
    sha = shas.pop()
    A.exiger(ordre.get("git") == sha, "exp013-ordre.json : autre commit")
    seules = {}
    for rep in range(1, V.REPETITIONS + 1):
        for mode, ips in V.MODES_VISION.items():
            fj = r / f"exp013-vision-seule-{mode}-r{rep}.json"
            A.exiger(fj.exists(), f"{fj.name} manquant")
            j = json.loads(fj.read_text())
            A.exiger(j.get("git") == sha and j.get("ips_cible") == ips, f"{fj.name} : autre commit ou autre cadence")
            canaux_ok(j["joules"], fj.name)
            lv = lire_jsonl(fj.with_suffix(".jsonl"))
            A.exiger(lv and lv[0].get("ips_cible") == ips and lv[0]["pret"] <= j["debut"], f"{fj.name} : vision pas prête avant la mesure")
            v = vision_fenetres(images(lv), [(j["debut"], j["fin"])])
            A.exiger(v is not None and couverture(lv, [(j["debut"], j["fin"])]), f"{fj.name} : vision interrompue pendant la mesure")
            A.exiger(v["ips"] > 0, f"{fj.name} : aucune image pendant la mesure de la vision seule")
            seules[(mode, rep)] = {"json": j, "vision": v, "puissance_puce_w": somme(j["joules"], PUCE) / j["duree_s"]}
    for rep in range(1, V.REPETITIONS + 1):            # ordre réel = ordre tiré
        t = []
        for i in ordre["ordres"][rep - 1]:
            m, c, mode = configs[i]
            t.append(seules[(mode, rep)]["json"]["debut"] if m.startswith("seule") else runs[(V.nom(m, c, mode), rep)]["debut"])
        A.exiger(all(a < b for a, b in zip(t, t[1:])), f"répétition {rep} : runs dans un autre ordre que l'ordre tiré")
    sens = {}
    for m, variante, k in V.SENSIBILITE_OMP:
        sens[(m, variante, k)] = lire_run(r / f"exp013-sens-{m}-omp-{variante}-{k}", m, "fp16", sha, essai)
        A.exiger((sens[(m, variante, k)]["temps"]["energie_puce"]["omp"].get("OMP_WAIT_POLICY") == "PASSIVE") == (variante == "passif"),
                 f"sensibilité OpenMP {m} {variante}-{k} : variable OMP_WAIT_POLICY inattendue")
    etiquettes = ["r1-avant"] + [f"r{k}-r{k + 1}" for k in range(1, V.REPETITIONS)] + [f"r{V.REPETITIONS}-apres", "sens-apres"]
    repos = {e: lire_repos(r, e, sha, essai) for e in etiquettes}

    def bornes(rep):
        d = [x["debut"] for (n, k), x in runs.items() if k == rep] + [s["json"]["debut"] for (m, k), s in seules.items() if k == rep]
        f = [x["fin"] for (n, k), x in runs.items() if k == rep] + [s["json"]["fin"] for (m, k), s in seules.items() if k == rep]
        return min(d), max(f)
    blocs = [bornes(k) for k in range(1, V.REPETITIONS + 1)] + [(min(x["debut"] for x in sens.values()), max(x["fin"] for x in sens.values()))]
    for k, e in enumerate(etiquettes):                 # chaque repos entre le bloc qui le précède et celui qui le suit
        A.exiger(k == 0 or repos[e]["debut"] >= blocs[k - 1][1], f"mesure au repos {e} : faite avant la fin des runs qui la précèdent")
        A.exiger(k == len(blocs) or repos[e]["fin"] <= blocs[k][0], f"mesure au repos {e} : faite après le début des runs qui la suivent")
    ref = r / "exp013-torch-r1" / "fp32_s42.npz" if essai else r / "pilote-exp012" / "torch" / "fp32_s42.npz"
    fin = r / "exp013-fin" / "fp32_s42.npz"
    A.exiger(fin.exists(), "exp013-fin : run torch final manquant")
    for f in [runs[("torch", rep)]["fichier"] for rep in range(1, V.REPETITIONS + 1)] + [fin]:
        A.exiger(V.identiques(f, ref), f"{f.parent.name} : run torch différent du pilote d'EXP-012")
    if verifier:
        verifier_code(sha)
    return runs, repos, seules, sens, sha, list(attendus)


def determinisme(runs, attendus):
    """Signalement par configuration (pas bloquant) : identique d'une répétition à l'autre, et avec ou sans vision."""
    d = {}
    for nom in attendus:
        base = nom.rsplit("-vision", 1)[0]
        d[nom] = {"identique_entre_repetitions": all(V.identiques(runs[(nom, k)]["fichier"], runs[(nom, 1)]["fichier"])
                                                     for k in range(2, V.REPETITIONS + 1)),
                  "identique_sans_vision": None if base == nom else V.identiques(runs[(nom, 1)]["fichier"], runs[(base, 1)]["fichier"])}
    return d


def octets_predicteur(t):
    """Taille des poids du seul prédicteur (prédicteur et projection de sortie), comptée de la même façon pour tous :
    fichier weight.bin du prédicteur Core ML, ou paramètres PyTorch au type utilisé."""
    c = t.get("coreml") or {}
    if c:
        return (c.get("octets_par_modele") or {}).get("predicteur", c.get("octets_poids"))
    o = t["energie_puce"]["octets_modules_torch"]
    return o.get("predictor", 0) + o.get("pred_proj", 0)


def par_run(run, repos_avant, repos_apres, lignes_vision=None, seule30=None, bornes30=None):
    t = run["temps"]
    e = t["energie_puce"]
    d = np.array([a for a, _ in t["appels_solve"]], dtype=np.float64)
    n = np.array([b for _, b in t["appels_solve"]], dtype=np.float64)
    plans, duree_solve = n.sum(), d.sum()
    fenetres = [tuple(x) for x in e["fenetres_solve"]]
    idle = {k: (repos_avant.get(k, 0) + repos_apres.get(k, 0)) / 2 for k in SOC}

    def net(joules, duree, w):
        return {g: (somme(joules, c) - somme(w, c) * duree) / plans for g, c in GROUPES.items()}
    s = {"temps_par_plan_ms": 1000 * duree_solve / plans, "plans_par_s": plans / duree_solve,
         "energie_solve": net(e["joules_solve"], duree_solve, idle),
         "energie_solve_brute_puce": somme(e["joules_solve"], PUCE) / plans,
         "energie_solve_repos_avant": net(e["joules_solve"], duree_solve, repos_avant)["puce"],
         "energie_solve_repos_apres": net(e["joules_solve"], duree_solve, repos_apres)["puce"],
         "energie_run": net(e["joules"], e["duree_s"], idle),
         "composants_solve_w": {nom: e["joules_solve"].get(k, 0) / duree_solve - idle.get(k, 0) for nom, k in PRINCIPAUX.items()},
         "puissance_puce_w": somme(e["joules"], PUCE) / e["duree_s"], "puissance_solve_puce_w": somme(e["joules_solve"], PUCE) / duree_solve,
         "cpu_s_par_plan": e["cpu_s"] / plans,
         "maxrss_mo": e["maxrss_octets"] / 1e6,
         "part_coreml": e["appels_predict"]["coreml_s"] / duree_solve if e["appels_predict"]["n"] else None,
         "succes": run["succes"], "octets_predicteur": octets_predicteur(t)}
    if lignes_vision is not None:
        img = images(lignes_vision)
        s["vision_couvre"] = couverture(lignes_vision, fenetres)
        s["vision"] = vision_fenetres(img, fenetres)
        if s["vision"]:
            s["vision"]["latence_p99_max_tranches_ms"] = p99_tranches(lignes_vision, fenetres)
        s["vision_avant"] = vision_fenetres(img, [(lignes_vision[0]["pret"], e["debut"])]) if e["debut"] > lignes_vision[0]["pret"] else None
        s["energie_solve"] = None                      # avec charge : pas d'énergie par plan au sens sans charge
    if seule30 is not None:                            # cadence fixe : énergie de planification au-dessus de la vision seule
        s["garde_30"] = garde_30(s["vision"])
        e_s = somme(e["joules_solve"], PUCE)
        s["energie_planification_30_j"] = (e_s - seule30["puissance_puce_w"] * duree_solve) / plans if s["garde_30"] else None
        # sensibilité : puissance de la vision seule à 30 images/s la plus basse et la plus haute des 3 répétitions
        s["energie_planification_30_bornes_j"] = [(e_s - p * duree_solve) / plans for p in bornes30] if s["garde_30"] else None
    return s


def resume(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return None
    return {"mediane": float(np.median(v)), "min": float(min(v)), "max": float(max(v)), "valeurs": [float(x) for x in v]}


def decision(rapports, marge):
    """Règle de constance : les 3 rapports appariés (même répétition) du même côté de 1, au-delà de la marge."""
    if len(rapports) < V.REPETITIONS or any(x is None or x != x or x <= 0 for x in rapports):   # l'infini reste permis
        return "non calculable"
    if all(x < 1 - marge for x in rapports):
        return "plus bas dans les 3 répétitions"
    if all(x > 1 + marge for x in rapports):
        return "plus haut dans les 3 répétitions"
    return "non constant"


def reponse_q2(s, v, g_ne, g_ref, interp30=True):
    """Réponse à la question 2 (amendement 6). s, v : verdicts de la règle de constance pour le ralentissement S (plus
    bas = en faveur de ne-complet) et la part de la vision V (plus haute = en faveur) ; g_ne, g_ref : 30 images/s gardées.
    « compromis » si un critère est en faveur et un autre contre ; « oui » si S et V sont en faveur et si ne-complet
    garde les 30 images/s (ou si le bras à 30 images/s n'est pas interprétable) ; « non » si aucun critère n'est en
    faveur ; « en partie » sinon."""
    pour = [s == "plus bas dans les 3 répétitions", v == "plus haut dans les 3 répétitions", interp30 and g_ne and not g_ref]
    contre = [s == "plus haut dans les 3 répétitions", v == "plus bas dans les 3 répétitions", interp30 and g_ref and not g_ne]
    if any(pour) and any(contre):
        return "compromis"
    if pour[0] and pour[1] and (g_ne or not interp30):
        return "oui"
    if not any(pour):
        return "non"
    return "en partie"


def wilson(p, n, z=1.96):
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [c - h, c + h]


def paires_sans_charge(noms):
    p = [(x, "torch") for x in noms if x != "torch"]
    p += [(x, "coreml-gpu") for x in ("coreml-ne", "ne-complet", "coreml-cpu")]
    p += [("coreml-gpu", "torch-fp16"), ("ne-complet", "coreml-ne"), ("ne-complet", "coreml-cpu")]
    p += [(f"{u}-{c}", u) for u in ("coreml-gpu", "coreml-ne") for c in ("int8", "int4b32", "pal4", "int4c")]
    p += [(f"{u}-int4b32", f"{u}-int4c") for u in ("coreml-gpu", "coreml-ne")]          # blocs contre canal (amendement 4)
    return [x for i, x in enumerate(p) if x not in p[:i] and x[0] in noms and x[1] in noms]


def analyser(racine, verifier=True, essai=False):
    runs, repos, seules, sens, sha, attendus = charger(racine, verifier, essai)
    r = Path(racine)
    R = range(1, V.REPETITIONS + 1)
    etiq = ["r1-avant"] + [f"r{k}-r{k + 1}" for k in range(1, V.REPETITIONS)] + [f"r{V.REPETITIONS}-apres", "sens-apres"]
    lignes, signalements = {}, []
    p30 = [seules[("vision30", k)]["puissance_puce_w"] for k in R]
    bornes30 = (min(p30), max(p30))
    for (mode, k), sv in seules.items():               # la vision seule doit tenir sa cadence, sinon le mode 30 ne veut rien dire
        if mode == "vision30" and not garde_30(sv["vision"]):
            signalements.append(f"vision seule à 30 images/s r{k} : cadence non tenue ({sv['vision']['ips']:.1f} images/s, "
                                f"{sv['vision']['manquees_par_s']:.2f} échéance manquée par s)")
    signalements += [f"mesure au repos {e} : processus de fond {x['fond']}" for e, x in repos.items() if x["fond"]]
    for nom in attendus:
        mode = "vision30" if nom.endswith("-vision30") else ("vision" if nom.endswith("-vision") else None)
        for k in R:
            lv = None
            if mode:
                lv = lire_jsonl(r / f"exp013-vision-{nom}-r{k}.jsonl")
                A.exiger(lv and lv[0].get("ips_cible") == V.MODES_VISION[mode] and lv[0]["pret"] <= runs[(nom, k)]["debut"],
                         f"{nom} r{k} : charge de vision absente, d'une autre cadence ou pas prête avant le run")
            x = par_run(runs[(nom, k)], repos[etiq[k - 1]]["w"], repos[etiq[k]]["w"], lv,
                        seules[("vision30", k)] if mode == "vision30" else None, bornes30 if mode == "vision30" else None)
            if mode:
                A.exiger(x["vision"] is not None and x["vision_couvre"],
                         f"{nom} r{k} : la charge de vision ne couvre pas les fenêtres de planification (arrêtée ou absente)")
                x["vision_part_seule"] = x["vision"]["ips"] / seules[(mode, k)]["vision"]["ips"]   # référence : vision seule, même répétition
            w = x["composants_solve_w"]["Neural Engine"]           # repli silencieux de Core ML : le Neural Engine calcule-t-il ?
            if runs[(nom, k)]["moteur"] in SUR_ANE and mode:  # sous charge, la puissance du Neural Engine baisse : signalé
                if w <= SEUIL_ANE_ACTIF:
                    signalements.append(f"{nom} r{k} : Neural Engine à {w:.2f} W sous charge de vision")
            elif runs[(nom, k)]["moteur"] in SUR_ANE and runs[(nom, k)]["codage"] == "fp16":
                A.exiger(w > SEUIL_ANE_ACTIF, f"{nom} r{k} : Neural Engine presque inactif ({w:.2f} W)")
            elif (runs[(nom, k)]["moteur"], runs[(nom, k)]["codage"]) in REPLI_CPU:   # int4 par blocs : repli connu sur le CPU
                if w <= SEUIL_ANE_ACTIF:
                    signalements.append(f"{nom} r{k} : Neural Engine à {w:.2f} W seulement (placement {runs[(nom, k)]['temps']['coreml'].get('placement')})")
            elif runs[(nom, k)]["moteur"] in SUR_ANE:      # autres codages sur le Neural Engine : même contrôle qu'en 16 bits
                A.exiger(w > SEUIL_ANE_ACTIF, f"{nom} r{k} : Neural Engine presque inactif ({w:.2f} W)")
            else:
                A.exiger(w < SEUIL_ANE_INACTIF, f"{nom} r{k} : Neural Engine actif ({w:.2f} W) pour un moteur qui ne l'utilise pas")
            if runs[(nom, k)]["moteur"] in HORS_GPU and not mode and x["composants_solve_w"]["GPU"] > SEUIL_GPU_HORS:
                signalements.append(f"{nom} r{k} : GPU à {x['composants_solve_w']['GPU']:.2f} W au-dessus du repos (planificateur censé être hors du GPU)")
            if x["energie_solve"] is not None and x["energie_solve"]["puce"] <= 0:
                signalements.append(f"{nom} r{k} : énergie par plan au-dessus du repos négative ou nulle")
            if runs[(nom, k)]["alerte_thermique"]:
                signalements.append(f"{nom} r{k} : alerte thermique ou de performance pendant le run")
            if not runs[(nom, k)]["secteur"]:
                signalements.append(f"{nom} r{k} : chargeur débranché pendant le run")
            lignes[(nom, k)] = x
    res = {}
    for nom in attendus:
        L = [lignes[(nom, k)] for k in R]
        c = {k: resume([x[k] for x in L]) for k in ("temps_par_plan_ms", "plans_par_s", "puissance_puce_w", "puissance_solve_puce_w",
                                                    "cpu_s_par_plan", "maxrss_mo", "part_coreml", "succes", "energie_solve_brute_puce")}
        c["octets_predicteur"] = L[0]["octets_predicteur"]
        c["placement"] = (runs[(nom, 1)]["temps"].get("coreml") or {}).get("placement")
        c["octets_par_modele"] = (runs[(nom, 1)]["temps"].get("coreml") or {}).get("octets_par_modele")
        c["octets_modules_torch"] = runs[(nom, 1)]["temps"]["energie_puce"]["octets_modules_torch"]
        c["succes_ic95"] = wilson(L[0]["succes"], runs[(nom, 1)]["n_episodes"])
        c["composants_solve_w"] = {k: resume([x["composants_solve_w"][k] for x in L]) for k in PRINCIPAUX}
        if L[0]["energie_solve"] is not None:
            for g in GROUPES:
                c[f"energie_solve_{g}_j"] = resume([x["energie_solve"][g] for x in L])
            c["energie_solve_puce_sensibilite_j"] = {"repos_avant": resume([x["energie_solve_repos_avant"] for x in L]),
                                                     "repos_apres": resume([x["energie_solve_repos_apres"] for x in L])}
            c["energie_run_puce_j"] = resume([x["energie_run"]["puce"] for x in L])
        if "vision" in L[0]:
            c["vision_part_seule"] = resume([x["vision_part_seule"] for x in L])
            for k in ("ips", "latence_p50_ms", "latence_p99_ms", "latence_p99_max_tranches_ms", "manquees_par_s", "trou_max_s"):
                c[f"vision_{k}"] = resume([x["vision"][k] for x in L])
            c["vision_avant_ips"] = resume([x["vision_avant"]["ips"] if x["vision_avant"] else None for x in L])
        if "garde_30" in L[0]:
            c["garde_30_par_repetition"] = [x["garde_30"] for x in L]
            c["garde_30"] = all(c["garde_30_par_repetition"])
            G = [x for x in L if x["garde_30"]]        # estimée sur les répétitions qui gardent la cadence, avec leur nombre
            c["energie_planification_30_j"] = resume([x["energie_planification_30_j"] for x in G])
            c["energie_planification_30_n"] = len(G)
            c["energie_planification_30_bornes_j"] = ([min(x["energie_planification_30_bornes_j"][1] for x in G),
                                                       max(x["energie_planification_30_bornes_j"][0] for x in G)] if G else None)
        res[nom] = c

    def rapports(a, b, val):                           # rapport non défini si une valeur est nulle ou négative
        return [val(lignes[(a, k)]) / val(lignes[(b, k)]) if val(lignes[(a, k)]) > 0 and val(lignes[(b, k)]) > 0 else None for k in R]
    comparaisons = {}
    sans_charge = [n for n in attendus if "-vision" not in n]
    for a, b in paires_sans_charge(sans_charge):
        rt = rapports(a, b, lambda x: x["temps_par_plan_ms"])
        cmp = {"temps": {"rapports": rt, "verdict": decision(rt, MARGE_TEMPS)}}
        verdicts = []
        for cle, val in (("puce", lambda x: x["energie_solve"]["puce"]), ("repos_avant", lambda x: x["energie_solve_repos_avant"]),
                         ("repos_apres", lambda x: x["energie_solve_repos_apres"])):
            re = rapports(a, b, val)
            cmp["energie" + ("" if cle == "puce" else f"_{cle}")] = {"rapports": re, "verdict": decision(re, MARGE_ENERGIE)}
            verdicts.append(decision(re, MARGE_ENERGIE))
        cmp["energie"]["robuste_au_repos"] = len(set(verdicts)) == 1
        comparaisons[f"{a} contre {b}"] = cmp
    # question 2 : ralentissement du planificateur sous charge, part de la vision gardée, cadence de 30 images/s
    q2 = {}
    for mode in V.MODES_VISION:
        ral = {e: [lignes[(f"{e}-{mode}", k)]["temps_par_plan_ms"] / lignes[(e, k)]["temps_par_plan_ms"] for k in R]
               for e, _ in V.AVEC_CHARGE}
        part = {e: [lignes[(f"{e}-{mode}", k)]["vision_part_seule"] for k in R] for e, _ in V.AVEC_CHARGE}
        p99 = {e: [lignes[(f"{e}-{mode}", k)]["vision"]["latence_p99_ms"] for k in R] for e, _ in V.AVEC_CHARGE}
        q = {"ralentissement": {e: resume(v) for e, v in ral.items()}, "part_vision": {e: resume(v) for e, v in part.items()},
             "contrastes": {}}
        for e in ("coreml-ne", "ne-complet"):
            for ref in ("torch", "coreml-gpu"):
                # référence entièrement affamée (part 0) et ne-complet non : rapport infini, donc en faveur de ne-complet
                div = lambda a, b: (a / b if b > 0 else (math.inf if a > 0 else None)) if a is not None and b is not None and a >= 0 else None
                rr = [div(a, b) for a, b in zip(ral[e], ral[ref])]
                rv = [div(a, b) for a, b in zip(part[e], part[ref])]
                rp = [div(a, b) for a, b in zip(p99[e], p99[ref])]
                q["contrastes"][f"{e} contre {ref}"] = {
                    "ralentissement": {"rapports": rr, "verdict": decision(rr, MARGE_TEMPS)},
                    "part_vision": {"rapports": rv, "verdict": decision(rv, MARGE_TEMPS)},
                    "latence_p99": {"rapports": rp, "verdict": decision(rp, MARGE_TEMPS)}}
        if mode == "vision30":
            q["garde_30"] = {e: res[f"{e}-{mode}"]["garde_30"] for e, _ in V.AVEC_CHARGE}
        q2[mode] = q
    # réponse pré-enregistrée (amendements 5 et 6) : ne-complet contre torch (principal) et contre coreml-gpu (même
    # précision fp16, principal aussi) ; le bras à 30 images/s n'est interprétable que si la vision seule tient sa
    # cadence dans les 3 répétitions
    interp30 = all(garde_30(seules[("vision30", k)]["vision"]) for k in R)
    g = q2["vision30"]["garde_30"]
    q2_principal = {"interpretable_30": interp30}
    for ref in ("torch", "coreml-gpu"):
        k_ = q2["vision"]["contrastes"][f"ne-complet contre {ref}"]
        q2_principal[ref] = {"ralentissement": k_["ralentissement"], "part_vision": k_["part_vision"],
                             "garde_30": {"ne-complet": g["ne-complet"], ref: g[ref]},
                             "reponse": reponse_q2(k_["ralentissement"]["verdict"], k_["part_vision"]["verdict"],
                                                   g["ne-complet"], g[ref], interp30)}
    tt = [lignes[("torch", k)]["temps_par_plan_ms"] for k in R]
    tf = json.loads((r / "exp013-fin" / "fp32_s42.temps.json").read_text())["appels_solve"]
    t_fin = 1000 * sum(x for x, _ in tf) / sum(y for _, y in tf)
    bruit = max(tt) / min(tt) - 1
    if bruit > BRUIT_TORCH:
        signalements.append(f"runs torch écartés de {100 * bruit:.1f} % entre répétitions : règle du temps peu informative")
    det = determinisme(runs, attendus)
    signalements += [f"{n} : trajectoires différentes d'une répétition à l'autre" for n, d in det.items() if not d["identique_entre_repetitions"]]
    signalements += [f"{n} : trajectoires différentes du run sans vision" for n, d in det.items() if d["identique_sans_vision"] is False]
    autres = [(x["autre_processus_max"], n, k) for (n, k), x in runs.items() if x["autre_processus_max"] and x["autre_processus_max"][0] > 50]
    signalements += [f"{n} r{k} : processus {p[1]} à {p[0]:.0f} % de CPU pendant le run" for p, n, k in autres]
    # sensibilité à l'attente active d'OpenMP (ne-complet, coreml-cpu), repos d'après la répétition 3 et d'après la
    # sensibilité ; un moteur dont l'énergie par plan change de plus de 10 % est dit sensible à OpenMP (amendement 5)
    s_omp = {}
    for m in dict.fromkeys(x[0] for x in V.SENSIBILITE_OMP):
        s_omp[m] = {}
        for variante in ("defaut", "passif"):
            L = [par_run(sens[(m_, v, k)], repos[f"r{V.REPETITIONS}-apres"]["w"], repos["sens-apres"]["w"])
                 for m_, v, k in V.SENSIBILITE_OMP if m_ == m and v == variante]
            s_omp[m][variante] = {"temps_par_plan_ms": resume([x["temps_par_plan_ms"] for x in L]),
                                  "energie_solve_puce_j": resume([x["energie_solve"]["puce"] for x in L]),
                                  "cpu_w": resume([x["composants_solve_w"]["CPU"] for x in L])}
        ep, ed = s_omp[m]["passif"]["energie_solve_puce_j"], s_omp[m]["defaut"]["energie_solve_puce_j"]
        rap = ep["mediane"] / ed["mediane"] if ep["mediane"] > 0 and ed["mediane"] > 0 else None   # non calculable sinon
        s_omp[m]["rapport_energie_passif_defaut"] = rap
        s_omp[m]["rapport_temps_passif_defaut"] = s_omp[m]["passif"]["temps_par_plan_ms"]["mediane"] / s_omp[m]["defaut"]["temps_par_plan_ms"]["mediane"]
        disjoints = ep["min"] > ed["max"] or ep["max"] < ed["min"]       # les 2 runs de chaque variante ne se recouvrent pas
        s_omp[m]["sensible"] = rap is not None and disjoints and not (1 - MARGE_ENERGIE <= rap <= 1 + MARGE_ENERGIE)
    for nom_c, cmp in comparaisons.items():
        cmp["sensible_openmp"] = [m for m in s_omp if s_omp[m]["sensible"] and m in nom_c.split(" contre ")]
    # bruit de la référence : écart des 3 runs torch (temps, énergie), et des p99 de la vision seule
    et = [lignes[("torch", k)]["energie_solve"]["puce"] for k in R]
    bruit_ref = {"temps": bruit, "energie": max(et) / min(et) - 1 if min(et) > 0 else None}
    for nom_c, cmp in comparaisons.items():
        cmp["temps"]["peu_informatif"] = bruit > BRUIT_TORCH
    for mode, q in q2.items():                         # mêmes mentions pour la question 2 (amendement 5, points 9 et 14)
        for nom_c, cmp in q["contrastes"].items():
            cmp["ralentissement"]["peu_informatif"] = bruit > BRUIT_TORCH
            cmp["sensible_openmp"] = [m for m in s_omp if s_omp[m]["sensible"] and m in nom_c.split(" contre ")]
    for ref in ("torch", "coreml-gpu"):
        q2_principal[ref]["sensible_openmp"] = s_omp["ne-complet"]["sensible"]
        q2_principal[ref]["ralentissement"]["peu_informatif"] = bruit > BRUIT_TORCH
    # question 3 : planificateur entièrement hors du GPU, si la puissance du GPU au-dessus du repos reste sous 0,3 W
    gpu = [lignes[("ne-complet", k)]["composants_solve_w"]["GPU"] for k in R]
    g_repos = [x["w"].get("GPU Energy", 0) for x in repos.values()]
    q3 = {"gpu_w": gpu, "hors_gpu": all(g <= SEUIL_GPU_HORS for g in gpu), "gpu_repos_w": [min(g_repos), max(g_repos)],
          "construction": "encodeurs et prédicteur en Core ML sur le Neural Engine (plan de calcul), CEM et coût sur le CPU (--device cpu)"}
    bal = balayage(r, sha, essai)
    if bal is None:
        signalements.append("balayage du lot absent (échec du balayage) : le reste de l'analyse vaut")
    return {"controles": {"commit": sha, "essai": essai}, "configurations": res, "comparaisons": comparaisons,
            "question_2": q2, "question_2_principal": q2_principal, "question_3": q3, "bruit_reference": bruit_ref,
            "determinisme": det, "signalements": signalements, "sensibilite_omp": s_omp,
            "derive_torch_ms": tt + [t_fin], "vision_seule": {m: {"puissance_puce_w": resume([seules[(m, k)]["puissance_puce_w"] for k in R]),
                                                     "ips": resume([seules[(m, k)]["vision"]["ips"] for k in R]),
                                                     "latence_p99_ms": resume([seules[(m, k)]["vision"]["latence_p99_ms"] for k in R])}
                                                 for m in V.MODES_VISION},
            "repos_puce_w": {e: somme(x["w"], PUCE) for e, x in repos.items()},
            "reussite_ne_complet": reussite(r, sha, essai), "balayage_lot": bal}


def reussite(r, sha, essai=False):
    """Contrôle descriptif du planificateur sur le Neural Engine sur les épisodes d'EXP-003 : moyenne sur ses graines contre
    la pleine précision d'EXP-003 (5 graines), par épisode (mêmes épisodes ; tirages de la CEM différents, sur le CPU)."""
    m, graines = V.REUSSITE
    d = r / f"exp013-{m}-reussite"
    ref = dict(np.load(r / "exp003" / "fp32_s42.npz"))
    S = []
    for g in graines:
        x = dict(np.load(d / f"fp32_s{g}.npz"))
        A.exiger(str(x["git"]) == sha and not bool(x["git_sale"]), f"{d.name} graine {g} : autre commit ou arbre modifié")
        n = len(x["episodes_idx"])
        if essai:                                      # essai : 10 épisodes, pas un sous-ensemble d'EXP-003 ; pas de comparaison
            S.append(x["success"].astype(np.float64))
            continue
        for k in ("episodes_idx", "start_steps", "init_state", "goal_state"):
            A.exiger(np.array_equal(x[k], ref[k][:n]), f"{d.name} graine {g} : {k} différent d'EXP-003")
        A.controler_course(x, f"{d.name} graine {g}")
        S.append(x["success"].astype(np.float64))
    n = len(S[0])
    if essai:
        return {"n_episodes": n, "reussite": float(np.mean(S)), "reussite_fp32_exp003": None, "ecart": None}
    f = np.mean([np.load(r / "exp003" / f"fp32_s{g}.npz")["success"].astype(np.float64)[:n] for g in (42, 43, 44, 45, 46)], axis=0)
    return {"n_episodes": n, "reussite": float(np.mean(S)), "reussite_fp32_exp003": float(f.mean()),
            "ecart": A.Bootstrap(n=n).ic(np.mean(S, axis=0) - f)}


def balayage(r, sha, essai):
    """Balayage du lot : médianes par cas et répétition ; verdict de la prédiction au plus grand lot (aucun codage plus
    rapide que fp16 de plus de 5 % dans les 3 répétitions, sur l'un ou l'autre processeur)."""
    f = r / "exp013-balayage-lot.json"
    if not f.exists():                                 # échec du balayage : signalé, le reste de l'analyse vaut (amendement 5)
        return None
    b = json.loads(f.read_text())
    canaux_ok(b["repos_avant_w"], "balayage du lot, repos d'avant")
    canaux_ok(b["repos_apres_w"], "balayage du lot, repos d'après")
    for c in b["cas"]:
        for m in c["mesures"]:
            canaux_ok(m["joules"], f"balayage du lot, {c['unite']} {c['codage']} lot {c['lot']}")
    A.exiger(b["git"] == sha and b["essai"] == essai, "balayage du lot : autre commit ou autre mode")
    if not essai:
        A.exiger(b["lots"] == [1, 8, 32, 100, 300] and b["codages"] == list(C.CODAGES) and b["repetitions"] == 3,
                 "balayage du lot : plan différent du pré-enregistrement")
    A.exiger(len(b["cas"]) == 2 * len(b["lots"]) * len(b["codages"]) and all(len(c["mesures"]) == b["repetitions"] for c in b["cas"]),
             "balayage du lot incomplet")
    cas = {(c["unite"], c["codage"], c["lot"]): c for c in b["cas"]}
    tab = {f"{u} {cod} {B}": {"mediane_ms": resume([m["mediane_ms"] for m in c["mesures"]]),
                              "energie_nette_par_appel_puce_j": resume([somme(m["joules_nets_par_appel"], PUCE) for m in c["mesures"]]),
                              "placement": c["placement"], "octets_poids": c["octets_poids"]}
           for (u, cod, B), c in cas.items()}
    grand, verdicts = max(b["lots"]), {}
    for u in ("ne", "gpu"):
        ref = {m["repetition"]: m["mediane_ms"] for m in cas[(u, "fp16", grand)]["mesures"]}
        for cod in b["codages"]:
            if cod == "fp16":
                continue
            rr = [m["mediane_ms"] / ref[m["repetition"]] for m in sorted(cas[(u, cod, grand)]["mesures"], key=lambda m: m["repetition"])]
            verdicts[f"{u} {cod}"] = {"rapports": rr, "verdict": decision(rr, MARGE_TEMPS)}
    tenue = (None if any(v["verdict"] == "non calculable" for v in verdicts.values())          # non évaluable
             else all(v["verdict"] != "plus bas dans les 3 répétitions" for v in verdicts.values()))
    return {"lots": b["lots"], "codages": b["codages"], "repos_puce_w": somme(b["repos_w"], PUCE), "cas": tab,
            "verdicts_lot_max": verdicts, "prediction_tenue": tenue}


def rapport(r):
    f = lambda s, n=0, k=1: "" if s is None else f"{k * s['mediane']:.{n}f} ({k * s['min']:.{n}f} à {k * s['max']:.{n}f})"
    L = ["# EXP-013 : vitesse, énergie et mémoire de la planification (analyse pré-enregistrée, descriptive)", "",
         f"Contrôles bloquants passés, commit {r['controles']['commit'][:7]}" + (" (ESSAI, chemin court)" if r["controles"]["essai"] else "")
         + " ; runs torch identiques au pilote d'EXP-012. Médiane des 3 répétitions (min à max). Énergie de la puce : "
         "compteurs IOReport « Energy Model » (estimation de la puce, non validée contre une mesure externe ; affichage exclu), "
         "pendant la planification seule, au-dessus du repos (moyenne des repos avant et après la répétition). "
         "Règle de constance : « plus bas (ou plus haut) dans les 3 répétitions » si les 3 rapports appariés sont au-delà "
         "de 5 % (temps, latence, part de la vision) ou 10 % (énergie), du même côté ; critère descriptif, pas un test, "
         "une machine, une session.", "", "## Signalements", ""]
    L += [f"- {s}" for s in r["signalements"]] or ["- aucun"]
    L += ["", "## Sans charge", "",
          "| Configuration | Temps par plan (ms) | Énergie par plan, puce (J) | calcul (J) | mémoire (J) | brute (J) | Poids du prédicteur (Mo) | Placement | Part de l'appel Core ML | Réussite, 50 épisodes (IC 95 %) |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for nom, c in r["configurations"].items():
        if "-vision" in nom:
            continue
        mo = f"{c['octets_predicteur'] / 1e6:.1f}" if c["octets_predicteur"] else ""
        ic = c["succes_ic95"]
        L.append(f"| {nom} | {f(c['temps_par_plan_ms'])} | {f(c.get('energie_solve_puce_j'), 2)} | {f(c.get('energie_solve_calcul_j'), 2)} | "
                 f"{f(c.get('energie_solve_memoire_j'), 2)} | {f(c['energie_solve_brute_puce'], 2)} | {mo} | "
                 f"{', '.join(f'{k} {v}' for k, v in (c['placement'] or {}).items())} | {f(c['part_coreml'], 0, 100)} | "
                 f"{100 * c['succes']['mediane']:.0f} [{100 * ic[0]:.0f} ; {100 * ic[1]:.0f}] |")
    L += ["", "Codages int8 (linéaire symétrique par canal), int4b32 (blocs de 32, arrondi sans calibration), int4c (int4 par "
          "canal) et pal4 (une palette de 16 valeurs par tenseur, k-moyennes) : fonctions de coremltools 9.0 depuis les poids "
          "16 bits ; placement attendu, celui de l'essai de placement (int4 par blocs sur le Neural Engine : 211 opérations sur "
          "le CPU). Le moment de la décompression des poids n'est pas observé ici. Réussite non établie (50 épisodes de "
          "contrôle seulement) ; mesurée par une autre expérience. Le coût par plan ne dépend pas des valeurs des poids (CEM "
          "sans arrêt anticipé). Core ML complète les longueurs T = 1 et 2 à 3 (prédicteur de forme fixe).",
          "", "| Comparaison | Temps | Énergie de la puce |", "| --- | --- | --- |"]

    def fr(x):
        if not x:
            return ""
        return (f"{x['verdict']} ({', '.join('nd' if v is None else f'{v:.2f}' for v in x['rapports'])})"
                + ("" if x.get("robuste_au_repos", True) else " ; pas robuste au repos")
                + (" ; référence bruitée, peu informatif" if x.get("peu_informatif") else ""))
    for nom, c in r["comparaisons"].items():
        so = f" ; sensible à OpenMP ({', '.join(c['sensible_openmp'])})" if c.get("sensible_openmp") else ""
        L.append(f"| {nom} | {fr(c['temps'])} | {fr(c['energie'])}{so} |")
    L += ["", "## Avec charge de vision sur le GPU (mesures sur les fenêtres de planification)", "",
          "| Configuration | Temps par plan (ms) | Puissance de la puce pendant la planification (W) | Vision (images/s) | Part de la vision seule | Latence p50 (ms) | Latence p99 (ms) | Plus grand trou sans image (s) | Échéances manquées par s | Énergie marginale de planification, 30 images/s (J par plan, n) | Bornes (puissance de la vision seule min et max) |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for nom, c in r["configurations"].items():
        if "-vision" not in nom:
            continue
        e30 = ""
        if "garde_30" in c:
            e30 = (f"{f(c['energie_planification_30_j'], 2)}, n = {c['energie_planification_30_n']}" if c["energie_planification_30_n"]
                   else "non attribuable (30 images/s non gardées)")
        bo = c.get("energie_planification_30_bornes_j")
        L.append(f"| {nom} | {f(c['temps_par_plan_ms'])} | {f(c['puissance_solve_puce_w'], 1)} | {f(c['vision_ips'], 1)} | {f(c['vision_part_seule'], 2)} | "
                 f"{f(c['vision_latence_p50_ms'], 1)} | {f(c['vision_latence_p99_ms'], 1)} | {f(c['vision_trou_max_s'], 2)} | {f(c['vision_manquees_par_s'], 2)} | "
                 f"{e30} | {'' if not bo else f'{bo[0]:.2f} à {bo[1]:.2f}'} |")
    L += ["", "Vision seule : " + " ; ".join(f"{m} {f(v['ips'], 1)} images/s, p99 {f(v['latence_p99_ms'], 1)} ms, puce {f(v['puissance_puce_w'], 1)} W"
                                           for m, v in r["vision_seule"].items()),
          "", "### Question 2 : le planificateur hors du GPU garde-t-il son temps, et la vision sa cadence ?", "",
          "| Mode | Contraste | Ralentissement du planificateur sous charge | Part de la vision gardée | Latence p99 de la vision |",
          "| --- | --- | --- | --- | --- |"]
    for mode, q in r["question_2"].items():
        for nom, c in q["contrastes"].items():
            so = f" ; sensible à OpenMP ({', '.join(c['sensible_openmp'])})" if c.get("sensible_openmp") else ""
            L.append(f"| {mode} | {nom} | {fr(c['ralentissement'])}{so} | {fr(c['part_vision'])} | {fr(c['latence_p99'])} |")
    L += ["", "Ralentissement sous charge (temps avec charge / sans charge, même répétition) : " + " ; ".join(
        f"{mode} {e} {f(v, 2)}" for mode, q in r["question_2"].items() for e, v in q["ralentissement"].items()),
          "", "30 images par seconde gardées (au moins 29,7 images/s et au plus 0,3 échéance manquée par seconde, dans les 3 "
          "répétitions) : " + ", ".join(f"{e} {'oui' if v else 'non'}" for e, v in r["question_2"]["vision30"]["garde_30"].items())]
    qp = r["question_2_principal"]
    for ref in ("torch", "coreml-gpu"):
        q = qp[ref]
        L += ["", f"Réponse pré-enregistrée à la question 2, ne-complet contre {ref}"
              + (" (principal ; précision différente : torch en float32, ne-complet en float16)" if ref == "torch" else
                 " (principal, même précision float16)") + f" : **{q['reponse']}** (ralentissement : {fr(q['ralentissement'])} ; "
              f"part de la vision : {fr(q['part_vision'])} ; 30 images/s gardées : ne-complet {'oui' if q['garde_30']['ne-complet'] else 'non'}, "
              f"{ref} {'oui' if q['garde_30'][ref] else 'non'}" + ("" if qp["interpretable_30"] else
              " ; bras à 30 images/s non interprétable : la vision seule n'y tient pas sa cadence")
              + (" ; ne-complet sensible à OpenMP" if q.get("sensible_openmp") else "") + ")."]
    L += ["", "Les autres contrastes de la question 2 sont secondaires. Le contraste principal a été choisi après l'essai de "
          "bout en bout (10 épisodes, EXP-011 en même temps) : ce qu'il montrait est consigné dans l'amendement 6."]
    q3 = r["question_3"]
    L += ["", f"Question 3 : par construction, {q3['construction']} ; puissance du GPU au-dessus du repos pendant la "
          "planification de ne-complet : " + ", ".join(f"{g:.2f} W" for g in q3["gpu_w"])
          + f" (GPU au repos de {q3['gpu_repos_w'][0]:.2f} à {q3['gpu_repos_w'][1]:.2f} W selon la mesure au repos)"
          + (" ; « planificateur entièrement hors du GPU » permis (au plus 0,3 W dans les 3 répétitions)." if q3["hors_gpu"]
             else " ; la formulation est « GPU à X W », pas « entièrement hors du GPU ».")]
    for m, s in r["sensibilite_omp"].items():
        L += ["", f"Sensibilité à l'attente active d'OpenMP ({m}, 2 runs par variante) : par défaut {f(s['defaut']['temps_par_plan_ms'])} ms, "
              f"{f(s['defaut']['energie_solve_puce_j'], 2)} J, CPU {f(s['defaut']['cpu_w'], 1)} W ; OMP_WAIT_POLICY=PASSIVE "
              f"{f(s['passif']['temps_par_plan_ms'])} ms, {f(s['passif']['energie_solve_puce_j'], 2)} J, CPU {f(s['passif']['cpu_w'], 1)} W ; "
              f"rapport d'énergie {'non calculable' if s['rapport_energie_passif_defaut'] is None else format(s['rapport_energie_passif_defaut'], '.2f')}, "
              f"de temps {s['rapport_temps_passif_defaut']:.2f}"
              + (" : sensible à OpenMP." if s["sensible"] else ".")]
    br = r["bruit_reference"]
    L += ["", "Dérive : runs torch r1, r2, r3 et final : " + ", ".join(f"{v:.0f} ms" for v in r["derive_torch_ms"]) + " par plan ; écart "
          f"des 3 répétitions : temps {100 * br['temps']:.1f} %" + ("" if br["energie"] is None else f", énergie {100 * br['energie']:.1f} %") + "."]
    rn = r["reussite_ne_complet"]
    L += [] if rn["ecart"] is None else ["", f"Contrôle descriptif de réussite du planificateur sur le Neural Engine ({rn['n_episodes']} épisodes d'EXP-003, "
          f"2 graines) : {100 * rn['reussite']:.1f} % contre {100 * rn['reussite_fp32_exp003']:.1f} % en pleine précision PyTorch, "
          f"écart {A.ic_txt(rn['ecart'], 100, 1)} points (tirages de la CEM différents : sur le CPU ; aucune affirmation de "
          "non-infériorité)."]
    b = r["balayage_lot"]
    if b is None:
        return "\n".join(L + ["", "## Balayage de la taille du lot", "", "Absent (échec du balayage)."]) + "\n"
    L += ["", "## Balayage de la taille du lot (appels isolés du prédicteur, caches chauds, ms, médiane des répétitions)", "",
          "| Processeur | Codage | " + " | ".join(f"lot {x}" for x in b["lots"]) + " |", "| --- | --- | " + " | ".join("---" for _ in b["lots"]) + " |"]
    for u in ("ne", "gpu"):
        for cod in b["codages"]:
            L.append(f"| {u} | {cod} | " + " | ".join(f"{b['cas'][f'{u} {cod} {x}']['mediane_ms']['mediane']:.2f}" for x in b["lots"]) + " |")
    L += ["", f"Prédiction enregistrée (au lot de {max(b['lots'])}, aucun codage plus rapide que fp16 de plus de 5 % dans "
          f"les 3 répétitions) : {'non évaluable' if b['prediction_tenue'] is None else 'tenue' if b['prediction_tenue'] else 'réfutée'} ; " + " ; ".join(
              f"{k} {fr(v)}" for k, v in b["verdicts_lot_max"].items()) + ". Des temps par appel au lot de 300 avaient été vus "
          "avant le balayage (essai de placement) : c'est une observation, pas une prédiction confirmée. Cases informatives "
          "sur la question « un codage accélère-t-il ? » : Neural Engine int8, pal4 et int4c, GPU pal4 ; les autres cases "
          "mesurent aussi le coût propre de l'implémentation du codage dans Core ML, qu'on ne sépare pas ici."]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--racine", required=True)
    ap.add_argument("--sortie", required=True)
    ap.add_argument("--essai", action="store_true")
    a = ap.parse_args()
    r = analyser(a.racine, essai=a.essai)
    out = Path(a.sortie)
    out.mkdir(parents=True, exist_ok=True)
    (out / "resultats.json").write_text(json.dumps(A.pur(r), indent=1, ensure_ascii=False))
    (out / "rapport.md").write_text(rapport(r))
    print((out / "rapport.md").read_text())


if __name__ == "__main__":
    main()
