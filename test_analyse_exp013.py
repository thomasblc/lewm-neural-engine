"""Tests de analyse_exp013.py (amendements 2 et 3) sur des runs fabriqués à valeurs connues (copies du vrai fichier pilote
d'EXP-012, donc déterministes et identiques au pilote), dans l'ordre tiré par vitesse.py : temps par plan, énergie de la
puce par plan au-dessus du repos, vision sur les fenêtres de planification (images par seconde, part de la vision seule,
latence p99 sur toutes les images, échéances manquées), cadence de 30 images par seconde gardée ou non, énergie de
planification à 30 images par seconde et sa sensibilité, contrastes de la question 2, sensibilité OpenMP, balayage du
lot ; signalements ; refus des données incohérentes, chacun avec son motif. À lancer sur un arbre commité.

Usage :
    python test_analyse_exp013.py
"""
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

import analyse_exp003 as A
import analyse_exp013 as A13
import vitesse as V

RES = A13.ICI.parent / "resultats"
TETE = subprocess.run(["git", "-C", str(A13.ICI), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
PROPRE = subprocess.run(["git", "-C", str(A13.ICI), "status", "--porcelain", "--untracked-files=no"],
                        capture_output=True, text=True).stdout.strip() == ""
REPOS = {k: 0.0 for k in A13.SOC}
REPOS.update({"CPU Energy": 2.0, "GPU Energy": 1.0, "DRAM": 0.5, "AMCC": 1.0, "DCS": 0.5, "FAB": 0.5, "DISP": 0.3})
TEMPS = {"torch": 0.80, "torch-fp16": 0.70, "coreml-gpu": 0.40, "coreml-ne": 0.90, "ne-complet": 0.85, "coreml-cpu": 2.0}
SUR_GPU = ("torch", "torch-fp16", "coreml-gpu")
NORMAL = ("Note: No thermal warning level has been recorded\nNote: No performance warning level has been recorded\n"
          "Note: No CPU power status has been recorded")


def moteur_de(nom):
    for m in ("torch-fp16", "torch", "coreml-gpu", "coreml-ne", "ne-complet", "coreml-cpu"):
        if nom.startswith(m):
            return m


def codage_de(nom):
    for c in ("int8", "int4b32", "pal4", "int4c"):
        if f"-{c}" in nom:
            return c
    return "fp16"


def puissance(nom):
    m = moteur_de(nom)
    p = dict(REPOS)
    p["CPU Energy"] += 4.0
    p["GPU Energy"] += 25.0 if m in SUR_GPU else 0.0
    p["ANE"] += 4.0 if m in ("coreml-ne", "ne-complet") else 0.0
    p["AMCC"] += 1.0
    p["DISP"] += 2.0                                       # l'affichage, hors de la puce comptée : sans effet
    if nom.endswith("-vision"):
        p["GPU Energy"] += 15.0
    elif nom.endswith("-vision30"):
        p["GPU Energy"] += 6.0
    return p


def lignes_vision(debut, fin, cadence, lat=10.0, retard_tous=0, cible=0.0):
    """Lignes de 5 s d'une charge de vision ; cadence(t) en images par seconde ; une image sur retard_tous en retard."""
    L, t, k = [{"pret": debut - 1.0, "ips_cible": cible}], debut, 0
    while t < fin:
        t1 = min(t + 5, fin)
        fins, x = [], t
        while x < t1:
            x += 1 / cadence(x)
            if x <= t1:
                fins.append(x)
        retard = [i for i in range(len(fins)) if retard_tous and (k + i) % retard_tous == 0]
        k += len(fins)
        L.append({"t0": t, "t1": t1, "images": len(fins), "latence_p50_ms": lat, "latence_p99_ms": lat, "echeances_manquees": len(retard),
                  "ips_cible": cible, "fins": fins, "lat_ms": [lat] * len(fins), "retard": retard})
        t = t1
    return L


def ecrire_jsonl(f, L):
    f.write_text("\n".join(json.dumps(x) for x in L) + "\n")


def ecrire_npz(f, x):
    x = dict(x)
    x.update(git=np.str_(TETE), git_sale=np.bool_(False))
    np.savez_compressed(f, **x)


def fabriquer(racine):
    r = Path(racine)
    (r / "pilote-exp012" / "torch").mkdir(parents=True)
    pil = RES / "pilote-exp012" / "torch" / "fp32_s42.npz"
    shutil.copy(pil, r / "pilote-exp012" / "torch" / "fp32_s42.npz")
    X = dict(np.load(pil))
    configs = V.configurations()
    ordres = V.ordres_contrebalances(len(configs), np.random.default_rng(13), configs)
    (r / "exp013-ordre.json").write_text(json.dumps({"configs": [V.nom(*c) for c in configs], "ordres": ordres, "git": TETE}))
    t = 1000.0

    def repos(etiquette):
        nonlocal t
        (r / f"exp013-repos-{etiquette}-{int(t)}.json").write_text(json.dumps({"joules": {k: v * 60 for k, v in REPOS.items()},
            "duree_s": 60.0, "debut": t, "fin": t + 60, "calculs_run_grid_vus": 0, "git": TETE}))
        t += 100

    def lecture(h):
        return {"t": h, "processus_run_grid": 1, "thermique": NORMAL, "CurrentCapacity": 80, "ExternalConnected": True,
                "processus": [[95.0, "python3.10"], [3.0, "WindowServer"]]}

    def run(d, nom, rep_facteur, env=None, x=None):
        nonlocal t
        d.mkdir()
        ecrire_npz(d / "fp32_s42.npz", X if x is None else x)
        m, c = moteur_de(nom), codage_de(nom)
        pp = TEMPS[m] * rep_facteur * (1.2 if nom.endswith(("-vision", "-vision30")) and m in SUR_GPU else 1.0)
        appels = [[pp * 50, 50], [pp * 12, 12]]
        fenetres = [[t, t + pp * 50], [t + pp * 50 + 1, t + pp * 62 + 1]]
        duree_solve, duree_run = pp * 62, pp * 62 + 20.0
        p = puissance(nom)
        coreml = {}
        if m in A13.PLACEMENT:
            coreml = {"placement": {A13.PLACEMENT[m]: 300}, "octets_poids": 23.2e6, "codage": c}
            if m in A13.HORS_GPU:
                coreml["octets_par_modele"] = {"predicteur": 23.2e6, "encodeur": 11.0e6, "actions": 0.1e6}
        ep = {"joules": {k: v * duree_run for k, v in p.items()}, "duree_s": duree_run, "debut": t,
              "joules_solve": {k: v * duree_solve for k, v in p.items()}, "cpu_s": 10.0, "maxrss_octets": 2e9,
              "appels_predict": {"n": 9300 if coreml else 0, "coreml_s": 0.6 * duree_solve if coreml else 0.0, "total_s": 0.0},
              "octets_modele_torch": None if coreml else 72.3e6,
              "octets_modules_torch": {"encoder": 22.0e6, "predictor": 46.0e6, "pred_proj": 0.4e6, "action_encoder": 0.2e6},
              "fenetres_solve": fenetres, "fils_torch": 6, "omp": env or {}, "priorite": 0}
        (d / "fp32_s42.temps.json").write_text(json.dumps({"moteur": m, "git": TETE, "appels_solve": appels, "plans": 62,
            "coreml": coreml, "energie_puce": ep, "lectures": [lecture(t + 5 * i) for i in range(int(duree_run // 5) + 1)]}))
        debut, fin = t, t + duree_run
        t += duree_run + 30
        return debut, fin, fenetres

    for rep in (1, 2, 3):
        repos("r1-avant" if rep == 1 else f"r{rep - 1}-r{rep}")
        for i in ordres[rep - 1]:
            m_, c, mode = configs[i]
            nom = V.nom(m_, c, mode)
            if m_.startswith("seule"):
                ips = 100.0 if mode == "vision" else 30.0
                debut = t + 13
                ecrire_jsonl(r / f"exp013-vision-seule-{mode}-r{rep}.jsonl",
                             lignes_vision(t + 3, t + 50, lambda x: ips, 9.0, cible=V.MODES_VISION[mode]))
                pv = dict(REPOS)
                pv["GPU Energy"] += 15.0 if mode == "vision" else 6.0
                (r / f"exp013-vision-seule-{mode}-r{rep}.json").write_text(json.dumps({"mode": mode, "ips_cible": V.MODES_VISION[mode],
                    "debut": debut, "fin": debut + 35, "duree_s": 35.0, "joules": {k: v * 35 for k, v in pv.items()}, "git": TETE}))
                t += 60
                continue
            m = moteur_de(nom)
            t0 = t
            t += 15                                    # la vision démarre avant le run
            debut, fin, fen = run(r / f"exp013-{nom}-r{rep}", nom, 1 + 0.01 * rep)
            if mode:
                cible = 30.0 if mode == "vision30" else (60.0 if m in ("coreml-ne", "ne-complet") else 40.0)
                hors = 100.0 if mode == "vision" else 30.0
                ecrire_jsonl(r / f"exp013-vision-{nom}-r{rep}.jsonl",
                             lignes_vision(t0 + 2, fin + 1, lambda x, d=debut: cible if x >= d else hors, 20.0 if m in SUR_GPU else 12.0,
                                           retard_tous=30 if (mode == "vision30" and m in SUR_GPU) else 0, cible=V.MODES_VISION[mode]))
    repos("r3-apres")
    for variante, k in V.SENSIBILITE_OMP:
        run(r / f"exp013-sens-ne-complet-omp-{variante}-{k}", "ne-complet", 1.0 if variante == "defaut" else 1.1,
            env={"OMP_WAIT_POLICY": "PASSIVE"} if variante == "passif" else None)
    repos("sens-apres")
    shutil.copytree(RES / "exp003", r / "exp003", ignore=shutil.ignore_patterns("*.log", "all-*", "encoder-*", "predictor-*"))
    (r / "exp013-ne-complet-reussite").mkdir()
    for g, src in ((42, "fp32_s47"), (43, "fp32_s48")):
        ecrire_npz(r / "exp013-ne-complet-reussite" / f"fp32_s{g}.npz", np.load(r / "exp003" / f"{src}.npz"))
    run(r / "exp013-fin", "torch", 1.04)
    cas = []
    for u in ("ne", "gpu"):
        for cod in ("fp16", "int8", "int4b32", "pal4", "int4c"):
            for b in (1, 8, 32, 100, 300):
                base = 1.0 + b / 100
                cas.append({"unite": u, "codage": cod, "lot": b, "placement": {"NeuralEngine" if u == "ne" else "GPU": 300},
                            "octets_poids": 23.2e6, "mesures": [{"repetition": k, "mediane_ms": base * (1 + 0.01 * k), "appels": 100,
                                                                 "joules_nets_par_appel": {"ANE": 0.01}} for k in (1, 2, 3)]})
    (r / "exp013-balayage-lot.json").write_text(json.dumps({"git": TETE, "essai": False, "lots": [1, 8, 32, 100, 300],
        "codages": ["fp16", "int8", "int4b32", "pal4", "int4c"], "repetitions": 3, "boucle_s": 3.0, "repos_w": REPOS, "cas": cas}))
    return r


def proche(a, b, tol=0.01):
    return abs(a - b) <= tol * abs(b)


def test_valeurs(tmp):
    r = fabriquer(Path(tmp) / "a")
    res = A13.analyser(r, verifier=False)
    c = res["configurations"]
    assert proche(c["torch"]["temps_par_plan_ms"]["mediane"], 816, 1e-12) and proche(c["coreml-gpu"]["temps_par_plan_ms"]["min"], 404, 1e-12)
    e_ne = sum(puissance("coreml-ne")[k] - REPOS[k] for k in A13.PUCE) * 0.90 * 1.02     # 9 W au-dessus du repos
    assert proche(c["coreml-ne"]["energie_solve_puce_j"]["mediane"], e_ne, 1e-12)
    assert proche(c["coreml-ne"]["composants_solve_w"]["Neural Engine"]["mediane"], 4.0, 1e-12)
    assert c["torch-vision"].get("energie_solve_puce_j") is None                          # pas d'énergie par plan avec charge
    assert proche(c["coreml-ne-vision"]["vision_ips"]["mediane"], 60.0) and proche(c["torch-vision"]["vision_part_seule"]["mediane"], 0.40)
    assert c["torch-vision"]["vision_latence_p99_ms"]["mediane"] == 20.0 and c["ne-complet-vision30"]["vision_latence_p99_ms"]["mediane"] == 12.0
    assert proche(c["torch-vision30"]["vision_manquees_par_s"]["mediane"], 1.0, 0.03)
    assert c["torch-vision30"]["garde_30"] is False and c["torch-vision30"]["energie_planification_30_j"] is None
    assert c["ne-complet-vision30"]["garde_30"] is True
    e_30 = 9.0 * 0.90 * 1.02                         # coreml-ne : puissance au-dessus de la vision seule = 9 W
    assert proche(c["coreml-ne-vision30"]["energie_planification_30_j"]["mediane"], e_30, 1e-9)
    assert proche(c["coreml-ne-vision30"]["energie_planification_30_images_j"]["mediane"], e_30, 0.02)
    assert proche(res["vision_seule"]["vision"]["ips"]["mediane"], 100.0) and proche(res["vision_seule"]["vision30"]["ips"]["mediane"], 30.0)
    cmp = res["comparaisons"]
    assert cmp["coreml-gpu contre torch"]["temps"]["verdict"] == "plus bas dans les 3 répétitions"
    assert cmp["coreml-ne contre torch"]["temps"]["verdict"] == "plus haut dans les 3 répétitions"
    assert cmp["coreml-ne contre coreml-gpu"]["energie"]["verdict"] == "plus bas dans les 3 répétitions"
    assert cmp["coreml-ne contre coreml-gpu"]["energie"]["robuste_au_repos"]
    assert cmp["coreml-gpu-int8 contre coreml-gpu"]["temps"]["verdict"] == "non constant"
    assert "ne-complet contre coreml-cpu" in cmp and "coreml-gpu contre torch-fp16" in cmp and "coreml-ne-int4b32 contre coreml-ne-int4c" in cmp
    q = res["question_2"]
    assert proche(q["vision"]["ralentissement"]["torch"]["mediane"], 1.2, 1e-9) and proche(q["vision"]["ralentissement"]["ne-complet"]["mediane"], 1.0, 1e-9)
    k = q["vision"]["contrastes"]["ne-complet contre torch"]
    assert k["ralentissement"]["verdict"] == "plus bas dans les 3 répétitions" and k["part_vision"]["verdict"] == "plus haut dans les 3 répétitions"
    assert k["latence_p99"]["verdict"] == "plus bas dans les 3 répétitions"
    assert q["vision30"]["garde_30"] == {"torch": False, "coreml-gpu": False, "coreml-ne": True, "ne-complet": True}
    assert c["torch"]["octets_predicteur"] == 46.4e6 and c["coreml-ne"]["octets_predicteur"] == 23.2e6
    assert proche(res["sensibilite_omp"]["passif"]["temps_par_plan_ms"]["mediane"], 935, 1e-9)
    assert all(proche(a, b, 1e-12) for a, b in zip(res["derive_torch_ms"], [808, 816, 824, 832]))
    assert res["balayage_lot"]["prediction_tenue"] and res["signalements"] == [], res["signalements"]
    rn = res["reussite_ne_complet"]
    f5 = np.mean([np.load(r / "exp003" / f"fp32_s{g}.npz")["success"].astype(float) for g in (42, 43, 44, 45, 46)], axis=0)
    n2 = np.mean([np.load(r / "exp003" / f"fp32_s{g}.npz")["success"].astype(float) for g in (47, 48)], axis=0)
    assert abs(rn["ecart"]["moyenne"] - (n2 - f5).mean()) < 1e-12
    txt = A13.rapport(res)
    assert "Balayage de la taille du lot" in txt and "Question 2" in txt and "non attribuable" in txt and "tenue" in txt
    print("ok valeurs : temps, énergie au-dessus du repos, vision sur les fenêtres, 30 images/s, question 2, sensibilité, balayage")
    return r


def copie(tmp, r, nom):
    rr = Path(tmp) / nom
    shutil.rmtree(rr, ignore_errors=True)
    shutil.copytree(r, rr)
    return rr


def json_mod(chemin, f):
    def m(rr):
        g = rr / chemin if isinstance(chemin, str) else chemin(rr)
        j = json.loads(g.read_text())
        f(j)
        g.write_text(json.dumps(j))
    return m


def jsonl_mod(chemin, f):
    def m(rr):
        g = rr / chemin
        L = [json.loads(l) for l in g.read_text().splitlines() if l.strip()]
        ecrire_jsonl(g, f(L))
    return m


def champ(d, cle, val):
    def m(rr):
        x = dict(np.load(rr / d / "fp32_s42.npz"))
        x[cle] = val(x)
        np.savez_compressed(rr / d / "fp32_s42.npz", **x)
    return m


def test_signalements(tmp, r):
    def essai(attendu, *mods):
        rr = copie(tmp, r, "signal")
        for m in mods:
            m(rr)
        s = A13.analyser(rr, verifier=False)["signalements"]
        assert any(attendu in x for x in s), (attendu, s)
    tps = lambda d: f"{d}/fp32_s42.temps.json"
    essai("trajectoires différentes d'une répétition à l'autre", champ("exp013-coreml-ne-vision-r2", "start_steps", lambda x: x["start_steps"] + 1))
    essai("trajectoires différentes du run sans vision", *[champ(f"exp013-coreml-ne-vision-r{k}", "start_steps", lambda x: x["start_steps"] + 1)
                                                            for k in (1, 2, 3)])
    essai("GPU à", json_mod(tps("exp013-ne-complet-r2"), lambda j: j["energie_puce"]["joules_solve"].update(
        {"GPU Energy": j["energie_puce"]["joules_solve"]["GPU Energy"] + 0.5 * sum(a for a, _ in j["appels_solve"])})))
    essai("alerte thermique", json_mod(tps("exp013-coreml-gpu-r3"), lambda j: j["lectures"][2].update(thermique="CPU_Speed_Limit = 80")))
    essai("règle du temps peu informative", json_mod(tps("exp013-torch-r3"), lambda j: j.update(
        appels_solve=[[a * 1.05, n] for a, n in j["appels_solve"]])))
    essai("processus", json_mod(tps("exp013-coreml-ne-r1"), lambda j: j["lectures"][1].update(processus=[[90.0, "mds_stores"]])))
    # codage : placement en partie sur le CPU accepté et rapporté, Neural Engine faible signalé (amendement 4)
    rr = copie(tmp, r, "codage")
    for k in (1, 2, 3):
        json_mod(tps(f"exp013-coreml-ne-int4b32-r{k}"), lambda j: (j["coreml"].update(placement={"CPU": 211, "NeuralEngine": 150}),
                 j["energie_puce"]["joules_solve"].update(ANE=0.5 * sum(a for a, _ in j["appels_solve"]))))(rr)
    res = A13.analyser(rr, verifier=False)
    assert res["configurations"]["coreml-ne-int4b32"]["placement"] == {"CPU": 211, "NeuralEngine": 150}
    assert any("coreml-ne-int4b32 r1 : Neural Engine à" in x for x in res["signalements"]), res["signalements"]
    print("ok signalements (déterminisme, GPU hors planification, thermique, bruit, processus de fond)")


def test_refus(tmp, r):
    def essai(motif, modifier):
        rr = copie(tmp, r, "refus")
        modifier(rr)
        try:
            A13.analyser(rr, verifier=False)
        except A.ControleEchoue as e:
            assert motif in str(e), (motif, str(e))
            return
        raise AssertionError(f"accepté à tort : {motif}")

    repos = lambda e: (lambda rr: sorted(rr.glob(f"exp013-repos-{e}-*.json"))[0])
    tps = lambda d: f"{d}/fp32_s42.temps.json"
    configs = V.configurations()
    o1 = V.ordres_contrebalances(len(configs), np.random.default_rng(13), configs)[0]
    i = next(k for k in range(len(o1) - 1) if not configs[o1[k]][0].startswith("seule") and not configs[o1[k + 1]][0].startswith("seule"))
    a_, b_ = (V.nom(*configs[o1[i]]), V.nom(*configs[o1[i + 1]]))

    def permuter(rr):
        fa, fb = rr / tps(f"exp013-{a_}-r1"), rr / tps(f"exp013-{b_}-r1")
        ja, jb = json.loads(fa.read_text()), json.loads(fb.read_text())
        ja["energie_puce"]["debut"], jb["energie_puce"]["debut"] = jb["energie_puce"]["debut"], ja["energie_puce"]["debut"]
        fa.write_text(json.dumps(ja))
        fb.write_text(json.dumps(jb))

    def trou(L):
        for x in L[3:5]:
            if "fins" in x:
                x.update(fins=[], lat_ms=[], retard=[], images=0)
        return L
    cas = [("mesure au repos r2-r3 manquante", lambda rr: [f.unlink() for f in rr.glob("exp013-repos-r2-r3-*.json")]),
           ("un calcul tournait", json_mod(repos("r1-avant"), lambda j: j.update(calculs_run_grid_vus=1))),
           ("faite avant la fin des runs", json_mod(repos("r1-r2"), lambda j: j.update(debut=j["debut"] - 200, fin=j["fin"] - 200))),
           ("canal IOReport manquant", json_mod(repos("sens-apres"), lambda j: j["joules"].pop("DRAM"))),
           ("manquant", lambda rr: (rr / "exp013-coreml-ne-pal4-r2" / "fp32_s42.npz").unlink()),
           ("hors du processeur demandé", json_mod(tps("exp013-coreml-ne-r1"), lambda j: j["coreml"].update(placement={"NeuralEngine": 200, "CPU": 100}))),
           ("hors du processeur demandé", json_mod(tps("exp013-coreml-cpu-r2"), lambda j: j["coreml"].update(placement={"NeuralEngine": 5}))),
           ("moteur ou codage différents", json_mod(tps("exp013-coreml-ne-int8-r1"), lambda j: j["coreml"].update(codage="int4b32"))),
           ("charge de vision absente", lambda rr: (rr / "exp013-vision-coreml-gpu-vision-r3.jsonl").unlink()),
           ("pas prête avant le run", jsonl_mod("exp013-vision-torch-vision30-r2.jsonl", lambda L: [dict(L[0], pret=1e12)] + L[1:])),
           ("autre cadence", jsonl_mod("exp013-vision-ne-complet-vision30-r1.jsonl", lambda L: [dict(L[0], ips_cible=0.0)] + L[1:])),
           ("ne couvre pas les fenêtres", jsonl_mod("exp013-vision-coreml-ne-vision-r2.jsonl", trou)),
           ("autre commit ou autre cadence", json_mod("exp013-vision-seule-vision30-r2.json", lambda j: j.update(git="c" * 40))),
           ("vision interrompue", jsonl_mod("exp013-vision-seule-vision-r1.jsonl", trou)),
           ("différent du pilote", lambda rr: [champ(f"exp013-{n}-r{k}", "success", lambda x: ~x["success"])(rr)
                                                for n in ("torch", "torch-vision", "torch-vision30") for k in (1, 2, 3)]),
           ("plusieurs commits", lambda rr: (champ("exp013-torch-fp16-r1", "git", lambda x: np.str_("b" * 40))(rr),
                                             json_mod(tps("exp013-torch-fp16-r1"), lambda j: j.update(git="b" * 40))(rr))),
           ("fichier de temps", json_mod(tps("exp013-coreml-gpu-int8-r3"), lambda j: j.update(energie_puce=None))),
           ("fenêtres de planification manquantes", json_mod(tps("exp013-coreml-gpu-r2"), lambda j: j["energie_puce"].update(fenetres_solve=[]))),
           ("canal IOReport manquant", json_mod(tps("exp013-torch-r2"), lambda j: j["energie_puce"]["joules_solve"].update(ANE=float("nan")))),
           ("arbre modifié", champ("exp013-coreml-ne-int8-r1", "git_sale", lambda x: np.bool_(True))),
           ("configurations différentes", json_mod("exp013-ordre.json", lambda j: j.update(configs=j["configs"][:5]))),
           ("ordre différent de l'ordre tiré", json_mod("exp013-ordre.json", lambda j: j["ordres"].reverse())),
           ("dans un autre ordre", permuter),
           ("pause pendant le run", json_mod(tps("exp013-coreml-ne-r3"), lambda j: j.update(lectures=[dict(j["lectures"][0], t=0), dict(j["lectures"][0], t=5), dict(j["lectures"][0], t=13)]))),
           ("autre calcul run_grid", json_mod(tps("exp013-torch-r1"), lambda j: j["lectures"][1].update(processus_run_grid=2))),
           ("Neural Engine presque inactif", json_mod(tps("exp013-ne-complet-r2"), lambda j: j["energie_puce"]["joules_solve"].update(ANE=0.0))),
           ("Neural Engine actif", json_mod(tps("exp013-coreml-gpu-r1"), lambda j: j["energie_puce"]["joules_solve"].update(ANE=500.0))),
           ("OMP_WAIT_POLICY inattendue", json_mod(tps("exp013-sens-ne-complet-omp-passif-2"), lambda j: j["energie_puce"].update(omp={}))),
           ("run torch final manquant", lambda rr: shutil.rmtree(rr / "exp013-fin")),
           ("balayage du lot incomplet", json_mod("exp013-balayage-lot.json", lambda j: j["cas"].pop())),
           ("init_state différent d'EXP-003", lambda rr: (lambda f: np.savez_compressed(f, **{**dict(np.load(f)), "init_state": np.load(f)["init_state"] + 1}))(
               rr / "exp013-ne-complet-reussite" / "fp32_s43.npz"))]
    for motif, m in cas:
        essai(motif, m)
    rr = copie(tmp, r, "balayage")
    json_mod("exp013-balayage-lot.json", lambda j: [m.update(mediane_ms=m["mediane_ms"] * 0.9) for c in j["cas"]
                                                    if c["unite"] == "gpu" and c["codage"] == "int8" and c["lot"] == 300 for m in c["mesures"]])(rr)
    assert not A13.analyser(rr, verifier=False)["balayage_lot"]["prediction_tenue"]
    print(f"ok contrôles bloquants ({len(cas)} refus, chacun avec son motif) ; prédiction du balayage réfutable")


def test_code():
    assert PROPRE, "arbre modifié : committer avant de lancer les tests"
    A13.verifier_code(TETE)
    print(f"ok code : fichiers d'analyse et de calcul identiques à {TETE[:7]}")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        r = test_valeurs(tmp)
        test_signalements(tmp, r)
        test_refus(tmp, r)
        test_code()
    print("tous les tests passent")
