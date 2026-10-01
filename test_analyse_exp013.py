"""Tests de analyse_exp013.py (amendement 2) sur des runs fabriqués à valeurs connues (copies du vrai fichier pilote
d'EXP-012, donc déterministes et identiques au pilote) : temps par plan, énergie de la puce par plan pendant la
planification au-dessus du repos (moyenne des repos avant et après, et sensibilité), vision (images par seconde,
latences, échéances), énergie de planification estimée à cadence fixe, règle de décision ; refus des données
incohérentes, chacun avec son motif. À lancer sur un arbre commité.

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
REPOS = {"CPU Energy": 2.0, "GPU Energy": 1.0, "ANE": 0.0, "DRAM": 0.5, "AMCC": 1.0, "DCS": 0.5, "FAB": 0.5, "DISP": 0.3}
TEMPS = {"torch": 0.80, "torch-fp16": 0.70, "coreml-gpu": 0.40, "coreml-ne": 0.90, "ne-complet": 0.85, "coreml-cpu": 2.0}


def moteur_de(nom):
    for m in ("torch-fp16", "torch", "coreml-gpu", "coreml-ne", "ne-complet", "coreml-cpu"):
        if nom.startswith(m):
            return m


def puissance(nom):
    m = moteur_de(nom)
    p = dict(REPOS)
    p["CPU Energy"] += 4.0
    p["GPU Energy"] += 25.0 if m in ("torch", "torch-fp16", "coreml-gpu") else 0.0
    p["ANE"] += 4.0 if m in ("coreml-ne", "ne-complet") else 0.0
    p["AMCC"] += 1.0
    if "vision" in nom:
        p["GPU Energy"] += 15.0
    return p


def fabriquer(racine):
    r = Path(racine)
    (r / "pilote-exp012" / "torch").mkdir(parents=True)
    pil = RES / "pilote-exp012" / "torch" / "fp32_s42.npz"
    shutil.copy(pil, r / "pilote-exp012" / "torch" / "fp32_s42.npz")
    configs = V.configurations()
    (r / "exp013-ordre.json").write_text(json.dumps({"configs": [V.nom(*c) for c in configs], "ordres": [list(range(len(configs)))] * 3}))
    t = 1000.0
    def repos(etiquette):
        nonlocal t
        (r / f"exp013-repos-{etiquette}-{int(t)}.json").write_text(json.dumps({"joules": {k: v * 60 for k, v in REPOS.items()},
            "duree_s": 60.0, "debut": t, "fin": t + 60, "calculs_run_grid_vus": 0, "git": TETE}))
        t += 100
    for rep in (1, 2, 3):
        repos("r1-avant" if rep == 1 else f"r{rep - 1}-r{rep}")
        for m_, c, mode in configs:
            nom = V.nom(m_, c, mode)
            if m_.startswith("seule"):
                ips = 100 if mode == "vision" else 30
                lv = [{"t0": t + 5 * i, "t1": t + 5 * (i + 1), "images": ips * 5, "latence_p50_ms": 9.0, "latence_p99_ms": 12.0,
                       "echeances_manquees": 0} for i in range(9)]
                (r / f"exp013-vision-seule-{mode}-r{rep}.jsonl").write_text("\n".join(json.dumps(x) for x in lv) + "\n")
                pv = dict(REPOS); pv["GPU Energy"] += 15.0 if mode == "vision" else 6.0
                (r / f"exp013-vision-seule-{mode}-r{rep}.json").write_text(json.dumps({"mode": mode, "debut": t + 10, "duree_s": 35.0,
                    "joules": {k: v * 35 for k, v in pv.items()}}))
                t += 60
                continue
            d = r / f"exp013-{nom}-r{rep}"
            d.mkdir()
            x = dict(np.load(pil))
            x.update(git=np.str_(TETE), git_sale=np.bool_(False))
            np.savez_compressed(d / "fp32_s42.npz", **x)
            pp = TEMPS[moteur_de(nom)] * (1 + 0.01 * rep) * (1.2 if "vision" in nom and moteur_de(nom) in ("torch", "coreml-gpu") else 1.0)
            appels = [[pp * 50, 50], [pp * 12, 12]]
            duree_solve, plans = pp * 62, 62
            duree_run = duree_solve + 20.0
            p = puissance(nom)
            coreml = {}
            m = moteur_de(nom)
            if m in A13.PLACEMENT:
                coreml = {"placement": {A13.PLACEMENT[m]: 300}, "octets_poids": 23.2e6}
            ep = {"joules": {k: v * duree_run for k, v in p.items()}, "duree_s": duree_run, "debut": t,
                  "joules_solve": {k: v * duree_solve for k, v in p.items()}, "cpu_s": 10.0, "maxrss_octets": 2e9,
                  "appels_predict": {"n": 9300 if coreml else 0, "coreml_s": 0.6 * duree_solve if coreml else 0.0, "total_s": 0.0},
                  "octets_modele_torch": None if coreml else 72.3e6}
            (d / "fp32_s42.temps.json").write_text(json.dumps({"moteur": m, "git": TETE, "appels_solve": appels, "plans": plans,
                "coreml": coreml, "energie_puce": ep, "lectures": [{"t": t + 5 * i} for i in range(int(duree_run // 5) + 1)]}))
            if mode:
                cible = 30 if mode == "vision30" else (60 if m in ("coreml-ne", "ne-complet") else 40)
                lv = [{"t0": t - 15 + 5 * i, "t1": t - 15 + 5 * (i + 1), "images": (cible if i >= 3 else 100 if mode == "vision" else 30) * 5,
                       "latence_p50_ms": 10.0, "latence_p99_ms": 20.0 if m in ("torch", "coreml-gpu") else 12.0,
                       "echeances_manquees": 5 if (mode == "vision30" and m in ("torch", "coreml-gpu") and i >= 3) else 0}
                      for i in range(int((duree_run + 15) // 5))]
                (r / f"exp013-vision-{nom}-r{rep}.jsonl").write_text("\n".join(json.dumps(x) for x in lv) + "\n")
            t += duree_run + 30
    repos("r3-apres")
    shutil.copytree(RES / "exp003", r / "exp003", ignore=shutil.ignore_patterns("*.log", "all-*", "encoder-*", "predictor-*"))
    (r / "exp013-ne-complet-reussite").mkdir()
    for g, src in ((42, "fp32_s47"), (43, "fp32_s48")):
        x = dict(np.load(r / "exp003" / f"{src}.npz"))
        x.update(git=np.str_(TETE), git_sale=np.bool_(False))
        np.savez_compressed(r / "exp013-ne-complet-reussite" / f"fp32_s{g}.npz", **x)
    (r / "exp013-fin").mkdir()
    x = dict(np.load(pil))
    x.update(git=np.str_(TETE), git_sale=np.bool_(False))
    np.savez_compressed(r / "exp013-fin" / "fp32_s42.npz", **x)
    (r / "exp013-balayage-lot.json").write_text(json.dumps([{"unite": u, "codage": c, "lot": b, "mediane_ms": 1.0 + b / 100}
                                                           for u in ("ne", "gpu") for c in ("fp16", "int8") for b in (1, 300)]))
    return r


def test_valeurs(tmp):
    r = fabriquer(Path(tmp) / "a")
    res = A13.analyser(r, verifier=False)
    c = res["configurations"]
    assert abs(c["torch"]["temps_par_plan_ms"]["mediane"] - 816) < 1e-9 and abs(c["coreml-gpu"]["temps_par_plan_ms"]["min"] - 404) < 1e-9
    # énergie pendant la planification, au-dessus du repos (repos avant = après = REPOS) : (P - repos) sur les canaux de la puce
    e_ne = sum(puissance("coreml-ne").get(k, 0) - REPOS.get(k, 0) for k in A13.SOC) * 0.90 * 1.02
    assert abs(c["coreml-ne"]["energie_solve_puce_j"]["mediane"] - e_ne) < 1e-9
    assert abs(c["coreml-ne"]["composants_solve_w"]["Neural Engine"]["mediane"] - 4.0) < 1e-9
    assert c["torch-vision"].get("energie_solve_puce_j") is None                          # pas d'énergie par plan avec charge
    assert abs(c["coreml-ne-vision"]["vision_ips"]["mediane"] - 60.0) < 1e-9 and abs(c["torch-vision"]["vision_avant_ips"]["mediane"] - 100.0) < 1e-9
    assert abs(c["torch-vision30"]["vision_manquees_par_s"]["mediane"] - 1.0) < 1e-9
    # cadence fixe : (E pendant la planification - puissance de la vision seule x durée) / plans
    pv = sum(v for v in REPOS.values()) + 6.0
    p_run = sum(puissance("coreml-ne-vision30").get(k, 0) for k in A13.SOC)
    assert abs(c["coreml-ne-vision30"]["energie_planification_30_j"]["mediane"] - (p_run - pv) * 0.90 * 1.02) < 1e-9
    assert abs(res["vision_seule"]["vision"]["ips"]["mediane"] - 100.0) < 1e-9
    cmp = res["comparaisons"]
    assert cmp["coreml-gpu"]["temps_contre_torch"]["verdict"] == "plus bas, établi"
    assert cmp["coreml-ne"]["temps_contre_torch"]["verdict"] == "plus haut, établi"
    assert cmp["coreml-ne"]["energie_contre_coreml-gpu"]["verdict"] in ("plus bas, établi", "plus haut, établi", "non établi")
    assert cmp["coreml-ne"]["energie_contre_coreml-gpu"]["robuste_au_repos"]
    assert c["torch"]["octets_poids"] == 72.3e6 and c["coreml-ne"]["octets_poids"] == 23.2e6
    rn = res["reussite_ne_complet"]
    f5 = np.mean([np.load(r / "exp003" / f"fp32_s{g}.npz")["success"].astype(float) for g in (42, 43, 44, 45, 46)], axis=0)
    n2 = np.mean([np.load(r / "exp003" / f"fp32_s{g}.npz")["success"].astype(float) for g in (47, 48)], axis=0)
    assert abs(rn["ecart"]["moyenne"] - (n2 - f5).mean()) < 1e-12
    txt = A13.rapport(res)
    assert "Balayage de la taille du lot" in txt and "Avec charge de vision" in txt
    print("ok valeurs : temps, énergie de la puce au-dessus du repos, vision et cadence fixe, décision, réussite")
    return r


def test_refus(tmp, r):
    def essai(motif, modifier):
        rr = Path(tmp) / "refus"
        shutil.rmtree(rr, ignore_errors=True)
        shutil.copytree(r, rr)
        modifier(rr)
        try:
            A13.analyser(rr, verifier=False)
        except A.ControleEchoue as e:
            assert motif in str(e), (motif, str(e))
            return
        raise AssertionError(f"accepté à tort : {motif}")

    def json_mod(chemin, f):
        def m(rr):
            g = rr / chemin if isinstance(chemin, str) else chemin(rr)
            j = json.loads(g.read_text())
            f(j)
            g.write_text(json.dumps(j))
        return m

    def champ(d, cle, val):
        def m(rr):
            x = dict(np.load(rr / d / "fp32_s42.npz"))
            x[cle] = val(x)
            np.savez_compressed(rr / d / "fp32_s42.npz", **x)
        return m
    repos = lambda e: (lambda rr: sorted(rr.glob(f"exp013-repos-{e}-*.json"))[0])
    tps = lambda d: f"{d}/fp32_s42.temps.json"
    cas = [("mesure au repos r2-r3 manquante", lambda rr: [f.unlink() for f in rr.glob("exp013-repos-r2-r3-*.json")]),
           ("un calcul tournait", json_mod(repos("r1-avant"), lambda j: j.update(calculs_run_grid_vus=1))),
           ("manquant", lambda rr: (rr / "exp013-coreml-ne-pal4-r2" / "fp32_s42.npz").unlink()),
           ("hors du processeur demandé", json_mod(tps("exp013-coreml-ne-r1"), lambda j: j["coreml"].update(placement={"NeuralEngine": 200, "CPU": 100}))),
           ("hors du processeur demandé", json_mod(tps("exp013-coreml-cpu-r2"), lambda j: j["coreml"].update(placement={"NeuralEngine": 5}))),
           ("aucune mesure de la charge", lambda rr: (rr / "exp013-vision-coreml-gpu-vision-r3.jsonl").unlink()),
           ("différent du pilote", lambda rr: [champ(f"exp013-{n}-r{k}", "success", lambda x: ~x["success"])(rr)
                                                for n in ("torch", "torch-vision", "torch-vision30") for k in (1, 2, 3)]),
           ("trajectoires différentes", champ("exp013-coreml-ne-vision-r2", "start_steps", lambda x: x["start_steps"] + 1)),
           ("plusieurs commits", lambda rr: (champ("exp013-torch-fp16-r1", "git", lambda x: np.str_("b" * 40))(rr),
                                             json_mod(tps("exp013-torch-fp16-r1"), lambda j: j.update(git="b" * 40))(rr))),
           ("fichier de temps", json_mod(tps("exp013-coreml-gpu-int8-r3"), lambda j: j.update(energie_puce=None))),
           ("arbre modifié", champ("exp013-coreml-ne-int8-r1", "git_sale", lambda x: np.bool_(True))),
           ("configurations différentes", json_mod("exp013-ordre.json", lambda j: j.update(configs=j["configs"][:5]))),
           ("pause de la garde", json_mod(tps("exp013-coreml-ne-r3"), lambda j: j.update(lectures=[{"t": 0}, {"t": 5}, {"t": 600}]))),
           ("Neural Engine presque inactif", json_mod(tps("exp013-ne-complet-r2"), lambda j: j["energie_puce"]["joules_solve"].update(ANE=0.0))),
           ("Neural Engine actif", json_mod(tps("exp013-coreml-gpu-r1"), lambda j: j["energie_puce"]["joules_solve"].update(ANE=500.0))),
           ("init_state différent d'EXP-003", lambda rr: (lambda f: np.savez_compressed(f, **{**dict(np.load(f)), "init_state": np.load(f)["init_state"] + 1}))(
               rr / "exp013-ne-complet-reussite" / "fp32_s43.npz"))]
    for motif, m in cas:
        essai(motif, m)
    print(f"ok contrôles bloquants ({len(cas)} refus, chacun avec son motif)")


def test_code():
    assert PROPRE, "arbre modifié : committer avant de lancer les tests"
    A13.verifier_code(TETE)
    print(f"ok code : fichiers d'analyse et de calcul identiques à {TETE[:7]}")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        r = test_valeurs(tmp)
        test_refus(tmp, r)
        test_code()
    print("tous les tests passent")
