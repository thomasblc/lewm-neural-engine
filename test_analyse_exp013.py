"""Tests de analyse_exp013.py sur des runs fabriqués à valeurs connues (copies du vrai fichier pilote d'EXP-012 pour la
réussite et le contrôle de reproduction) : temps par plan, énergie par plan au-dessus du repos, images par seconde de la
vision, taille des poids, résumés sur 3 répétitions ; refus des données incohérentes, chacun avec son motif. À lancer
sur un arbre commité.

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
REPOS = {"CPU Energy": 2.0, "GPU Energy": 1.0, "ANE": 0.0, "DRAM": 0.5}          # W


def puissance(nom):                                    # W pendant le run, par composant
    gpu = 30.0 if "torch" in nom or "coreml-gpu" in nom else 1.0
    ane = 4.0 if "coreml-ne" in nom else 0.0
    return {"CPU Energy": 6.0, "GPU Energy": gpu + (20.0 if nom.endswith("-vision") else 0.0), "ANE": ane, "DRAM": 1.5}


def fabriquer(racine):
    r = Path(racine)
    (r / "pilote-exp012" / "torch").mkdir(parents=True)
    pil = RES / "pilote-exp012" / "torch" / "fp32_s42.npz"
    shutil.copy(pil, r / "pilote-exp012" / "torch" / "fp32_s42.npz")
    configs = [(m, c, False) for m, c in V.SANS_CHARGE] + [(m, c, True) for m, c in V.AVEC_CHARGE] + [("charge-seule", "", True)]
    noms = [V.nom(*c) for c in configs]
    (r / "exp013-ordre.json").write_text(json.dumps({"configs": noms, "ordres": [list(range(len(noms)))] * 3}))
    t = 1000.0
    for rep in (1, 2, 3):
        (r / f"exp013-repos-r{rep}-{int(t)}.json").write_text(json.dumps({
            "joules": {k: v * 75 for k, v in REPOS.items()}, "duree_s": 75.0, "calculs_run_grid_vus": 0, "git": TETE}))
        t += 200
        lignes_seule = [{"t0": t + 5 * i, "t1": t + 5 * (i + 1), "images": 500} for i in range(12)]   # 100 images/s
        (r / f"exp013-vision-seule-r{rep}.jsonl").write_text("\n".join(json.dumps(x) for x in lignes_seule) + "\n")
        t += 100
        for nom in noms:
            if nom == "charge-seule":
                continue
            d = r / f"exp013-{nom}-r{rep}"
            d.mkdir()
            x = dict(np.load(pil))
            x.update(git=np.str_(TETE), git_sale=np.bool_(False))
            np.savez_compressed(d / "fp32_s42.npz", **x)
            par_plan = (0.8 if nom.startswith("torch") else 0.4 if "coreml-gpu" in nom else 0.9) + 0.01 * rep
            appels = [[par_plan * 50, 50]] * 10 + [[par_plan * 12, 12]] * 10
            if rep == 2 and nom == "coreml-ne":
                appels[0] = [par_plan * 50 + 600, 50]     # pause de la garde : appel écarté
            plans = 620
            duree = 120.0
            j = {k: w * duree for k, w in puissance(nom).items()}
            coreml = {}
            if nom.startswith("coreml"):
                coreml = {"placement": {"NeuralEngine" if "coreml-ne" in nom else "GPU": 300},
                          "octets_poids": {"int8": 11e6, "int4b32": 6e6, "pal4": 5.5e6}.get(nom.split("-")[2] if nom.count("-") >= 2 and "vision" not in nom else "", 21.5e6)}
            (d / "fp32_s42.temps.json").write_text(json.dumps({
                "moteur": nom.replace("-vision", "").rsplit("-int", 1)[0].rsplit("-pal", 1)[0], "git": TETE, "appels_solve": appels,
                "plans": plans, "coreml": coreml, "energie_puce": {"joules": j, "duree_s": duree, "debut": t}}))
            if nom.endswith("-vision"):
                lv = [{"t0": t + 5 * i, "t1": t + 5 * (i + 1), "images": 300 if "coreml-ne" in nom else 200} for i in range(24)]
                (r / f"exp013-vision-{nom}-r{rep}.jsonl").write_text("\n".join(json.dumps(x) for x in lv) + "\n")
            t += 150
    (r / "exp013-fin").mkdir()
    x = dict(np.load(pil))
    x.update(git=np.str_(TETE), git_sale=np.bool_(False))
    np.savez_compressed(r / "exp013-fin" / "fp32_s42.npz", **x)
    return r


def test_valeurs(tmp):
    r = fabriquer(Path(tmp) / "a")
    res = A13.analyser(r, verifier=False)
    c = res["configurations"]
    assert abs(c["torch"]["temps_par_plan_ms"]["mediane"] - 820) < 1e-9 and abs(c["coreml-gpu"]["temps_par_plan_ms"]["min"] - 410) < 1e-9
    assert abs(c["coreml-ne"]["temps_par_plan_ms"]["mediane"] - 920) < 1e-9          # l'appel allongé est écarté
    # énergie au-dessus du repos : (P - repos) x 120 s / 620 plans
    attendu = sum((puissance("coreml-ne")[k] - REPOS[k]) * 120 / 620 for k in REPOS)
    assert abs(c["coreml-ne"]["energie_puce_par_plan_j"]["mediane"] - attendu) < 1e-9
    assert abs(c["coreml-ne"]["energie_Neural Engine_j"]["mediane"] - 4.0 * 120 / 620) < 1e-9
    assert abs(c["coreml-ne-vision"]["vision_ips"]["mediane"] - 60.0) < 1e-9 and abs(c["torch-vision"]["vision_ips"]["mediane"] - 40.0) < 1e-9
    assert abs(res["vision_seule_ips"]["mediane"] - 100.0) < 1e-9
    assert c["coreml-ne-int8"]["octets_poids"] == 11e6 and c["torch"]["vision_ips"] is None
    assert "Vision seule" in A13.rapport(res)
    print("ok valeurs : temps (appel allongé écarté), énergie au-dessus du repos, vision, poids")
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
            g = rr / chemin
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
    repos2 = lambda rr: sorted(rr.glob("exp013-repos-r2-*.json"))[0].relative_to(rr)
    cas = [("mesure au repos manquante", lambda rr: [f.unlink() for f in rr.glob("exp013-repos-r3-*.json")]),
           ("un calcul tournait", lambda rr: json_mod(repos2(rr), lambda j: j.update(calculs_run_grid_vus=1))(rr)),
           ("manquant", lambda rr: (rr / "exp013-coreml-ne-pal4-r2" / "fp32_s42.npz").unlink()),
           ("hors du processeur demandé", json_mod("exp013-coreml-ne-r1/fp32_s42.temps.json", lambda j: j["coreml"].update(placement={"NeuralEngine": 200, "CPU": 100}))),
           ("aucune mesure de la charge", lambda rr: (rr / "exp013-coreml-gpu-vision-r3.jsonl").unlink()),
           ("différent du pilote", champ("exp013-torch-r2", "success", lambda x: ~x["success"])),
           ("différent du pilote", champ("exp013-fin", "start_steps", lambda x: x["start_steps"] + 1)),
           ("plusieurs commits", lambda rr: (champ("exp013-torch-fp16-r1", "git", lambda x: np.str_("b" * 40))(rr),
                                             json_mod("exp013-torch-fp16-r1/fp32_s42.temps.json", lambda j: j.update(git="b" * 40))(rr))),
           ("fichier de temps", json_mod("exp013-coreml-gpu-int8-r3/fp32_s42.temps.json", lambda j: j.update(energie_puce=None))),
           ("arbre modifié", champ("exp013-coreml-ne-int8-r1", "git_sale", lambda x: np.bool_(True))),
           ("configurations différentes", json_mod("exp013-ordre.json", lambda j: j.update(configs=j["configs"][:5])))]
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
