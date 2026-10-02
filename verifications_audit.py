"""Audit du 2 octobre 2026 : chiffres demandés par les trois audits des fiches (EXP-002 à EXP-014) et absents des sorties
des analyses. Tous descriptifs, aucun test nouveau : différences appariées par épisode (somme sur les graines / nombre de
graines), intervalle bootstrap de l'analyse d'EXP-003 (Bootstrap, graine 2, 10 000 rééchantillonnages des épisodes).
Chaque chiffre cité après l'audit vient de ce script ; les fichiers lus sont listés dans la sortie.

Usage (depuis code-exp015, environnement de le-wm) :
    python verifications_audit.py ../resultats/audit-2026-10-02
"""
import json
import sys
from pathlib import Path

import numpy as np

import analyse_exp003 as A

R = Path(__file__).resolve().parent.parent / "resultats"
G5 = (42, 43, 44, 45, 46)
G10 = (52, 53, 54, 55, 56)
lus = set()


def charge(d, v, graines=G5):
    X = []
    for g in graines:
        f = R / d / f"{v}_s{g}.npz"
        lus.add(str(f.relative_to(R)))
        X.append(np.load(f))
    ep = X[0]["episodes_idx"]
    assert all(np.array_equal(x["episodes_idx"], ep) for x in X), (d, v)
    return np.stack([x["success"].astype(int) for x in X]), ep          # (graines, épisodes)


def ecart(a, b, ep_a, ep_b):
    """a - b en points, apparié par épisode (somme sur les graines / nombre de graines)."""
    assert np.array_equal(ep_a, ep_b), "épisodes différents : pas d'appariement"
    x = (a.sum(0) - b.sum(0)) / a.shape[0]
    ic = A.Bootstrap(n=len(x)).ic(x)
    return {"moyenne": 100 * ic["moyenne"], "bas": 100 * ic["bas"], "haut": 100 * ic["haut"],
            "erreur_type": 100 * float(x.std(ddof=1) / np.sqrt(len(x))), "episodes": int(len(x)), "graines": int(a.shape[0])}


res = {}
# 1. essais répétés : épisodes que la pleine précision réussit 5 fois sur 5 et que la version ne réussit jamais
CAS = {"pusht": ("exp003", "exp009-pusht"), "tworoom": ("exp006", "exp009-tworoom")}
res["essais_repetes"] = {}
for t, (d0, d9) in CAS.items():
    f, ep = charge(d0, "fp32")
    toujours = f.sum(0) == 5
    r = {"episodes": int(f.shape[1]), "fp32_toujours": int(toujours.sum()),
         "echecs_fp32_sur_episodes_jamais_reussis": float((f.sum(0) == 0).sum() * 5 / max(1, (1 - f).sum()))}
    for nom, d, v in (("RTN 4", d0, "all-4"), ("GPTQ 4", d9, "all-4-gptq"), ("RTN 3", d0, "all-3"), ("GPTQ 3", d9, "all-3-gptq")):
        s, e = charge(d, v)
        assert np.array_equal(e, ep)
        n = int((toujours & (s.sum(0) == 0)).sum())
        r[nom] = {"casses": n, "part_de_tous": n / f.shape[1], "part_des_toujours_reussis": n / int(toujours.sum())}
    res["essais_repetes"][t] = r
# 2. GPTQ contre la pleine précision sur les 500 épisodes d'EXP-003 et EXP-006 (EXP-009), et écart à EXP-010
res["gptq_contre_fp32_500"] = {}
for t, (d0, d9) in CAS.items():
    f, ep = charge(d0, "fp32")
    for b in (4, 3):
        s, e = charge(d9, f"all-{b}-gptq")
        res["gptq_contre_fp32_500"][f"{t} {b} bits"] = ecart(s, f, e, ep)
f10, ep10 = charge("exp005", "fp32", G10)
res["exp010"] = {}
for b in (4, 3):
    s, e = charge("exp010", f"all-{b}-gptq", G10)
    res["exp010"][f"gptq {b} bits contre fp32"] = ecart(s, f10, e, ep10)
for b in (8, 6):
    s, e = charge("exp005", f"all-{b}", G10)
    res["exp010"][f"rtn {b} bits contre fp32"] = ecart(s, f10, e, ep10)
s4, e4 = charge("exp010", "all-4-gptq", G10)
s6, e6 = charge("exp005", "all-6", G10)
res["exp010"]["gptq 4 bits contre rtn 6 bits"] = ecart(s4, s6, e4, e6)
d1, d2 = res["gptq_contre_fp32_500"]["pusht 4 bits"], res["exp010"]["gptq 4 bits contre fp32"]
res["exp010"]["z_ecart_500_contre_2000_4_bits"] = (d1["moyenne"] - d2["moyenne"]) / np.hypot(d1["erreur_type"], d2["erreur_type"])
# 3. additivité à 3 bits, groupes de 64 à 3 bits sur TwoRoom, encpred contre tout à 4 bits sur TwoRoom
add = {}
for t, parts in (("pusht", (("exp003", "encoder-3"), ("exp003", "predictor-3"), ("exp004", "autres-3"))),
                 ("tworoom", (("exp008", "encoder-3"), ("exp008", "predictor-3"), ("exp008", "autres-3")))):
    d0 = "exp003" if t == "pusht" else "exp006"
    f, ep = charge(d0, "fp32")
    a3, e = charge(d0, "all-3")
    somme = sum(charge(d, v)[0].sum(0) - f.sum(0) for d, v in parts)
    x = (a3.sum(0) - f.sum(0) - somme) / 5
    ic = A.Bootstrap(n=len(x)).ic(x)
    add[t] = {"tout_moins_somme_des_parties": {k: 100 * ic[k] for k in ("moyenne", "bas", "haut")},
              "tout": ecart(a3, f, e, ep), **{v: ecart(charge(d, v)[0], f, e, ep) for d, v in parts}}
res["additivite_3_bits"] = add
f6, ep6 = charge("exp006", "fp32")
a3, e3 = charge("exp006", "all-3")
g3, eg = charge("exp008", "all-3-g64")
res["tworoom_3_bits_g64_contre_rtn"] = ecart(g3, a3, eg, e3)
a4, e4 = charge("exp006", "all-4")
ep4, ee = charge("exp008", "encpred-4")
res["tworoom_encpred4_moins_all4"] = ecart(ep4, a4, ee, e4)
res["tworoom_all4_contre_fp32"] = ecart(a4, f6, e4, ep6)
# 4. EXP-012 : énergie exploratoire par moteur (journal IOReport, TwoRoom), fichier par fichier
L = [json.loads(l) for l in (R / "energie-puce-2026-10-01.jsonl").read_text().splitlines() if l.strip()]
lus.add("energie-puce-2026-10-01.jsonl")
PUCE = ("CPU Energy", "GPU Energy", "ANE", "DRAM", "AMCC", "DCS", "FAB")
e12 = {}
for f in sorted(R.glob("exp012-*-tworoom/fp32_s*.temps.json")):
    t = json.loads(f.read_text())
    ts = [x["t"] for x in t["lectures"]]
    a, b = ts[0], ts[-1]
    d = [x for x in L if x["t0"] >= a and x["t1"] <= b]
    if not d:
        continue
    lus.add(str(f.relative_to(R)))
    duree = sum(x["t1"] - x["t0"] for x in d)
    w = sum(sum(x["joules"].get(k, 0) for k in PUCE) for x in d) / duree
    plans = sum(n for _, n in t["appels_solve"])
    e12[f"{f.parent.name}/{f.name}"] = {"puissance_w": w, "joules_par_plan_brut": w * (b - a) / plans, "couverture": duree / (b - a)}
res["exp012_energie_exploratoire"] = e12
# fichiers couverts à moins de 90 % par le journal (lancé en cours de fichier) : écartés du résumé
moy = lambda cle, cond: float(np.mean([v[cle] for k, v in e12.items() if cond(k) and v["couverture"] >= 0.9]))
ne_tous = moy("joules_par_plan_brut", lambda k: "exp012-ne" in k)
ne_sans43 = moy("joules_par_plan_brut", lambda k: "exp012-ne" in k and "s43" not in k)
gpu = moy("joules_par_plan_brut", lambda k: "exp012-gpu" in k)
res["exp012_energie_resume"] = {"ne_tous": ne_tous, "ne_graines_44_a_46": ne_sans43, "gpu": gpu,
                                "economie_ne_contre_gpu_tous": 1 - ne_tous / gpu, "economie_ne_contre_gpu_44_a_46": 1 - ne_sans43 / gpu,
                                "torch": moy("joules_par_plan_brut", lambda k: "exp012-torch" in k)}
res["fichiers_lus"] = sorted(lus)
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
(out / "resultats.json").write_text(json.dumps(A.pur(res), indent=1, ensure_ascii=False))
print(json.dumps({k: v for k, v in A.pur(res).items() if k not in ("fichiers_lus", "exp012_energie_exploratoire")}, indent=1, ensure_ascii=False))
