"""Analyse pré-enregistrée d'EXP-006 (PREREG-EXP006.md) : les résultats principaux d'EXP-003 et d'EXP-004 refaits
sur TwoRoom. Contrôles bloquants, 4 tests appariés par inversion de signe (Holm), descriptifs.

Usage :
    python analyse_exp006.py --grille ../resultats/exp006 --sanite-fin ../resultats/exp006-sanite-fin \
        --sortie ../resultats/analyse-exp006
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

import analyse_exp003 as A

N6 = 500
T6 = 50
SEUIL = 16.0                                  # TwoRoom : succès si la distance de l'agent au but passe sous 16
GA, GB = (42, 43, 44, 45, 46), (47, 48, 49, 50, 51)
VERSIONS6 = ("fp32", "all:8", "all:6", "all:4", "all:3", "encoder:4", "predictor:4", "autres:4", "all:4:g64")
COUCHES = {"fp32": 0, "all": 109, "encoder": 73, "predictor": 30, "autres": 6}
TESTS = (                                     # nom, version, référence, signe attendu de la somme des écarts
    ("R1", "all:4", "fp32", -1),              # tout le modèle à 4 bits fait baisser la réussite (P1 d'EXP-003)
    ("R2", "encoder:4", "predictor:4", -1),   # l'encodeur est plus sensible que le prédicteur (P2)
    ("R3", "autres:4", "fp32", -1),           # les 6 petites couches seules à 4 bits font baisser la réussite (Q3a)
    ("R4", "all:4:g64", "all:4", +1),         # les groupes de 64 relèvent la réussite à 4 bits (Q5a)
)
FICHIERS_ANALYSE = ("analyse_exp006.py", "analyse_exp003.py", "PREREG-EXP006.md", "run_grid.py", "quant.py")
ICI = Path(__file__).resolve().parent


def git(*args):
    return subprocess.run(["git", "-C", str(ICI), *args], capture_output=True, text=True)


def verifier_code(sha):
    A.exiger(git("cat-file", "-e", f"{sha}^{{commit}}").returncode == 0, f"commit {sha[:7]} inconnu du dépôt")
    A.exiger(git("diff", "--quiet", sha, "--", *FICHIERS_ANALYSE).returncode == 0,
             f"code d'analyse ou de calcul différent de celui du commit {sha[:7]}")


def controler(d, nom):
    """Contrôles d'une course (version, graine) ; renvoie la marge M = min distance / 16 (sous 1 : succès)."""
    s, k = d["success"].astype(bool), d["first_success"].astype(np.int64)
    A.exiger(np.array_equal(s, k >= 0) and np.all(k < T6), f"{nom} : succès et pas du premier succès incohérents")
    etat = d["state"].astype(np.float64)
    A.exiger(etat.shape == (T6, N6, 2), f"{nom} : forme de l'état {etat.shape}")
    nan = np.isnan(etat).any(axis=2)
    t = np.arange(T6)[:, None]
    attendu = s[None, :] & (t > k[None, :])                     # gelé après le pas du succès, fini avant
    A.exiger(np.array_equal(nan, attendu), f"{nom} : NaN ailleurs qu'après un succès")
    dist = np.linalg.norm(etat - d["goal_state"].astype(np.float64)[None], axis=-1)
    m = np.nanmin(dist, axis=0) / SEUIL
    A.exiger(np.isfinite(m).all() and (m >= 0).all(), f"{nom} : marge non finie")
    desaccord = (m < 1) != s
    limites = desaccord & (np.abs(m - 1) <= 1e-4)                # arrondi float32 de l'environnement
    A.exiger(not (desaccord & ~limites).any(), f"{nom} : distance et succès diffèrent sur {int((desaccord & ~limites).sum())} courses")
    if s.any():
        dk = dist[k[s], np.where(s)[0]]
        A.exiger((dk < SEUIL + 1e-3).all(), f"{nom} : distance au pas du succès au-dessus du seuil")
    return m, int(limites.sum())


def charger(dossier):
    dossier = Path(dossier)
    meta = json.loads((dossier / "meta.json").read_text())
    proto = meta["protocole"]
    A.exiger(proto.get("tache") == "tworoom" and proto["n"] == N6, "meta.json : pas le protocole TwoRoom de 500 épisodes")
    A.exiger(not meta["git_sale"], "arbre git modifié au lancement")
    res = {}
    for f in sorted(dossier.glob("*.npz")):
        if ".tmp." in f.name:
            continue
        d = dict(np.load(f))
        cle = (str(d["spec"]), int(d["seed"]))
        A.exiger(cle not in res, f"{f.name} : doublon")
        res[cle] = d
    attendus = {(v, g) for v in VERSIONS6 for g in GA} | {("fp32", g) for g in GB}
    A.exiger(set(res) == attendus, f"fichiers manquants {sorted(attendus - set(res))}, en trop {sorted(set(res) - attendus)}")
    ref, shas = res[("fp32", GA[0])], set()
    for (v, g), d in res.items():
        nom = f"{v} graine {g}"
        A.exiger(not bool(d["git_sale"]), f"{nom} : arbre git modifié")
        shas.add(str(d["git"]))
        A.exiger(np.array_equal(d["episodes_idx"], proto["episodes_idx"]) and
                 np.array_equal(d["start_steps"], proto["start_steps"]), f"{nom} : épisodes différents de meta.json")
        for k in ("init_state", "goal_state"):
            A.exiger(np.array_equal(d[k], ref[k]), f"{nom} : {k} différent")
        A.exiger(int(d["n_couches"]) == COUCHES[v.split(":")[0]], f"{nom} : {int(d['n_couches'])} couches quantifiées")
        ch = int(d["chunk"])
        for c, pas in enumerate(np.asarray(d["n_steps"]).tolist()):
            sl = slice(c * ch, (c + 1) * ch)
            A.exiger(pas == T6 or (d["success"][sl].all() and int(d["first_success"][sl].max()) == pas - 1),
                     f"{nom} : paquet {c} arrêté à {pas} pas")
    A.exiger(len(shas) == 1 and meta["git"] in shas, f"fichiers de plusieurs commits : {sorted(shas)}")
    return res, shas.pop()


def verifier_sanite(dossier, fin):
    a, b = dict(np.load(Path(fin) / "all-4_s42.npz")), dict(np.load(Path(dossier) / "all-4_s42.npz"))
    A.exiger(set(a) == set(b), "sanité de clôture : clés différentes")
    for k in sorted(set(a) - {"git", "git_sale"}):
        A.exiger(np.array_equal(a[k], b[k], equal_nan=a[k].dtype.kind == "f"), f"sanité de clôture : {k} diffère")


def analyser(dossier, fin, verifier=True):
    res, sha = charger(dossier)
    if verifier:
        verifier_code(sha)
    verifier_sanite(dossier, fin)
    K, M, limites = {}, {}, 0
    for v in VERSIONS6:
        S, Mv = [], []
        for g in GA:
            m, lim = controler(res[(v, g)], f"{v} graine {g}")
            S.append(res[(v, g)]["success"].astype(np.int64))
            Mv.append(m)
            limites += lim
        K[v], M[v] = np.stack(S).sum(axis=0), np.stack(Mv)
    rep = np.stack([res[("fp32", g)]["success"].astype(np.int64) for g in GB])
    for g in GB:
        limites += controler(res[("fp32", g)], f"fp32 graine {g}")[1]
    inv, boot = A.Inversion(n=N6), A.Bootstrap(n=N6)
    tests = {}
    for nom, v, ref, signe in TESTS:
        x = K[v] - K[ref]
        tests[nom] = {"version": v, "reference": ref, "somme": int(x.sum()), "ecart_points": boot.ic(x / 5),
                      "p": inv.p(x, "plus" if signe > 0 else "moins")}
    for (nom, *_), q in zip(TESTS, A.holm([t["p"] for t in tests.values()])):
        tests[nom]["p_holm"] = float(q)
        tests[nom]["verdict"] = "soutenue" if q < A.ALPHA else "non soutenue"
    lg = {v: np.log(np.maximum(M[v], 0.05)).mean(axis=0) for v in VERSIONS6}
    desc = {v: {"reussite": float(K[v].mean() / 5),
                "ecart_points": None if v == "fp32" else boot.ic((K[v] - K["fp32"]) / 5),
                "ecart_log_marge": None if v == "fp32" else boot.ic(lg[v] - lg["fp32"])} for v in VERSIONS6}
    desc["replique_fp32"] = {"reussite": float(rep.mean()), "ecart_points": boot.ic((rep.sum(axis=0) - K["fp32"]) / 5)}
    Sf = np.stack([res[("fp32", g)]["success"] for g in GA])
    paires = [(a, b) for a in range(5) for b in range(5) if a != b]
    d0 = res[("fp32", GA[0])]
    desc["episodes_partant_sous_16"] = int((np.linalg.norm(d0["init_state"] - d0["goal_state"], axis=-1) < SEUIL).sum())
    desc["plancher_discordants_par_50"] = boot.ic(np.mean([Sf[a] != Sf[b] for a, b in paires], axis=0) * 50)
    additivite = boot.ic((K["all:4"] - K["fp32"] - (K["encoder:4"] - K["fp32"]) - (K["predictor:4"] - K["fp32"])
                          - (K["autres:4"] - K["fp32"])) / 5)
    return {"controles": {"commit": sha, "fichiers": len(res), "cas_limites_arrondi": limites},
            "tests": tests, "descriptifs": desc, "additivite_4_bits": additivite}


def rapport(r):
    t = lambda e, f=100, d=1: A.ic_txt(e, f, d)
    L = ["# EXP-006 : analyse pré-enregistrée (TwoRoom)", "",
         f"Contrôles passés : {r['controles']['fichiers']} fichiers, commit {r['controles']['commit'][:7]}.", "",
         "| Test | Comparaison | Écart (points) | p | p Holm | Verdict |", "| --- | --- | --- | --- | --- | --- |"]
    for nom, x in r["tests"].items():
        L.append(f"| {nom} | {x['version']} contre {x['reference']} | {t(x['ecart_points'])} | {x['p']:.4f} | "
                 f"{x['p_holm']:.4f} | {x['verdict']} |")
    L += ["", "| Version | Réussite | Écart à fp32 (points) | Écart de log marge |", "| --- | --- | --- | --- |"]
    for v in VERSIONS6:
        d = r["descriptifs"][v]
        L.append(f"| {v} | {100 * d['reussite']:.1f} % | {t(d['ecart_points']) if d['ecart_points'] else ''} | "
                 f"{t(d['ecart_log_marge'], 1, 3) if d['ecart_log_marge'] else ''} |")
    rp = r["descriptifs"]["replique_fp32"]
    L += ["", f"Réplique fp32 (graines 47 à 51) : {100 * rp['reussite']:.1f} %, écart {t(rp['ecart_points'])} points ; "
          f"plancher de bruit : {t(r['descriptifs']['plancher_discordants_par_50'], 1, 2)} épisodes discordants sur 50 ; "
          f"écart à l'additivité à 4 bits : {t(r['additivite_4_bits'])} points ; épisodes partant déjà à moins de 16 du but : "
          f"{r['descriptifs']['episodes_partant_sous_16']}."]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grille", required=True)
    ap.add_argument("--sanite-fin", required=True)
    ap.add_argument("--sortie", required=True)
    a = ap.parse_args()
    r = analyser(a.grille, a.sanite_fin)
    out = Path(a.sortie)
    out.mkdir(parents=True, exist_ok=True)
    (out / "resultats.json").write_text(json.dumps(A.pur(r), indent=1, ensure_ascii=False))
    (out / "rapport.md").write_text(rapport(r))
    print((out / "rapport.md").read_text())


if __name__ == "__main__":
    main()
