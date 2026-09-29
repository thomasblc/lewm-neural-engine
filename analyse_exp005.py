"""EXP-005 : analyse pré-enregistrée (PREREG-EXP005.md), figée avec lui.

Contrôle les 10 fichiers d'EXP-005 (fp32 et all:6, graines 42 à 46, 1 000 nouveaux épisodes) et la sanité
de clôture, puis teste H6 (réussite équivalente à ± 2 points et pose du T dégradée à 6 bits) et calcule les
descriptifs. Les outils statistiques et les contrôles par course sont ceux d'analyse_exp003.py.

Usage :
    python analyse_exp005.py --grille ../resultats/exp005 --sanite-fin ../resultats/exp005-sanite-fin \\
        --exclure ../resultats/exp003/meta.json ../resultats/sanite-n50/meta.json \\
        --exp003 ../resultats/analyse-exp003/resultats.json --sortie ../resultats/analyse-exp005
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

import analyse_exp003 as A

N5 = 1000
VERSIONS5 = ("fp32", "all:6")
GRAINES = A.GA
GRAINE_EPISODES = 2026
RUN_GRID_SHA256 = "31e665344fbc5a312a7714d7252bde9eae2c931e65e31cdfc1b2f3c5af513fcc"
QUANT_SHA256 = "2c450c5b06ebd6753e77ff9c189fffc339b655070efaf32e103f13ad5e29c65d"
EFFECTIFS5 = (653, 120)
COUCHES = {"fp32": 0, "all:6": 109}
CLES_SANITE = ("state", "block_pose", "agent_pos", "contacts", "success", "first_success", "n_steps")


def git(*args, texte=True):
    return subprocess.run(["git", "-C", str(A.ICI), *args], capture_output=True, text=texte)


def verifier_code5(sha):
    A.exiger(git("cat-file", "-e", f"{sha}^{{commit}}").returncode == 0, f"commit {sha[:7]} inconnu du dépôt")
    for f, empreinte in (("run_grid.py", RUN_GRID_SHA256), ("quant.py", QUANT_SHA256)):
        src = git("show", f"{sha}:{f}", texte=False)
        A.exiger(src.returncode == 0 and hashlib.sha256(src.stdout).hexdigest() == empreinte,
                 f"le commit {sha[:7]} n'a pas le {f} d'EXP-005")


def lire_exclus(chemins):
    exclus = set()
    for c in chemins:
        m = json.loads(Path(c).read_text())
        m = m.get("protocole", m)
        exclus |= set(zip(m["episodes_idx"], m["start_steps"]))
    return exclus


def charger5(dossier, exclus):
    dossier = Path(dossier)
    proto = json.loads((dossier / "meta.json").read_text())["protocole"]
    A.exiger(proto["n"] == N5, f"meta.json : {proto['n']} épisodes au lieu de {N5}")
    A.exiger(proto.get("graine_episodes") == GRAINE_EPISODES, "graine de tirage des épisodes différente")
    A.exiger({tuple(x) for x in proto.get("exclus", [])} == exclus, "liste d'exclusion différente")
    paires = list(zip(proto["episodes_idx"], proto["start_steps"]))
    A.exiger(len(set(paires)) == N5 and not (set(paires) & exclus), "épisodes en commun avec EXP-003 ou EXP-002")
    res = {}
    for f in sorted(dossier.glob("*.npz")):
        if ".tmp." in f.name:
            continue
        d = dict(np.load(f))
        cle = (str(d["spec"]), int(d["seed"]))
        A.exiger(cle not in res, f"{f.name} : doublon de {cle}")
        res[cle] = d
    attendus = {(v, g) for v in VERSIONS5 for g in GRAINES}
    A.exiger(set(res) == attendus, f"EXP-005 : fichiers manquants {sorted(attendus - set(res))}, "
                                   f"en trop {sorted(set(res) - attendus)}")
    ref = res[("fp32", GRAINES[0])]
    shas = set()
    for (v, g), d in res.items():
        nom = f"{v} graine {g}"
        A.exiger(not bool(d["git_sale"]), f"{nom} : arbre git modifié au lancement")
        shas.add(str(d["git"]))
        A.exiger(np.array_equal(d["positions"], np.arange(N5)), f"{nom} : positions")
        A.exiger(np.array_equal(d["episodes_idx"], proto["episodes_idx"]) and
                 np.array_equal(d["start_steps"], proto["start_steps"]), f"{nom} : épisodes différents de meta.json")
        for k in ("init_state", "goal_state"):
            A.exiger(np.array_equal(d[k], ref[k]), f"{nom} : {k} différent")
        A.exiger(int(d["chunk"]) == proto["chunk"], f"{nom} : taille des paquets")
        A.exiger(int(d["n_couches"]) == COUCHES[v], f"{nom} : {int(d['n_couches'])} couches quantifiées")
        A.exiger(d["state"].shape == (A.T, N5, 7), f"{nom} : forme de l'état")
        ch = int(d["chunk"])
        for c, pas in enumerate(np.asarray(d["n_steps"]).tolist()):
            sl = slice(c * ch, (c + 1) * ch)
            A.exiger(pas == A.T or (d["success"][sl].all() and int(d["first_success"][sl].max()) == pas - 1),
                     f"{nom} : paquet {c} arrêté à {pas} pas")
    for sha in sorted(shas):
        verifier_code5(sha)
    return res, sorted(shas)


def verifier_sanite_fin(dossier_grille, dossier_fin):
    m1 = json.loads((Path(dossier_grille) / "meta.json").read_text())
    m2 = json.loads((Path(dossier_fin) / "meta.json").read_text())
    A.exiger(json.dumps(m1["protocole"], sort_keys=True, default=str) == json.dumps(m2["protocole"], sort_keys=True, default=str)
             and m1.get("torch") == m2.get("torch"), "sanité de clôture : protocole ou torch différent")
    a, b = dict(np.load(Path(dossier_fin) / "all-6_s42.npz")), dict(np.load(Path(dossier_grille) / "all-6_s42.npz"))
    A.exiger(not bool(a["git_sale"]), "sanité de clôture : arbre git modifié au lancement")
    A.exiger((str(a["spec"]), int(a["seed"])) == ("all:6", 42), "sanité de clôture : pas la même version")
    verifier_code5(str(a["git"]))
    for k in CLES_SANITE:
        A.exiger(np.array_equal(a[k], b[k], equal_nan=a[k].dtype.kind == "f"), f"sanité de clôture : {k} diffère")


def table(res, v):
    S, M2, M3, divergences, limites = [], [], [], 0, 0
    for g in GRAINES:
        d = res[(v, g)]
        m, mT, lim, div = A.controler_course(d, f"{v} graine {g}")
        limites += lim
        divergences += div
        S.append(d["success"].astype(np.int64))
        M2.append(m)
        M3.append(mT)
    S, M3 = np.stack(S), np.stack(M3)
    return {"S": S, "K": S.sum(axis=0), "M2": np.stack(M2), "M3": M3,
            "classe": np.where(S == 1, 0, np.where(M3 < 1, 1, 2)), "divergences": divergences, "limites": limites}


def analyser(grille, sanite_fin, exclure, exp003=None, effectifs=EFFECTIFS5):
    exclus = lire_exclus(exclure)
    res, shas = charger5(grille, exclus)
    verifier_sanite_fin(grille, sanite_fin)
    dep = A.depart(res)
    if effectifs is not None:
        A.exiger((int(dep["bouge"].sum()), int(dep["au_but"].sum())) == effectifs,
                 f"strates de départ inattendues : {int(dep['bouge'].sum())}, {int(dep['au_but'].sum())}")
    f, q = table(res, "fp32"), table(res, "all:6")
    bouge = dep["bouge"]
    inv = A.Inversion(n=N5)
    boot, boot_strate = A.Bootstrap(n=N5), A.Bootstrap(n=int(bouge.sum()), graine=3)

    x = q["K"] - f["K"]
    w = (A.lg(q["M3"]).mean(axis=0) - A.lg(f["M3"]).mean(axis=0))[bouge]
    p_bas, p_haut = inv.p(10 * x + 1, "plus"), inv.p(10 * x - 1, "moins")
    p_b = inv.p(w, "plus")
    p = max(p_bas, p_haut, p_b)
    h6 = {"delta_reussite": float(x.mean() / 5), "p_bas": p_bas, "p_haut": p_haut,
          "delta_log_M3_bouge": float(w.mean()), "episodes_bouge": int(w.size), "p_b": p_b, "p": p,
          "verdict": "établie" if p < A.ALPHA else "non établie", "controle_tost_t": A.tost_t(x / 5)}

    d3 = A.lg(q["M3"]).mean(axis=0) - A.lg(f["M3"]).mean(axis=0)
    d2 = A.lg(q["M2"]).mean(axis=0) - A.lg(f["M2"]).mean(axis=0)
    paires = [(a, b) for a in range(5) for b in range(5) if a != b]
    discord = np.mean([f["S"][a] != f["S"][b] for a, b in paires], axis=0) * 50
    desc = {v: {"reussite": float(t["S"].mean()), "classes": A.parts(t["classe"]),
                "classes_bouge": A.parts(t["classe"][:, bouge])} for v, t in (("fp32", f), ("all:6", q))}
    desc["ecarts"] = {"reussite": boot.ic(x / 5), "reussite_bouge": boot_strate.ic((x / 5)[bouge]),
                      "log_M3": boot.ic(d3), "log_M3_bouge": boot_strate.ic(d3[bouge]),
                      "log_M2": boot.ic(d2)}
    desc["plancher_discordants_par_50"] = boot.ic(discord)
    comparaison = None
    if exp003 is not None:
        r3 = json.loads(Path(exp003).read_text())["descriptifs"]["all:6"]
        comparaison = {"exp003_reussite": r3["delta_reussite"], "exp003_bouge": r3["bouge"]}
    return {"controles": {"commits": shas, "episodes_bouge": int(bouge.sum()), "episodes_T_au_but": int(dep["au_but"].sum()),
                          "divergences": {"fp32": f["divergences"], "all:6": q["divergences"]},
                          "cas_limites_arrondi": {"fp32": f["limites"], "all:6": q["limites"]}},
            "H6": h6, "descriptifs": desc, "comparaison_exp003": comparaison}


def rapport(r):
    h, c, e = r["H6"], r["controles"], r["descriptifs"]["ecarts"]
    L = ["# EXP-005 : analyse pré-enregistrée", "",
         f"Contrôles passés : 10 fichiers (commits {', '.join(s[:7] for s in c['commits'])}), sanité de clôture identique, "
         f"{c['episodes_bouge']} épisodes où le T doit bouger, {c['episodes_T_au_but']} où il part au but, "
         f"courses divergentes : {c['divergences']['fp32'] + c['divergences']['all:6']}.", "",
         "## H6 : à 6 bits, réussite équivalente à ± 2 points et pose du T dégradée", "",
         "| Partie | Estimation | p |", "| --- | --- | --- |",
         f"| (a) réussite, 1 000 épisodes | {100 * h['delta_reussite']:+.2f} points | bas {h['p_bas']:.4f}, haut {h['p_haut']:.4f} |",
         f"| (b) log M3, {h['episodes_bouge']} épisodes où le T doit bouger | {h['delta_log_M3_bouge']:+.3f} | {h['p_b']:.4f} |",
         f"| H6 (intersection-union) | | {h['p']:.4f} : **{h['verdict']}** |", "",
         f"Contrôle TOST t : p = {h['controle_tost_t']['p']:.4f}, intervalle à 90 % "
         f"[{100 * h['controle_tost_t']['ic90'][0]:+.2f}, {100 * h['controle_tost_t']['ic90'][1]:+.2f}] points.", "",
         "## Descriptif (intervalle bootstrap à 95 %)", "",
         f"- Réussite : fp32 {100 * r['descriptifs']['fp32']['reussite']:.1f} %, all:6 {100 * r['descriptifs']['all:6']['reussite']:.1f} %.",
         f"- Écart de réussite : {A.ic_txt(e['reussite'])} points ; dans la strate {A.ic_txt(e['reussite_bouge'])}.",
         f"- Écart de log M3 : {A.ic_txt(e['log_M3'], 1, 3)} ; dans la strate {A.ic_txt(e['log_M3_bouge'], 1, 3)}.",
         f"- Écart de log M2 : {A.ic_txt(e['log_M2'], 1, 3)}.",
         f"- Plancher de bruit : {A.ic_txt(r['descriptifs']['plancher_discordants_par_50'], 1, 2, '')} épisodes discordants sur 50."]
    for v in ("fp32", "all:6"):
        cl = r["descriptifs"][v]["classes_bouge"]
        L.append(f"- Classes dans la strate, {v} : " + " / ".join(f"{100 * cl[k]:.1f}" for k in A.CLASSES)
                 + " (réussite / T placé seul / T jamais placé).")
    if r["comparaison_exp003"]:
        k = r["comparaison_exp003"]
        L += ["", "## À côté d'EXP-003 (sans regroupement)", "",
              f"- EXP-003, 6 bits : écart de réussite {A.ic_txt(k['exp003_reussite'])} points ; dans la strate, "
              f"réussite {A.ic_txt(k['exp003_bouge']['delta_reussite'])}, log M3 {A.ic_txt(k['exp003_bouge']['delta_log_M3'], 1, 3)}."]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grille", required=True)
    ap.add_argument("--sanite-fin", required=True)
    ap.add_argument("--exclure", nargs="+", required=True)
    ap.add_argument("--exp003")
    ap.add_argument("--sortie", required=True)
    a = ap.parse_args()
    r = analyser(a.grille, a.sanite_fin, a.exclure, a.exp003)
    out = Path(a.sortie)
    out.mkdir(parents=True, exist_ok=True)
    (out / "resultats.json").write_text(json.dumps(A.pur(r), indent=1, ensure_ascii=False))
    (out / "rapport.md").write_text(rapport(r))
    print((out / "rapport.md").read_text())


if __name__ == "__main__":
    main()
