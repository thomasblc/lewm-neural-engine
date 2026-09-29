"""EXP-005 : analyse pré-enregistrée (PREREG-EXP005.md), figée avec lui.

Contrôle les 15 fichiers d'EXP-005 (fp32, all:6 et le contrôle all:8, graines 52 à 56, 2 000 nouveaux
épisodes pris dans 2 000 trajectoires distinctes) et la sanité de clôture, vérifie que le code d'analyse est
celui du commit des données, puis teste H6 (réussite à ± 2 points et pose du T dégradée à 6 bits) et calcule
les descriptifs, dont le contrôle à 8 bits. Les outils statistiques et les contrôles par course sont ceux
d'analyse_exp003.py.

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
from metriques import ecart_angle, prolonger

N5 = 2000
VERSIONS5 = ("fp32", "all:6", "all:8")
GRAINES = (52, 53, 54, 55, 56)      # graines neuves : ni EXP-003 ni EXP-004 ne les ont utilisées
GRAINE_EPISODES = 2026
N_EXCLUS = 550                      # couples d'EXP-003 (500) et d'EXP-002 (50)
REGLE_TIRAGE = "trajectoires exclues entières, une position par trajectoire"
FICHIERS_ANALYSE = ("analyse_exp005.py", "analyse_exp003.py", "metriques.py", "PREREG-EXP005.md")
RUN_GRID_SHA256 = "f37d7218edea918f71bc3e44fe8019576f624597bced2954882c4aea21a74697"
QUANT_SHA256 = "2c450c5b06ebd6753e77ff9c189fffc339b655070efaf32e103f13ad5e29c65d"
EFFECTIFS5 = (1333, 233)
COUCHES = {"fp32": 0, "all:6": 109, "all:8": 109}
CLES_SANITE = ("state", "block_pose", "agent_pos", "contacts", "success", "first_success", "n_steps")


def git(*args, texte=True):
    return subprocess.run(["git", "-C", str(A.ICI), *args], capture_output=True, text=texte)


def verifier_code5(sha):
    A.exiger(git("cat-file", "-e", f"{sha}^{{commit}}").returncode == 0, f"commit {sha[:7]} inconnu du dépôt")
    for f, empreinte in (("run_grid.py", RUN_GRID_SHA256), ("quant.py", QUANT_SHA256)):
        src = git("show", f"{sha}:{f}", texte=False)
        A.exiger(src.returncode == 0 and hashlib.sha256(src.stdout).hexdigest() == empreinte,
                 f"le commit {sha[:7]} n'a pas le {f} d'EXP-005")


def verifier_analyse(sha):
    """Le code d'analyse et le pré-enregistrement du dossier courant sont ceux du commit des données."""
    A.exiger(git("diff", "--quiet", sha, "--", *FICHIERS_ANALYSE).returncode == 0,
             f"le code d'analyse diffère de celui du commit {sha[:7]} des données")


def lire_exclus(chemins):
    exclus = set()
    for c in chemins:
        m = json.loads(Path(c).read_text())
        m = m.get("protocole", m)
        exclus |= set(zip(m["episodes_idx"], m["start_steps"]))
    A.exiger(len(exclus) == N_EXCLUS, f"liste d'exclusion : {len(exclus)} couples au lieu de {N_EXCLUS}")
    return exclus


def charger5(dossier, exclus):
    dossier = Path(dossier)
    proto = json.loads((dossier / "meta.json").read_text())["protocole"]
    A.exiger(proto["n"] == N5, f"meta.json : {proto['n']} épisodes au lieu de {N5}")
    A.exiger(proto.get("graine_episodes") == GRAINE_EPISODES, "graine de tirage des épisodes différente")
    A.exiger({tuple(x) for x in proto.get("exclus", [])} == exclus, "liste d'exclusion différente")
    A.exiger(proto.get("regle_tirage") == REGLE_TIRAGE, "règle de tirage différente")
    trajectoires = set(proto["episodes_idx"])
    A.exiger(not (trajectoires & {a for a, _ in exclus}), "trajectoires en commun avec EXP-003 ou EXP-002")
    A.exiger(len(trajectoires) == N5, "plusieurs positions dans une même trajectoire")
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
    f = f"all-6_s{GRAINES[0]}.npz"
    a, b = dict(np.load(Path(dossier_fin) / f)), dict(np.load(Path(dossier_grille) / f))
    A.exiger(not bool(a["git_sale"]), "sanité de clôture : arbre git modifié au lancement")
    A.exiger((str(a["spec"]), int(a["seed"])) == ("all:6", GRAINES[0]), "sanité de clôture : pas la même version")
    verifier_code5(str(a["git"]))
    for k in CLES_SANITE:
        A.exiger(np.array_equal(a[k], b[k], equal_nan=a[k].dtype.kind == "f"), f"sanité de clôture : {k} diffère")


def etat_mesure(d):
    """État sur lequel portent les marges, comme dans controler_course (amendement 4) : pas gelés prolongés ; pour
    une course divergente, pas entièrement finis seulement, ou état de départ si elle diverge dès le pas 0."""
    etat = d["state"].astype(np.float64)
    nan_pas = np.isnan(etat).any(axis=2)
    j = np.where(nan_pas.any(axis=0), nan_pas.argmax(axis=0), etat.shape[0])
    diverge = ~d["success"].astype(bool) & (j < etat.shape[0])
    etat[(np.arange(etat.shape[0])[:, None] >= j[None, :]) & diverge[None, :]] = np.nan
    zero = diverge & (j == 0)
    etat[0, zero] = d["init_state"][zero]
    return prolonger(etat)


def table(res, v):
    S, M2, M3, dT, dth, divergences, limites = [], [], [], [], [], 0, 0
    for g in GRAINES:
        d = res[(v, g)]
        m, mT, lim, div = A.controler_course(d, f"{v} graine {g}")
        limites += lim
        divergences += div
        S.append(d["success"].astype(np.int64))
        M2.append(m)
        M3.append(mT)
        # pour traduire l'écart de pose en unités physiques : position du T (px) et angle (degrés) au plus près,
        # sur les mêmes pas que M3
        etat, but = etat_mesure(d), d["goal_state"].astype(np.float64)
        dT.append(np.nanmin(np.linalg.norm(etat[..., 2:4] - but[None, :, 2:4], axis=-1), axis=0))
        dth.append(np.degrees(np.nanmin(ecart_angle(etat[..., 4], but[None, :, 4]), axis=0)))
    S, M3 = np.stack(S), np.stack(M3)
    return {"S": S, "K": S.sum(axis=0), "M2": np.stack(M2), "M3": M3, "dT_min": np.stack(dT), "dth_min": np.stack(dth),
            "classe": np.where(S == 1, 0, np.where(M3 < 1, 1, 2)), "divergences": divergences, "limites": limites}


def verdict_h6(p):
    return "établie" if p < A.ALPHA else "non établie"


def signe_ic(e):
    """-1 si l'intervalle est entièrement sous 0, +1 s'il est entièrement au-dessus, 0 s'il contient 0."""
    return -1 if e["haut"] < 0 else (1 if e["bas"] > 0 else 0)


def formulation_h6(etablie, ic_tous, ic_strate, sens_progression):
    """Phrase fixée d'avance (PREREG-EXP005, règle de rédaction), selon le sens de l'écart de réussite (tous les
    épisodes et strate) et le sens de l'écart de pose entre 6 et 8 bits (signe_ic de son intervalle)."""
    if not etablie:
        return "H6 non établie"
    sens = {signe_ic(ic_tous), signe_ic(ic_strate)}
    if sens == {0}:
        base = "la pose du T se dégrade à 6 bits alors que la réussite ne bouge pas"
    elif -1 in sens and 1 not in sens:
        base = ("à 6 bits, la pose du T et la réussite se dégradent toutes deux, la réussite restant à ± 2 points "
                "sur l'ensemble des épisodes")
    elif 1 in sens and -1 not in sens:
        base = ("la pose du T se dégrade à 6 bits alors que la réussite augmente légèrement, en restant à ± 2 points "
                "sur l'ensemble des épisodes")
    else:
        base = "la pose du T se dégrade à 6 bits ; l'écart de réussite change de sens entre tous les épisodes et la strate"
    return base + {1: " ; la dégradation est plus faible à 8 bits (progression selon les bits)",
                   0: " ; mais l'effet à 6 bits ne se distingue pas du contrôle à 8 bits",
                   -1: " ; mais la pose est plus mauvaise à 8 bits qu'à 6 bits (pas de progression selon les bits)"}[
        sens_progression]


def stat_h6(x, w, inv):
    """H6 : TOST sur x (différences de K sur 5 graines, tous les épisodes) et test unilatéral sur w (strate)."""
    p_bas, p_haut = inv.p(10 * x + 1, "plus"), inv.p(10 * x - 1, "moins")
    p_b = inv.p(w, "plus")
    return {"p_bas": p_bas, "p_haut": p_haut, "p_b": p_b, "p": max(p_bas, p_haut, p_b)}


def comparer_version(f, q, bouge, inv, boot, boot_strate):
    x = q["K"] - f["K"]
    d3 = A.lg(q["M3"]).mean(axis=0) - A.lg(f["M3"]).mean(axis=0)
    d2 = A.lg(q["M2"]).mean(axis=0) - A.lg(f["M2"]).mean(axis=0)
    stat = stat_h6(x, d3[bouge], inv)
    ecarts = {"reussite": boot.ic(x / 5), "reussite_bouge": boot_strate.ic((x / 5)[bouge]),
              "log_M3": boot.ic(d3), "log_M3_bouge": boot_strate.ic(d3[bouge]), "log_M2": boot.ic(d2)}
    physique = {"dT_min_px": boot_strate.ic((q["dT_min"].mean(axis=0) - f["dT_min"].mean(axis=0))[bouge]),
                "dth_min_deg": boot_strate.ic((q["dth_min"].mean(axis=0) - f["dth_min"].mean(axis=0))[bouge])}
    return {"delta_reussite": float(x.mean() / 5), "delta_log_M3_bouge": float(d3[bouge].mean()),
            "episodes_bouge": int(bouge.sum()), **stat, "ecarts": ecarts,
            "sens_reussite": [signe_ic(ecarts["reussite"]), signe_ic(ecarts["reussite_bouge"])],
            "physique_strate": physique,
            "controle_tost_t": A.tost_t(x / 5)}


def analyser(grille, sanite_fin, exclure, exp003=None, effectifs=EFFECTIFS5):
    exclus = lire_exclus(exclure)
    res, shas = charger5(grille, exclus)
    A.exiger(len(shas) == 1, f"fichiers de plusieurs commits : {shas}")
    verifier_analyse(shas[0])
    verifier_sanite_fin(grille, sanite_fin)
    # depart() d'EXP-003 lit le fichier fp32 de la graine 42 ; EXP-005 utilise les graines 52 à 56
    dep = A.depart({("fp32", A.GA[0]): res[("fp32", GRAINES[0])]})
    if effectifs is not None:
        A.exiger((int(dep["bouge"].sum()), int(dep["au_but"].sum())) == effectifs,
                 f"strates de départ inattendues : {int(dep['bouge'].sum())}, {int(dep['au_but'].sum())}")
    T5 = {v: table(res, v) for v in VERSIONS5}
    bouge = dep["bouge"]
    inv = A.Inversion(n=N5)
    boot, boot_strate = A.Bootstrap(n=N5), A.Bootstrap(n=int(bouge.sum()), graine=3)
    h6 = comparer_version(T5["fp32"], T5["all:6"], bouge, inv, boot, boot_strate)
    h6["verdict"] = verdict_h6(h6["p"])
    controle = comparer_version(T5["fp32"], T5["all:8"], bouge, inv, boot, boot_strate)
    # progression selon les bits : écart apparié de log M3 entre 6 et 8 bits dans la strate (fp32 s'annule)
    d68 = (A.lg(T5["all:6"]["M3"]).mean(axis=0) - A.lg(T5["all:8"]["M3"]).mean(axis=0))[bouge]
    ic68 = boot_strate.ic(d68)
    # décidée par l'intervalle (entièrement au-dessus de 0) ; la p par inversion de signe est rapportée sans valeur de décision
    progression = {"ecart_6_moins_8": ic68, "p_unilaterale": inv.p(d68, "plus"), "sens": signe_ic(ic68),
                   "etablie": signe_ic(ic68) == 1}
    h6["formulation"] = formulation_h6(h6["verdict"] == "établie", h6["ecarts"]["reussite"], h6["ecarts"]["reussite_bouge"],
                                       progression["sens"])
    f = T5["fp32"]
    paires = [(a, b) for a in range(5) for b in range(5) if a != b]
    discord = np.mean([f["S"][a] != f["S"][b] for a, b in paires], axis=0) * 50
    desc = {v: {"reussite": float(t["S"].mean()), "classes": A.parts(t["classe"]),
                "classes_bouge": A.parts(t["classe"][:, bouge])} for v, t in T5.items()}
    desc["plancher_discordants_par_50"] = boot.ic(discord)
    comparaison = None
    if exp003 is not None:
        r3 = json.loads(Path(exp003).read_text())["descriptifs"]
        comparaison = {v: {"reussite": r3[v]["delta_reussite"], "bouge": r3[v]["bouge"]} for v in ("all:6", "all:8")}
    tete = git("rev-parse", "HEAD").stdout.strip()
    return {"controles": {"commits": shas, "commit_analyse": tete,
                          "analyse_modifiee": bool(git("status", "--porcelain", "--untracked-files=no").stdout.strip()),
                          "episodes_bouge": int(bouge.sum()), "episodes_T_au_but": int(dep["au_but"].sum()),
                          "divergences": {v: t["divergences"] for v, t in T5.items()},
                          "cas_limites_arrondi": {v: t["limites"] for v, t in T5.items()}},
            "H6": h6, "controle_8_bits": controle,
            "progression_8_vers_6": progression if h6["verdict"] == "établie" else None,   # rapportée seulement si H6
            "descriptifs": desc, "comparaison_exp003": comparaison}


def rapport(r):
    h, k, c = r["H6"], r["controle_8_bits"], r["controles"]
    L = ["# EXP-005 : analyse pré-enregistrée", "",
         f"Contrôles passés : 15 fichiers (commit {', '.join(x[:7] for x in c['commits'])}), code d'analyse identique à ce commit "
         f"(analyse lancée depuis {c['commit_analyse'][:7]}{', arbre modifié' if c['analyse_modifiee'] else ''}), sanité de clôture identique, "
         f"{c['episodes_bouge']} épisodes où le T doit bouger, {c['episodes_T_au_but']} où il part au but, "
         f"courses divergentes : {sum(c['divergences'].values())}.", "",
         "## H6 : à 6 bits, réussite à ± 2 points et pose du T dégradée", "",
         "| Partie | Estimation, intervalle à 95 % | p |", "| --- | --- | --- |",
         f"| (a) réussite, {N5} épisodes | {A.ic_txt(h['ecarts']['reussite'], 100, 2)} points | bas {h['p_bas']:.4f}, haut {h['p_haut']:.4f} |",
         f"| réussite, {h['episodes_bouge']} épisodes où le T doit bouger | {A.ic_txt(h['ecarts']['reussite_bouge'], 100, 2)} points | |",
         f"| (b) log M3, mêmes {h['episodes_bouge']} épisodes | {A.ic_txt(h['ecarts']['log_M3_bouge'], 1, 3)} | {h['p_b']:.4f} |",
         f"| H6 (intersection-union) | | {h['p']:.4f} : **{h['verdict']}** |", "",
         f"Formulation retenue (règle fixée d'avance) : {h['formulation']}.",
         f"En unités physiques, dans la strate : position du T au plus près {A.ic_txt(h['physique_strate']['dT_min_px'], 1, 2)} px, "
         f"angle au plus près {A.ic_txt(h['physique_strate']['dth_min_deg'], 1, 2)} degrés (repères : la marge du T seul, M3, "
         f"passe sous 1 à 20 px et 20 degrés ; le succès mesure les 20 px sur l'agent et le T ensemble).",
         f"Contrôle TOST t : p = {h['controle_tost_t']['p']:.4f}, intervalle à 90 % "
         f"[{100 * h['controle_tost_t']['ic90'][0]:+.2f}, {100 * h['controle_tost_t']['ic90'][1]:+.2f}] points.", "",
         "## Contrôle à 8 bits (descriptif, mêmes statistiques)", "",
         f"- Réussite : {A.ic_txt(k['ecarts']['reussite'], 100, 2)} points ; log M3 dans la strate : "
         f"{A.ic_txt(k['ecarts']['log_M3_bouge'], 1, 3)} (p unilatérale {k['p_b']:.4f})."]
    if r["progression_8_vers_6"] is not None:
        g = r["progression_8_vers_6"]
        issue = {1: "établie", 0: "non établie", -1: "non établie, pose plus mauvaise à 8 bits qu'à 6 bits"}[g["sens"]]
        L.append(f"- Progression selon les bits (écart de log M3 entre 6 et 8 bits dans la strate, décidée par l'intervalle) : "
                 f"{A.ic_txt(g['ecart_6_moins_8'], 1, 3)} : {issue} (p unilatérale {g['p_unilaterale']:.4f}, "
                 f"sans valeur de décision).")
    L += ["",
         "## Descriptif", ""]
    for v in ("fp32", "all:6", "all:8"):
        d = r["descriptifs"][v]
        L.append(f"- {v} : réussite {100 * d['reussite']:.1f} % ; classes dans la strate "
                 + " / ".join(f"{100 * d['classes_bouge'][x]:.1f}" for x in A.CLASSES) + " (réussite / T placé seul / T jamais placé).")
    L.append(f"- Écart de log M2 à 6 bits : {A.ic_txt(h['ecarts']['log_M2'], 1, 3)} ; plancher de bruit : "
             f"{A.ic_txt(r['descriptifs']['plancher_discordants_par_50'], 1, 2, '')} épisodes discordants sur 50.")
    if r["comparaison_exp003"]:
        L += ["", "## À côté d'EXP-003 (sans regroupement)", ""]
        for v, e in r["comparaison_exp003"].items():
            L.append(f"- EXP-003, {v} : réussite {A.ic_txt(e['reussite'])} points ; dans la strate, réussite "
                     f"{A.ic_txt(e['bouge']['delta_reussite'])}, log M3 {A.ic_txt(e['bouge']['delta_log_M3'], 1, 3)}.")
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
