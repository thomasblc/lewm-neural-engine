"""Analyse pré-enregistrée d'EXP-021 (PREREG-EXP021.md, amendement 1) : prédicteur ternaire avec 1 % de poids gardés et
réglage fin (A), contre le 3 bits réglé (B), le 3 bits non réglé (B0, contrôle de la barre), A sans réglage (C), sans
poids gardés (D), et le ternaire à 3,1 % gardés réglé (E, mémoire voisine de B), sur les 1 000 premières positions des
épisodes d'EXP-005 (graines 52 à 55, un réglage par graine). Tests H1 à H3 (Holm sur 3) ; descriptifs.

Usage :
    python analyse_exp021.py --racine ../resultats --sortie ../resultats/analyse-exp021
    python analyse_exp021.py --racine ../resultats --sortie ../resultats/essai-exp021-analyse --essai
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

import analyse_exp003 as A
from analyse_exp012 import borne_bootstrap_t, p_bootstrap_t

GRAINES = (52, 53, 54, 55)
S = len(GRAINES)
N, N_ESSAI, N_REF = 1000, 50, 2000
K_REPRO, K_REPRO_ESSAI = 100, 50
MARGE = 2
REF = "exp005"
PAS, PAS_ESSAI, LOT, LR = 5000, 50, 256, 1e-3
N_POIDS, LIGNES = 10_764_288, 39_936                   # couches linéaires du prédicteur de LeWM : 30 couches
# données de réglage : 32 épisodes de calibration, CEM graine 0, 8 lignes par appel (lignes par longueur d'historique, pilote)
DONNEES = {"episodes": 32, "graine": 0, "par_appel": 8, "tache": "pusht", "lignes": {1: 8640, 2: 8640, 3: 25920}}
COUCHES = 30
# clé : (version, poids gardés attendus)
BRAS = {"A": ("predictor:2:clip:gptq:k1:obs:aff", 107_640),
        "B": ("predictor:3:clip:gptq:aff", 0),
        "B0": ("predictor:3:clip:gptq", 0),
        "C": ("predictor:2:clip:gptq:k1:obs", 107_640),
        "D": ("predictor:2:clip:gptq:aff", 0),
        "E": ("predictor:2:clip:gptq:k3.1:obs:aff", 333_702)}
TESTS = (("H1", "A", "C", "superiorite"), ("H2", "A", "D", "superiorite"), ("H3", "A", "B", "non_inferiorite"))
DESCRIPTIFS = (("A", "fp32"), ("B", "fp32"), ("B0", "fp32"), ("C", "fp32"), ("D", "fp32"), ("E", "fp32"), ("A", "B"),
               ("A", "C"), ("A", "D"), ("B0", "B"), ("E", "B"), ("E", "A"))
CLES_LIBRES = {"gptq_calib", "affinage_dossier", "premiers"}   # seules clés du protocole qui diffèrent d'EXP-005
FICHIERS_CALCUL = ("metriques.py", "quant.py", "gptq.py", "garder.py", "run_grid.py", "affinage.py")
FICHIERS_ANALYSE = ("analyse_exp021.py", "analyse_exp012.py", "analyse_exp009.py", "analyse_exp006.py", "analyse_exp003.py",
                    "PREREG-EXP021.md")
SENSIBILITE = ((1, 5), (2, 6), (3, 7), (4, 8), (5, 9))
ICI = Path(__file__).resolve().parent


def bits(spec, gardes):
    """Bits par poids des couches linéaires du prédicteur (biais en float32, normes et plongements exclus, identiques
    dans tous les bras) : entiers sur 2 ou 3 bits, une échelle float16 par ligne ; avec poids gardés, 16 bits de valeur
    et 16 bits d'indice de colonne par poids gardé, plus un compte de 16 bits par ligne (format par lignes)."""
    b = 3 if spec.startswith("predictor:3") else 2
    return b + (16 * LIGNES + (32 * gardes + 16 * LIGNES if gardes else 0)) / N_POIDS


def base(spec):
    return spec[:-len(":aff")] if spec.endswith(":aff") else spec


def nom(spec, g):
    return f"{spec.replace(':', '-')}_s{g}.npz"


def sha256(f):
    return hashlib.sha256(Path(f).read_bytes()).hexdigest()


def git(*args):
    return subprocess.run(["git", "-C", str(ICI), *args], capture_output=True, text=True)


def verifier_code(sha):
    """Commit des runs connu, et code de calcul et d'analyse identique entre ce commit et l'arbre de l'analyse."""
    A.exiger(git("cat-file", "-e", f"{sha}^{{commit}}").returncode == 0, f"commit {sha[:7]} inconnu du dépôt")
    A.exiger(git("diff", "--quiet", sha, "--", *FICHIERS_CALCUL, *FICHIERS_ANALYSE).returncode == 0,
             f"code d'analyse ou de calcul différent de celui du commit {sha[:7]}")


def niveaux_holm(ps, alpha=A.ALPHA):
    """Niveau de chaque test dans Holm ; ex aequo rangés dans l'ordre des tests (tri stable), comme A.holm."""
    m = len(ps)
    niv = [None] * m
    k, arrete = 0, False
    for rang, i in enumerate(np.argsort(ps, kind="stable")):
        if not arrete and ps[i] <= alpha / (m - rang):
            niv[i] = alpha / (m - rang)
            k += 1
        else:
            arrete = True
            niv[i] = alpha / (m - k)
    return niv


def tranche(x, k, chunk):
    """Les k premières positions d'un fichier de run_grid : axe 0 pour les tableaux à une dimension par épisode et pour
    init_state et goal_state, axe 1 pour les tableaux (pas, épisodes, ...), un élément par paquet pour n_steps."""
    out = {}
    for c, v in x.items():
        if v.ndim == 0:                                  # champs d'un seul nombre ou d'un seul texte (version, commit...)
            continue
        if c == "n_steps":
            out[c] = v[:k // chunk]
        elif v.ndim >= 2 and c not in ("init_state", "goal_state"):
            out[c] = v[:, :k]
        else:
            out[c] = v[:k]
    return out


def identiques_premiers(f, ref, k, nom_):
    a, b = dict(np.load(f)), dict(np.load(ref))
    A.exiger(int(a["chunk"]) == int(b["chunk"]), f"{nom_} : paquets de tailles différentes")
    ta, tb = tranche(a, k, int(a["chunk"])), tranche(b, k, int(b["chunk"]))
    A.exiger(set(ta) == set(tb), f"{nom_} : clés différentes {sorted(set(ta) ^ set(tb))}")
    for c in sorted(ta):
        A.exiger(np.array_equal(ta[c], tb[c], equal_nan=ta[c].dtype.kind == "f"), f"{nom_} : {c} diffère")


def charger(racine, essai=False, verifier=True):
    import torch
    r = Path(racine)
    n, k_repro, pas_attendu = (N_ESSAI, K_REPRO_ESSAI, PAS_ESSAI) if essai else (N, K_REPRO, PAS)
    pre = "essai-exp021" if essai else "exp021"
    dossier, collecte = r / f"{pre}-pusht", r / f"{pre}-collecte"
    refs = {g: dict(np.load(r / REF / f"fp32_s{g}.npz")) for g in GRAINES}
    for g, x in refs.items():
        A.exiger(len(x["success"]) == N_REF, f"{REF} graine {g} : {len(x['success'])} épisodes au lieu de {N_REF}")
    proto_ref = json.loads((r / REF / "meta.json").read_text())["protocole"]
    calib10 = json.loads((r / "exp010" / "meta.json").read_text())["protocole"]["gptq_calib"]

    A.exiger((dossier / "meta.json").exists(), f"{dossier.name}/meta.json manquant")
    p = json.loads((dossier / "meta.json").read_text())["protocole"]
    A.exiger({c: v for c, v in p.items() if c not in CLES_LIBRES} == {c: v for c, v in proto_ref.items() if c not in CLES_LIBRES},
             f"{dossier.name} : protocole différent de {REF}")
    A.exiger(p.get("gptq_calib") == calib10, f"{dossier.name} : calibration GPTQ {p.get('gptq_calib')}")
    A.exiger(p.get("affinage_dossier") == collecte.name, f"{dossier.name} : dossier de réglage {p.get('affinage_dossier')}")
    A.exiger(p.get("premiers") == n, f"{dossier.name} : premiers {p.get('premiers')} au lieu de {n}")

    f_donnees = collecte / "affinage-donnees.pt"
    A.exiger(f_donnees.exists(), f"{collecte.name}/affinage-donnees.pt manquant")
    donnees = torch.load(f_donnees, map_location="cpu")
    sha_donnees = sha256(f_donnees)
    A.exiger((len(donnees["episodes_idx"]), int(donnees["graine"]), int(donnees["par_appel"]), donnees["tache"])
             == (DONNEES["episodes"], DONNEES["graine"], DONNEES["par_appel"], DONNEES["tache"]),
             f"données de réglage : {len(donnees['episodes_idx'])} épisodes, graine {donnees['graine']}, "
             f"{donnees['par_appel']} lignes par appel, tâche {donnees['tache']}")
    lignes = {int(T): int(v.shape[0]) for T, v in donnees["X"].items()}
    A.exiger(lignes == DONNEES["lignes"] and all(int(donnees["C"][T].shape[0]) == lignes[int(T)] for T in donnees["C"]),
             f"données de réglage : lignes {lignes} au lieu de {DONNEES['lignes']}")
    eval_traj = set(int(e) for e in refs[GRAINES[0]]["episodes_idx"][:n])
    commun = eval_traj & set(int(e) for e in donnees["episodes_idx"])
    A.exiger(not commun, f"données de réglage : {len(commun)} trajectoires communes avec l'évaluation")

    Kg = {"fp32": {g: refs[g]["success"][:n].astype(np.int64) for g in GRAINES}}
    shas, reglages = {str(donnees["git"])}, {}
    for cle, (spec, gardes) in BRAS.items():
        Kg[cle] = {}
        for g in GRAINES:
            f = dossier / nom(spec, g)
            A.exiger(f.exists(), f"{dossier.name}/{f.name} manquant")
            x = dict(np.load(f))
            ici = f"{dossier.name}/{f.name}"
            A.exiger((str(x["spec"]), int(x["seed"])) == (spec, g), f"{ici} : version ou graine")
            A.exiger(not bool(x["git_sale"]), f"{ici} : arbre git modifié")
            A.exiger(int(x["n_couches"]) == COUCHES, f"{ici} : {int(x['n_couches'])} couches quantifiées au lieu de {COUCHES}")
            A.exiger(int(x["n_gardes"]) == gardes, f"{ici} : {int(x['n_gardes'])} poids gardés au lieu de {gardes}")
            A.exiger((int(x["n_poids_quantifies"]), int(x["n_lignes_quantifiees"])) == (N_POIDS, LIGNES),
                     f"{ici} : {int(x['n_poids_quantifies'])} poids et {int(x['n_lignes_quantifiees'])} lignes quantifiés")
            A.exiger(len(x["success"]) == n, f"{ici} : {len(x['success'])} épisodes au lieu de {n}")
            for c in ("episodes_idx", "start_steps", "init_state", "goal_state"):
                A.exiger(np.array_equal(x[c], refs[g][c][:n]), f"{ici} : {c} différent de la référence")
            A.controler_course(x, ici)
            if spec.endswith(":aff"):
                f_aff = collecte / f"{base(spec).replace(':', '-')}_g{g}.pt"
                A.exiger(f_aff.exists(), f"{collecte.name}/{f_aff.name} manquant")
                A.exiger(str(x["affinage_sha256"]) == sha256(f_aff), f"{ici} : réglage chargé différent de {f_aff.name}")
                R = torch.load(f_aff, map_location="cpu")
                A.exiger((R["spec"], int(R["graine"])) == (base(spec), g), f"{f_aff.name} : version ou graine du réglage")
                A.exiger(R["donnees_sha256"] == sha_donnees, f"{f_aff.name} : données de réglage différentes")
                A.exiger((int(R["pas"]), int(R["lot"]), float(R["lr"])) == (pas_attendu, LOT, LR),
                         f"{f_aff.name} : réglage {R['pas']} pas, lots de {R['lot']}, taux {R['lr']}")
                av, ap = float(R["controle_avant"]), float(R["controle_apres"])
                A.exiger(np.isfinite([av, ap]).all() and 0 <= ap < av, f"{f_aff.name} : perte de contrôle {av} -> {ap}, pas de baisse")
                reglages[f"{cle}_g{g}"] = {"controle_avant": av, "controle_apres": ap}
            else:
                A.exiger(str(x["affinage_sha256"]) == "", f"{ici} : réglage chargé alors que la version n'est pas réglée")
            shas.add(str(x["git"]))
            Kg[cle][g] = x["success"].astype(np.int64)
    A.exiger(len(shas) == 1, f"EXP-021 : fichiers de plusieurs commits {sorted(shas)}")
    sha = shas.pop()
    for d, g in ((f"{pre}-repro-debut", GRAINES[0]), (f"{pre}-repro-fin", GRAINES[-1])):
        f = r / d / f"fp32_s{g}.npz"
        A.exiger(f.exists(), f"{d}/{f.name} manquant")
        identiques_premiers(f, r / REF / f"fp32_s{g}.npz", k_repro, f"reproduction d'{REF}, graine {g}")
        x = np.load(f)
        A.exiger(str(x["git"]) == sha and not bool(x["git_sale"]), f"{d} : autre commit ou arbre modifié")
    if verifier:
        verifier_code(sha)
    return Kg, {"commit": sha, "n": n, "donnees_sha256": sha_donnees, "reglages": reglages,
               "lignes_donnees": {int(t): int(v.shape[0]) for t, v in donnees["X"].items()}}


def sensibilite(y, p0, niveau):
    ps = [max(A.Inversion(n=len(y), graine=gs).p(y, "plus"), p_bootstrap_t(y, graine=gb)) for gs, gb in SENSIBILITE]
    return {"p": ps, "part_qui_change": float(np.mean([(p <= niveau) != (p0 <= niveau) for p in ps]))}


def lecture(v, ecarts, par_graine, b0_moins_b):
    """Lecture écrite d'avance (PREREG-EXP021.md, amendement 1). Les verdicts portent sur ces quatre réglages (un par
    graine) ; la phrase dit si l'écart estimé passe le seuil dans chacun des quatre réglages (descriptif)."""
    f = lambda e: f"{e['moyenne']:+.1f} [{e['bas']:+.1f} ; {e['haut']:+.1f}]"
    ind = "indéterminé (borne et verdict discordants)"
    ens = lambda t, seuil: " ; " + (f"l'écart estimé dépasse {seuil:+d} points dans chacun des 4 réglages"
                                    if min(par_graine[t]) > seuil else
                                    f"l'écart estimé varie selon le réglage ({min(par_graine[t]):+.1f} à {max(par_graine[t]):+.1f} points)")
    L = {}
    L["H1"] = ind if v["H1"] is None else (
        "pour ces quatre réglages, régler les échelles, les valeurs gardées et les biais sur des entrées enregistrées en "
        "planification améliore le prédicteur ternaire avec 1 % gardé de plus de 2 points" + ens("H1", MARGE) if v["H1"] else
        f"gain de plus de 2 points du réglage non établi (écart {f(ecarts['A - C'])} points, descriptif)")
    L["H2"] = ind if v["H2"] is None else (
        "pour ces quatre réglages, garder 1 % des poids en 16 bits apporte plus de 2 points au prédicteur ternaire réglé"
        + ens("H2", MARGE) if v["H2"] else
        f"apport de plus de 2 points des poids gardés non établi (écart {f(ecarts['A - D'])} points, descriptif)")
    L["H3"] = ind if v["H3"] is None else (
        "pour ces quatre réglages, à 2,44 bits par poids des couches linéaires du prédicteur, la version ternaire avec "
        "1 % gardé et réglée ne perd pas plus de 2 points face au 3 bits réglé de la même façon (3,06 bits), sur des "
        "épisodes neufs" + ens("H3", -MARGE) if v["H3"] else
        f"non-infériorité face au 3 bits réglé non établie (écart {f(ecarts['A - B'])} points, descriptif)")
    if b0_moins_b > MARGE:
        L["H3"] += (f" ; B0 non réglé dépasse B de {f(ecarts['B0 - B'])} points (descriptif, non testé) : la barre B réglé "
                    "est peut-être dégradée par le réglage")
    L["pleine_precision"] = f"écart de A à la pleine précision {f(ecarts['A - fp32'])} points (descriptif, aucun test)"
    L["memoire_voisine"] = f"E moins B, mémoire voisine (3,11 contre 3,06 bits) : {f(ecarts['E - B'])} points, non testé"
    return L


def analyser(racine, essai=False, verifier=True):
    Kg, info = charger(racine, essai, verifier)
    n = info["n"]
    K = {c: sum(Kg[c][g] for g in GRAINES) for c in Kg}
    inv, boot = A.Inversion(n=n), A.Bootstrap(n=n)
    ecart = lambda a, b: K[a] - K[b]
    y_de = {"non_inferiorite": lambda x: 100 * x + MARGE * S, "superiorite": lambda x: 100 * x - MARGE * S}
    tests, par_graine = {}, {}
    for t, b, ref, s in TESTS:
        x = ecart(b, ref)
        y = y_de[s](x)
        ps, pb = inv.p(y, "plus"), p_bootstrap_t(y)
        par_graine[t] = [100 * float((Kg[b][g] - Kg[ref][g]).mean()) for g in GRAINES]
        tests[t] = {"bras": b, "reference": ref, "sorte": s,
                    "ecart_points_descriptif": {c: 100 * v for c, v in boot.ic(x / S).items()},
                    "ecart_par_reglage_points": dict(zip(map(str, GRAINES), par_graine[t])),
                    "p_signe": ps, "p_bootstrap_t": pb, "erreur_mc_p_signe": float(np.sqrt(ps * (1 - ps) / A.B)),
                    "erreur_mc_p_bootstrap_t": float(np.sqrt(pb * (1 - pb) / 10_000)), "p": max(ps, pb)}
    ps = [tests[t]["p"] for t, *_ in TESTS]
    for (t, b, ref, s), q, niv in zip(TESTS, A.holm(ps), niveaux_holm(ps)):
        d = tests[t]
        d["p_holm"], d["niveau_holm"] = float(q), float(niv)
        d["verdict"] = "soutenue" if q < A.ALPHA else "non soutenue"
        d["borne_points"] = 100 * borne_bootstrap_t(ecart(b, ref) / S, niv)
        seuil = -MARGE if s == "non_inferiorite" else MARGE
        d["borne_et_verdict_concordent"] = (d["borne_points"] > seuil) == (d["verdict"] == "soutenue")
        d["verdict_ecrit"] = d["verdict"] if d["borne_et_verdict_concordent"] else "indéterminé (borne et verdict discordants)"
        mc = d["erreur_mc_p_signe"] if d["p"] == d["p_signe"] else d["erreur_mc_p_bootstrap_t"]
        if abs(d["p"] - niv) < 3 * mc:
            d["sensibilite_tirages"] = sensibilite(y_de[s](ecart(b, ref)), d["p"], niv)
    ecarts = {f"{a} - {b}": {c: 100 * v for c, v in boot.ic(ecart(a, b) / S).items()} for a, b in DESCRIPTIFS}
    v = {t: (tests[t]["verdict"] == "soutenue") if tests[t]["borne_et_verdict_concordent"] else None for t, *_ in TESTS}
    desc = {"ecarts": ecarts, "reussite": {c: float(K[c].mean() / S) for c in K},
            "reussite_par_graine": {c: {str(g): float(Kg[c][g].mean()) for g in GRAINES} for c in Kg},
            "bits_par_poids": {c: bits(spec, g) for c, (spec, g) in BRAS.items()}, "reglages": info["reglages"],
            "lignes_donnees": info["lignes_donnees"]}
    return {"controles": {"commit": info["commit"], "episodes": n, "graines": list(GRAINES),
                          "donnees_sha256": info["donnees_sha256"]},
            "tests": tests, "lecture": lecture(v, ecarts, par_graine, ecarts["B0 - B"]["moyenne"]), "descriptifs": desc}


def rapport(r):
    f = lambda e: f"{e['moyenne']:+.1f} [{e['bas']:+.1f} ; {e['haut']:+.1f}]"
    d = r["descriptifs"]
    L = [f"# EXP-021 : rapport de l'analyse pré-enregistrée ({r['controles']['episodes']} positions d'EXP-005, PushT, "
         f"graines {', '.join(map(str, r['controles']['graines']))})", "",
         f"Commit des runs : `{r['controles']['commit'][:7]}`. Verdicts conditionnels aux quatre réglages (un par graine).", "",
         "| Test | Bras contre référence | Écart moyen (points) | Borne inférieure du test (niveau de Holm) | Écart par réglage | p Holm | Verdict |",
         "| --- | --- | --- | --- | --- | --- | --- |"]
    for t, x in r["tests"].items():
        L.append(f"| {t} ({x['sorte']}) | {x['bras']} contre {x['reference']} | {x['ecart_points_descriptif']['moyenne']:+.1f} | "
                 f"{x['borne_points']:+.2f} | {', '.join(f'{v:+.1f}' for v in x['ecart_par_reglage_points'].values())} | "
                 f"{x['p_holm']:.4f} | {x['verdict_ecrit']} |")
    sens = [(t, x["sensibilite_tirages"]) for t, x in r["tests"].items() if "sensibilite_tirages" in x]
    if sens:
        L.append("")
        L += [f"- {t}, sensibilité aux tirages : p {', '.join(f'{p:.4f}' for p in s['p'])} ; part qui change de côté "
              f"{s['part_qui_change']:.1f}" for t, s in sens]
    L += ["", "Lecture :", ""] + [f"- {c} : {v}" for c, v in r["lecture"].items()]
    L += ["", "| Bras | Bits par poids des couches linéaires du prédicteur | Réussite |", "| --- | --- | --- |",
          f"| pleine précision | 32 | {100 * d['reussite']['fp32']:.1f} % |"]
    L += [f"| {c} | {d['bits_par_poids'][c]:.2f} | {100 * d['reussite'][c]:.1f} % |" for c in BRAS]
    L += ["", "Écarts descriptifs (points, intervalle percentile à 95 %, non utilisé par les tests) :", ""]
    L += [f"- {c} : {f(e)}" for c, e in d["ecarts"].items()]
    L += ["", "Pertes de contrôle du réglage (avant, après) : "
          + " ; ".join(f"{c} {v['controle_avant']:.2e} -> {v['controle_apres']:.2e}" for c, v in d["reglages"].items())]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--racine", required=True)
    ap.add_argument("--sortie", required=True)
    ap.add_argument("--essai", action="store_true")
    a = ap.parse_args()
    r = A.pur(analyser(a.racine, a.essai))
    out = Path(a.sortie)
    out.mkdir(parents=True, exist_ok=True)
    (out / "resultats.json").write_text(json.dumps(r, indent=1, ensure_ascii=False))
    (out / "rapport.md").write_text(rapport(r))
    print(rapport(r))


if __name__ == "__main__":
    main()
