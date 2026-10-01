"""Tests de analyse_exp012.py (amendement 1) sur de vrais fichiers de référence (EXP-003, EXP-006, copiés dans un
dossier temporaire) et des fichiers Core ML fabriqués à effet connu. Scénario A : chaque moteur = copie de la pleine
précision (aucune perte). Scénario C : effets proches de la marge, p strictement entre le plancher et 1, qui changent
avec la marge. p par inversion de signe, bootstrap-t et Holm recomptés en numpy, indépendamment du code testé ;
contrôle négatif et sensibilité recomptés ; temps, puissance et énergie recomptés, appel allongé par une pause
écarté ; refus des fichiers incohérents, chacun avec son motif. À lancer sur un arbre commité.

Usage :
    python test_analyse_exp012.py
"""
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

import analyse_exp003 as A
import analyse_exp012 as A12

RES = A12.ICI.parent / "resultats"
SIGNES = np.random.default_rng(0).choice(np.array([-1.0, 1.0]), size=(10_000, 500))
TETE = subprocess.run(["git", "-C", str(A12.ICI), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
PROPRE = subprocess.run(["git", "-C", str(A12.ICI), "status", "--porcelain", "--untracked-files=no"],
                        capture_output=True, text=True).stdout.strip() == ""
SOURCES = {"A": {(t, m): "fp32" for t in A12.TACHES for m in A12.MOTEURS},
           "C": {("pusht", "ne"): "all-6", ("tworoom", "ne"): "all-6", ("pusht", "gpu"): "predictor-3",
                 ("tworoom", "gpu"): "autres-4"}}
TEMPS = {"torch": 0.79, "ne": 0.88, "gpu": 0.40}         # secondes par plan, fabriquées
PUISS = {"torch": 84.0, "ne": 93.0, "gpu": 66.0}
REPOS = 30.0


def ecrire_temps(f, moteur, cle, g, empreinte=None):
    appels = [[TEMPS[cle] * n, n] for n in [50] * 10 + [12] * 10]
    if cle == "ne" and g == 44:
        appels[3] = [TEMPS[cle] * 50 + 600.0, 50]           # un appel allongé de 10 min par une pause de batterie
    coreml = {} if cle == "torch" else {"placement": {A12.PLACEMENT[cle]: 361 if cle == "ne" else 197},
                                        "sha256_poids": empreinte, "sha256_modele": f"modele-{empreinte}"}
    f.with_suffix(".temps.json").write_text(json.dumps({"moteur": moteur, "appels_solve": appels, "coreml": coreml,
                                                        "puissance": {"puissance_moyenne_w": PUISS[cle]}, "git": TETE,
                                                        "lectures": [{"t": 2000.0 + g}]}))


def fabriquer(racine, scenario="A"):
    r = Path(racine)
    for d in A12.TACHES.values():
        shutil.copytree(RES / d, r / d, ignore=shutil.ignore_patterns("*.log"))
    for t, ref_d in A12.TACHES.items():
        for debut, w in ((1000, REPOS), (5000, 999.0)):         # la seconde mesure vient après tous les fichiers
            (r / f"exp012-repos-{t}-{debut}.json").write_text(json.dumps({
                "tache": t, "debut": debut, "fin": debut + 130, "git": TETE, "secondes_couvertes": 120,
                "puissance_moyenne_w": w, "lectures": [{"processus_run_grid": 0}]}))
        for d, g in ((f"exp012-repro-{t}", 42), (f"exp012-torch-{t}", 43), (f"exp012-fin-{t}", 46)):
            (r / d).mkdir()
            x = dict(np.load(r / ref_d / f"fp32_s{g}.npz"))
            x.update(git=np.str_(TETE), git_sale=np.bool_(False))
            np.savez_compressed(r / d / f"fp32_s{g}.npz", **x)
            ecrire_temps(r / d / f"fp32_s{g}.npz", "torch", "torch", g)
        proto = json.loads((r / ref_d / "meta.json").read_text())["protocole"]
        for m, moteur in A12.MOTEURS.items():
            d = r / A12.dossier(t, m)
            d.mkdir()
            pm = dict(proto, moteur={"nom": moteur, "coremltools": "9.0", "macos": "26.6.2", "precision": "float16"})
            (d / "meta.json").write_text(json.dumps({"protocole": pm, "git": TETE, "git_sale": False}))
            for g in A12.GRAINES:
                x = dict(np.load(r / ref_d / f"{SOURCES[scenario][(t, m)]}_s{g}.npz"))
                x.update(spec=np.str_("fp32"), git=np.str_(TETE), n_couches=np.int64(0), moteur=np.str_(moteur),
                         coreml_sha256=np.str_(f"{t}-{m}" * 4))
                np.savez_compressed(d / f"fp32_s{g}.npz", **x)
                ecrire_temps(d / f"fp32_s{g}.npz", moteur, m, g, f"{t}-{m}" * 4)
    return r


def K(r, d, v, graines=A12.GRAINES):
    return sum(np.load(Path(r) / d / f"{v}_s{g}.npz")["success"].astype(float) for g in graines)


def p_signe(y):
    return (1 + int((SIGNES @ y >= y.sum()).sum())) / 10_001


def p_boot_t(y):
    """Bootstrap-t écrit à part : t = moyenne / erreur-type, t* = (moyenne* - moyenne) / erreur-type*."""
    n, rng = len(y), np.random.default_rng(4)
    t = y.mean() / (y.std(ddof=1) / np.sqrt(n)) if y.std() > 0 else np.inf
    idx = rng.integers(0, n, size=(10_000, n))
    compte = 0
    for i in range(0, 10_000, 2_000):
        yb = y[idx[i:i + 2_000]]
        se = yb.std(axis=1, ddof=1) / np.sqrt(n)
        tb = np.where(se > 0, (yb.mean(axis=1) - y.mean()) / np.where(se > 0, se, 1), 0.0)
        compte += int((tb >= t).sum())
    return (1 + compte) / 10_001


def holm_numpy(ps):
    o = np.argsort(ps)
    q = np.minimum(1, np.maximum.accumulate(np.array(ps)[o] * (len(ps) - np.arange(len(ps)))))
    out = np.empty(len(ps))
    out[o] = q
    return out


def verifier_p(r, res):
    ps = []
    for nom, t, m in A12.TESTS:
        x = K(r, A12.dossier(t, m), "fp32") - K(r, A12.TACHES[t], "fp32")
        y = 20 * x + 2
        s, b = p_signe(y), p_boot_t(y)
        T = res["tests"][nom]
        assert T["p_signe"] == s and abs(T["p_bootstrap_t"] - b) < 1e-12 and T["p"] == max(s, b), (nom, T, s, b)
        ps.append(max(s, b))
    for (nom, *_), q in zip(A12.TESTS, holm_numpy(ps)):
        assert abs(res["tests"][nom]["p_holm"] - q) < 1e-12, nom
        assert res["tests"][nom]["verdict"] == ("soutenue" if q < 0.05 else "non soutenue"), nom
    for t, ref_d in A12.TACHES.items():                       # contrôle négatif et sensibilité
        f, rep = K(r, ref_d, "fp32"), K(r, ref_d, "fp32", A12.REPLIQUE)
        c = res["controle_negatif"][t]
        assert c["p_signe"] == p_signe(20 * (rep - f) + 2) and abs(c["ecart_points"]["moyenne"] - (rep - f).mean() / 5) < 1e-12
    for nom, t, m in A12.TESTS:
        x = K(r, A12.dossier(t, m), "fp32") - (K(r, A12.TACHES[t], "fp32") + K(r, A12.TACHES[t], "fp32", A12.REPLIQUE)) / 2
        assert res["sensibilite_10_graines"][nom]["p_signe"] == p_signe(20 * x + 2), nom
    return ps


def test_scenario_a(tmp):
    r = fabriquer(Path(tmp) / "a")
    res = A12.analyser(r, verifier=False)
    verifier_p(r, res)
    assert all(res["tests"][n]["verdict"] == "soutenue" and res["tests"][n]["somme"] == 0 for n in res["tests"])
    for t in A12.TACHES:
        d = res["descriptifs"][t]
        assert d["ne_moins_gpu"]["moyenne"] == 0 and d["reussite_ne"] == d["reussite_fp32"]
        for m in ("torch", "ne", "gpu"):
            s = d["temps"][m]
            assert abs(s["par_plan_ms"] / 1000 - TEMPS[m]) < 1e-9 and s["puissance_w"] == PUISS[m], (t, m, s)
            assert abs(s["energie_par_plan_j"] - (PUISS[m] - REPOS) * TEMPS[m]) < 1e-9 and s["puissance_repos_w"] == REPOS
        assert d["temps"]["ne"]["appels_ecartes"] == 1 and d["temps"]["gpu"]["appels_ecartes"] == 0
        assert d["temps"]["ne"]["plans"] == 5 * 620 and d["temps"]["torch"]["fichiers"] == 1
    for nom in res["tests"]:                                  # borne inférieure : nulle quand l'écart est nul partout
        assert res["tests"][nom]["borne_inferieure_points"] == 0.0 and res["tests"][nom]["borne_et_verdict_concordent"]
    niv = A12.niveaux_holm([0.02, 0.03, 0.03, 0.04])         # Holm s'arrête au premier test : tous au niveau 0,0125
    assert niv == [0.0125] * 4 and A12.niveaux_holm([0.001, 0.5, 0.01, 0.02]) == [0.0125, 0.05 / 1, 0.05 / 3, 0.05 / 2]
    assert "Contrôle négatif" in A12.rapport(res)
    print("ok scénario A : 4 hypothèses soutenues, contrôle négatif et sensibilité recomptés, temps et énergie au-dessus "
          "du repos recomptés, appel allongé par une pause écarté")
    return r


def test_scenario_c(tmp):
    r = fabriquer(Path(tmp) / "c", "C")
    res = A12.analyser(r, verifier=False)
    ps = verifier_p(r, res)
    assert all(1 / 10_001 < p < 1 for p in ps), ps
    x = K(r, A12.dossier("pusht", "ne"), "fp32") - K(r, "exp003", "fp32")
    assert p_signe(20 * x + 1) != res["tests"]["N1"]["p_signe"] != p_signe(20 * x + 3)       # la p dépend de la marge
    T = res["tests"]
    assert T["N1"]["borne_inferieure_points"] < T["N1"]["ecart_points"]["moyenne"] * 100
    print(f"ok scénario C : p intermédiaires, sensibles à la marge, Holm recompté ; verdicts "
          f"{[T[n]['verdict'] for n in ('N1', 'N2', 'N3', 'N4')]}")


def test_refus(tmp, r):
    def essai(motif, modifier):
        rr = Path(tmp) / "refus"
        shutil.rmtree(rr, ignore_errors=True)
        shutil.copytree(r, rr)
        modifier(rr)
        try:
            A12.analyser(rr, verifier=False)
        except A.ControleEchoue as e:
            assert motif in str(e), (motif, str(e))
            return
        raise AssertionError(f"accepté à tort : {motif}")

    def champ(d, nom, cle, val):
        def m(rr):
            x = dict(np.load(rr / d / nom))
            x[cle] = val(x)
            np.savez_compressed(rr / d / nom, **x)
        return m

    def meta(d, f):
        def m(rr):
            g = rr / d / "meta.json"
            mm = json.loads(g.read_text())
            f(mm["protocole"])
            g.write_text(json.dumps(mm))
        return m

    def repos(t, f):
        def m(rr):
            g = rr / f"exp012-repos-{t}-1000.json"
            j = json.loads(g.read_text())
            f(j)
            g.write_text(json.dumps(j))
        return m

    def temps(d, nom, f):
        def m(rr):
            g = rr / d / nom
            j = json.loads(g.read_text())
            f(j)
            g.write_text(json.dumps(j))
        return m
    cas = [("reproduction pusht graine 42", lambda rr: shutil.copy(rr / "exp003" / "all-4_s42.npz", rr / "exp012-repro-pusht" / "fp32_s42.npz")),
           ("reproduction tworoom graine 43", champ("exp012-torch-tworoom", "fp32_s43.npz", "success", lambda x: ~x["success"])),
           ("reproduction pusht graine 46", lambda rr: shutil.copy(rr / "exp003" / "fp32_s45.npz", rr / "exp012-fin-pusht" / "fp32_s46.npz")),
           ("reproduction tworoom graine 46 : autre commit", champ("exp012-fin-tworoom", "fp32_s46.npz", "git", lambda x: np.str_("c" * 40))),
           ("manquant", lambda rr: (rr / "exp012-ne-pusht" / "fp32_s44.npz").unlink()),
           ("version ou graine", champ("exp012-gpu-tworoom", "fp32_s43.npz", "seed", lambda x: np.int64(42))),
           ("arbre git modifié", champ("exp012-ne-tworoom", "fp32_s45.npz", "git_sale", lambda x: np.bool_(True))),
           ("couches quantifiées", champ("exp012-gpu-pusht", "fp32_s46.npz", "n_couches", lambda x: np.int64(30))),
           ("moteur", champ("exp012-ne-pusht", "fp32_s42.npz", "moteur", lambda x: np.str_("coreml-gpu"))),
           ("empreintes", champ("exp012-ne-tworoom", "fp32_s44.npz", "coreml_sha256", lambda x: np.str_("autre"))),
           ("modèle Core ML différent", temps("exp012-gpu-pusht", "fp32_s45.temps.json", lambda j: j["coreml"].update(sha256_modele="x"))),
           ("hors du processeur demandé", temps("exp012-ne-tworoom", "fp32_s46.temps.json",
                                                lambda j: j["coreml"].update(placement={"NeuralEngine": 300, "CPU": 61}))),
           ("hors du processeur demandé", temps("exp012-gpu-pusht", "fp32_s42.temps.json", lambda j: j["coreml"].update(placement={}))),
           ("protocole différent", meta("exp012-gpu-pusht", lambda p: p.update(chunk=32))),
           ("moteur", meta("exp012-ne-tworoom", lambda p: p["moteur"].update(precision="float32"))),
           ("init_state différent", champ("exp012-gpu-tworoom", "fp32_s42.npz", "init_state", lambda x: x["init_state"] + 1)),
           ("plusieurs commits", lambda rr: (champ("exp012-gpu-tworoom", "fp32_s45.npz", "git", lambda x: np.str_("b" * 40))(rr),
                                             temps("exp012-gpu-tworoom", "fp32_s45.temps.json", lambda j: j.update(git="b" * 40))(rr))),
           ("fichier de temps", champ("exp012-ne-pusht", "fp32_s46.npz", "git", lambda x: np.str_("d" * 40))),
           ("fichier de temps", temps("exp012-ne-pusht", "fp32_s43.temps.json", lambda j: j.update(moteur="torch"))),
           ("fichier de temps", temps("exp012-torch-pusht", "fp32_s43.temps.json", lambda j: j.update(moteur="coreml-ne"))),
           ("aucune mesure de la puissance au repos", lambda rr: (rr / "exp012-repos-tworoom-1000.json").unlink()),
           ("mesure au repos invalide", repos("pusht", lambda j: j.update(lectures=[{"processus_run_grid": 2}]))),
           ("mesure au repos invalide", repos("tworoom", lambda j: j.update(git="e" * 40))),
           ("mesure au repos invalide", repos("pusht", lambda j: j.update(secondes_couvertes=40))),
           ("sans lectures", temps("exp012-gpu-tworoom", "fp32_s44.temps.json", lambda j: j.update(lectures=[])))]
    for motif, m in cas:
        essai(motif, m)
    print(f"ok contrôles bloquants ({len(cas)} refus, chacun avec son motif)")


def test_code():
    assert PROPRE, "arbre modifié : committer avant de lancer les tests"
    A12.verifier_code(TETE)
    print(f"ok code : fichiers d'analyse et de calcul identiques à {TETE[:7]}")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp, np.errstate(all="ignore"):
        r = test_scenario_a(tmp)
        test_scenario_c(tmp)
        test_refus(tmp, r)
        test_code()
    print("tous les tests passent")
