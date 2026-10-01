"""EXP-013 (PREREG-EXP013.md, amendements 2 à 5) : enchaîne les mesures de vitesse et d'énergie.

- 3 répétitions ; ordre de la répétition 1 tiré au hasard (default_rng(13)), répétition 2 dans l'ordre inverse
  (contrebalancement de la dérive), répétition 3 tirée au hasard ; la vision seule jamais en tête ni en dernier, ses
  deux mesures jamais côte à côte (la garde de batterie ne doit pas croire le calcul arrêté).
- Mesures au repos par composant (repos_puce.py) avant la répétition 1, entre les répétitions et après la 3 ; refaites
  une fois sur-le-champ si un calcul, une charge de vision ou un processus de fond tournait, jamais refaites plus tard.
- Chaque configuration est un appel de run_grid.py sur les 50 épisodes de la sonde, graine 42, avec l'énergie de la
  puce. Avec charge : run_grid charge et convertit le modèle, fait sa mise en route, écrit GO.pret et attend ; la
  vision démarre alors (prête, puis 10 s de chauffe), puis le signal GO lance la planification ; à pleine vitesse ou à
  30 images par seconde ; si la garde gèle le run, la vision est gelée aussi.
- Vision seule (pleine vitesse et 30 images par seconde) : prête, 10 s de chauffe, 35 s de mesure d'énergie.
- Un run allongé par une pause (fil de lecture gelé plus de 7,5 s), partagé avec un autre calcul run_grid, ou dont la
  vision s'est arrêtée, est effacé et refait, 3 essais au plus ; à la reprise, un run déjà écrit est recontrôlé.
- Ensuite : sensibilité à l'attente active d'OpenMP (ne-complet et coreml-cpu, OMP_WAIT_POLICY par défaut et PASSIVE,
  ordre ABBA), contrôle de réussite du planificateur sur le Neural Engine (500 épisodes, 2 graines, sans contrôle de
  pause : la réussite ne dépend pas du temps), run torch final, balayage de la taille du lot (son échec n'empêche pas
  les contrôles finaux), contrôles bloquants (runs torch identiques au pilote d'EXP-012).
- État de la machine (pmset, batterie, fils, processus actifs) écrit au début et à la fin.
- Reprise : refusée si un fichier d'EXP-013 vient d'un autre commit (tout recommencer dans un dossier propre).
--essai : même chemin en court (10 épisodes, mesures brèves), dans un dossier de travail, pour vérifier le
fonctionnement de bout en bout ; jamais dans resultats/.

Usage (depuis code-exp013, par le lanceur) :
    python vitesse.py --racine ../resultats
    python vitesse.py --racine DOSSIER_DE_TRAVAIL --essai
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import energie_puce
from coreml_pred import Puissance, processus_actifs

PY = sys.executable
ICI = Path(__file__).resolve().parent
SANS_CHARGE = [("torch", "fp16"), ("torch-fp16", "fp16"), ("coreml-gpu", "fp16"), ("coreml-ne", "fp16"),
               ("coreml-gpu", "int8"), ("coreml-gpu", "int4b32"), ("coreml-gpu", "pal4"),
               ("coreml-ne", "int8"), ("coreml-ne", "int4b32"), ("coreml-ne", "pal4"), ("ne-complet", "fp16"),
               ("coreml-cpu", "fp16"), ("coreml-gpu", "int4c"), ("coreml-ne", "int4c")]   # int4c : amendement 4
AVEC_CHARGE = [("torch", "fp16"), ("coreml-gpu", "fp16"), ("coreml-ne", "fp16"), ("ne-complet", "fp16")]
MODES_VISION = {"vision": 0.0, "vision30": 30.0}      # pleine vitesse ; 30 images par seconde
REUSSITE = ("ne-complet", (42, 43))                    # contrôle de réussite du planificateur hors du GPU, 500 épisodes
SENSIBILITE_OMP = [(m, v, k) for m in ("ne-complet", "coreml-cpu")
                   for v, k in (("defaut", 1), ("passif", 1), ("passif", 2), ("defaut", 2))]   # ordre ABBA par moteur
REPETITIONS = 3
SUR_CPU = ("ne-complet", "coreml-cpu")
SEUIL_TROU_S = 7.5                                     # lectures toutes les 5 s : un trou plus long est une pause
SEUIL_FOND = 20.0                                      # % d'un cœur : processus de fond pendant une mesure au repos
P = {"n": 50, "chunk": 50, "n_reussite": 500, "repos": ("60", "30"), "chauffe_vision": 10, "mesure_vision": 35,
     "essai": False}
ESSAI = {"n": 10, "chunk": 10, "n_reussite": 10, "repos": ("10", "5"), "chauffe_vision": 3, "mesure_vision": 8,
         "essai": True}


def nom(moteur, codage, mode=None):
    if moteur.startswith("seule"):
        return moteur
    base = moteur if codage == "fp16" or moteur.startswith("torch") else f"{moteur}-{codage}"
    return base + (f"-{mode}" if mode else "")


def configurations():
    c = [(m, k, None) for m, k in SANS_CHARGE]
    for mode in MODES_VISION:
        c += [(m, k, mode) for m, k in AVEC_CHARGE]
    return c + [(f"seule-{mode}", "", mode) for mode in MODES_VISION]


def dire(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def git_tete():
    return subprocess.run(["git", "-C", str(ICI), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def ecrire_json(f, d):
    tmp = f.with_name(f.name + ".tmp")
    tmp.write_text(json.dumps(d, indent=1))
    os.replace(tmp, f)                                 # jamais de fichier partiel, même sur SIGTERM


def probleme(dossier, g=42):
    """Motif de refaire le run, ou None : fil de lecture gelé plus de 7,5 s (pause de la garde ou du régulateur de
    batterie), ou un autre calcul run_grid vu pendant le run (pgrep exclut le run lui-même : 0 attendu)."""
    t = json.loads((dossier / f"fp32_s{g}.temps.json").read_text())
    ts = [x["t"] for x in t["lectures"] if "t" in x]
    if any(b - a > SEUIL_TROU_S for a, b in zip(ts, ts[1:])):
        return "pause pendant le run"
    if not P["essai"] and max((x.get("processus_run_grid", 0) for x in t["lectures"]), default=0) > 0:   # essai : EXP-011 tourne
        return "autre calcul run_grid pendant le run"
    return None


def pid_calcul(dossier, p, attente=60):
    """PID du processus python de run_grid qui écrit dans dossier, ou None s'il a déjà fini."""
    motif = f"^[^ ]*[Pp]ython[^ ]* ([^ ]+ )*run_grid\\.py --out {dossier}( |$)"
    for _ in range(attente * 2):
        out = subprocess.run(["pgrep", "-f", motif], capture_output=True, text=True).stdout.split()
        if out:
            return int(out[0])
        if p.poll() is not None:
            return None
        time.sleep(0.5)
    return None


def gele(pid):
    st = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return st.startswith("T")


def charge_vision(racine, etiquette, ips):
    f = racine / f"exp013-vision-{etiquette}.jsonl"
    f.unlink(missing_ok=True)                          # fichier neuf : jamais de mélange avec un lancement interrompu
    return subprocess.Popen([PY, "charge_vision.py", str(f), str(4 * 3600), str(ips)]), f


def attendre_pret(f, p, attente=180):
    for _ in range(attente * 2):
        if p.poll() is not None:
            raise SystemExit(f"[arrêt] la charge de vision s'est arrêtée au démarrage ({f.name})")
        if f.exists() and f.read_text().strip():
            return
        time.sleep(0.5)
    raise SystemExit(f"[arrêt] charge de vision pas prête après {attente} s ({f.name})")


def arreter(p):
    if p is None or p.poll() is not None:
        return
    p.send_signal(signal.SIGCONT)
    p.send_signal(signal.SIGTERM)
    try:
        p.wait(timeout=30)
    except subprocess.TimeoutExpired:
        p.kill()


def effacer(racine, dossier, graines):
    for g in graines:
        for suffixe in (".npz", ".temps.json"):
            (racine / dossier / f"fp32_s{g}{suffixe}").unlink(missing_ok=True)


def run(racine, dossier, moteur, codage, n=None, graines=(42,), mode=None, etiquette=None, env=None, controle=True):
    """Un run de run_grid, refait (3 essais) s'il a été gelé, partagé ou si sa vision s'est arrêtée. mode : charge de
    vision (None, "vision", "vision30"), démarrée après la mise en route du run (signal GO)."""
    n = n or P["n"]
    if all((racine / dossier / f"fp32_s{g}.npz").exists() for g in graines):
        raison = next((probleme(racine / dossier, g) for g in graines if probleme(racine / dossier, g)), None) if controle else None
        if raison is None:
            dire(f"[saut] {dossier}")
            return
        dire(f"[refait] {dossier} (reprise) : {raison}")
        effacer(racine, dossier, graines)
    (racine / dossier).mkdir(parents=True, exist_ok=True)
    go = racine / dossier / "go"
    cmd = [PY, "run_grid.py", "--out", str(racine / dossier), "--n", str(n), "--chunk", str(min(P["chunk"], n)),
           "--specs", "fp32", "--seeds", *map(str, graines), "--moteur", moteur, "--energie-puce"] + \
        ([] if moteur.startswith("torch") else ["--codage", codage]) + (["--device", "cpu"] if moteur in SUR_CPU else []) + \
        (["--go", str(go)] if mode else [])
    for essai in (1, 2, 3):
        dire(f"[run] {dossier}" + (f" (essai {essai})" if essai > 1 else ""))
        for f in (go, go.with_name("go.pret")):
            f.unlink(missing_ok=True)
        p = subprocess.Popen(["caffeinate", "-d", "-i"] + cmd, stdout=subprocess.DEVNULL, env={**os.environ, **(env or {})})
        vision, vision_gelee, pid, raison = None, False, None, None
        try:
            pid = pid_calcul(racine / dossier, p)
            if mode:                                   # le run est chargé et mis en route : la vision démarre, puis GO
                while not go.with_name("go.pret").exists():
                    if p.poll() is not None:
                        raise SystemExit(f"[arrêt] {dossier} : run_grid arrêté avant la mise en route (code {p.returncode})")
                    time.sleep(0.5)
                vision, fj = charge_vision(racine, etiquette, MODES_VISION[mode])
                attendre_pret(fj, vision)
                time.sleep(P["chauffe_vision"])
                go.write_text(str(time.time()))
            while p.poll() is None:                    # garde : si le calcul est gelé, la vision l'est aussi
                if pid is not None and vision is not None:
                    g = gele(pid)
                    if g != vision_gelee:
                        vision.send_signal(signal.SIGSTOP if g else signal.SIGCONT)
                        vision_gelee = g
                    if vision.poll() is not None:
                        raison = "la charge de vision s'est arrêtée pendant le run"
                time.sleep(1)
        finally:
            if vision_gelee:
                vision.send_signal(signal.SIGCONT)
            if p.poll() is None:                       # arrêt de vitesse.py : le calcul ne reste pas orphelin
                if pid is not None:
                    subprocess.run(["kill", "-CONT", str(pid)])
                    subprocess.run(["kill", "-TERM", str(pid)])
                p.terminate()
            arreter(vision)                            # la vision écrit ses dernières images en s'arrêtant
        if p.returncode != 0:
            effacer(racine, dossier, graines)
            raise SystemExit(f"[arrêt] {dossier} : run_grid a échoué (code {p.returncode})")
        if raison is None and controle:
            raison = next((probleme(racine / dossier, g) for g in graines if probleme(racine / dossier, g)), None)
        if raison is None:
            return
        dire(f"[refait] {dossier} : {raison}")
        effacer(racine, dossier, graines)
    raise SystemExit(f"[arrêt] {dossier} : 3 essais refaits")


def vision_seule(racine, mode, r, compteur, sha):
    f = racine / f"exp013-vision-seule-{mode}-r{r}.json"
    if f.exists():
        dire(f"[saut] vision seule {mode} r{r}")
        return
    dire(f"[vision seule] {mode} r{r}")
    p, fj = charge_vision(racine, f"seule-{mode}-r{r}", MODES_VISION[mode])
    try:
        attendre_pret(fj, p)
        time.sleep(P["chauffe_vision"])                # chauffe, hors mesure
        e0 = compteur.lire()
        time.sleep(P["mesure_vision"])
        e1 = compteur.lire()
        if p.poll() is not None:
            raise SystemExit(f"[arrêt] la vision seule {mode} r{r} s'est arrêtée pendant la mesure")
    finally:
        arreter(p)                                     # la vision écrit ses dernières images en s'arrêtant
    j, dt = compteur.ecart(e0, e1)
    ecrire_json(f, {"mode": mode, "ips_cible": MODES_VISION[mode], "debut": e0[0], "fin": e1[0], "duree_s": dt,
                    "joules": j, "git": sha})


def fond(j):
    """Processus hors de l'expérience au-dessus de 20 % d'un cœur au début ou à la fin d'une mesure au repos."""
    return [p for p in j.get("processus_debut", []) + j.get("processus_fin", [])
            if p[0] > SEUIL_FOND and not p[1].lower().startswith(("python", "caffeinate", "ps"))]


def repos(racine, etiquette):
    if list(racine.glob(f"exp013-repos-{etiquette}-*.json")):
        return
    for essai in (1, 2):                               # deux essais au plus : la garde ne doit pas croire à un arrêt
        subprocess.run([PY, "repos_puce.py", str(racine), etiquette, *P["repos"]], check=True)
        f = sorted(racine.glob(f"exp013-repos-{etiquette}-*.json"))[-1]
        j = json.loads(f.read_text())
        if P["essai"] or (j["calculs_run_grid_vus"] == 0 and not fond(j)):
            return
        if essai == 2 and j["calculs_run_grid_vus"] == 0:
            dire(f"[repos] {etiquette} : processus de fond encore là ({fond(j)}), mesure gardée et signalée")
            return
        if essai == 2:
            raise SystemExit(f"[arrêt] mesure au repos {etiquette} : un calcul ou une vision tourne encore")
        dire(f"[repos] {etiquette} : calcul, vision ou processus de fond ({fond(j)}), mesure refaite")
        f.rename(f.with_name(f.name.replace(".json", ".contamine")))


def identiques(fa, fb):
    a, b = np.load(fa), np.load(fb)
    if set(a.files) != set(b.files):
        return False
    return all(np.array_equal(a[k], b[k], equal_nan=a[k].dtype.kind == "f") for k in a.files if k not in ("git", "git_sale"))


def ordres_contrebalances(n, rng, configs):
    """Répétition 1 au hasard, 2 à l'envers, 3 au hasard ; dans chacune, la vision seule ni en tête, ni en dernier, ni
    côte à côte."""
    def contraindre(o):
        seule = [configs[v][0].startswith("seule") for v in o]
        s = [v for v, x in zip(o, seule) if x]
        a = [v for v, x in zip(o, seule) if not x]
        assert len(s) == 2 and len(o) >= 5
        p1, p2 = [i for i, x in enumerate(seule) if x]
        p1 = max(p1, 1)
        p2 = min(max(p2, p1 + 2), len(o) - 2)
        p1 = min(p1, p2 - 2)
        a.insert(p1, s[0])
        a.insert(p2, s[1])
        return a
    o1 = contraindre(rng.permutation(n).tolist())
    o2 = contraindre(o1[::-1])
    o3 = contraindre(rng.permutation(n).tolist())
    return [o1, o2, o3]


def etat_machine(racine, moment, sha):
    """pmset (mode d'énergie, chargeur), thermique, batterie, fils de torch, OpenMP, processus actifs, canaux IOReport."""
    import torch
    c = energie_puce.Compteur()
    e0 = c.lire()
    time.sleep(1)
    canaux, _ = c.ecart(e0, c.lire())
    d = {"moment": moment, "t": time.time(), "git": sha,
         "pmset": subprocess.run(["pmset", "-g"], capture_output=True, text=True).stdout,
         "pmset_custom": subprocess.run(["pmset", "-g", "custom"], capture_output=True, text=True).stdout,
         "thermique": subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True).stdout,
         "batterie": Puissance.lire(), "fils_torch": torch.get_num_threads(),
         "parallelisme_torch": torch.__config__.parallel_info(),
         "omp": {k: v for k, v in os.environ.items() if k.startswith("OMP_")}, "processus": processus_actifs(8),
         "canaux_ioreport": sorted(canaux), "macos": subprocess.run(["sw_vers"], capture_output=True, text=True).stdout}
    ecrire_json(racine / f"exp013-etat-{moment}-{int(d['t'])}.json", d)


def autre_commit(racine, sha):
    """Fichiers d'EXP-013 écrits par un autre commit (une reprise les mélangerait)."""
    ko = []
    for f in racine.glob("exp013-*.json"):
        try:
            g = json.loads(f.read_text()).get("git")
        except Exception:  # noqa: BLE001
            g = "illisible"
        if g not in (None, sha):
            ko.append(f.name)
    for m in racine.glob("exp013-*/meta.json"):
        if json.loads(m.read_text()).get("git") != sha:
            ko.append(m.parent.name)
    return sorted(ko)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--racine", required=True)
    ap.add_argument("--essai", action="store_true", help="chemin complet en court, dans un dossier de travail")
    a = ap.parse_args()
    racine = Path(a.racine).resolve()
    if a.essai:
        P.update(ESSAI)
        assert "resultats" not in racine.parts, "--essai : jamais dans resultats/"
        racine.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, lambda *x: sys.exit(143))   # les blocs finally arrêtent la vision et le calcul
    if not a.essai:
        autres = subprocess.run(["pgrep", "-f", "(charge_vision|regulateur_batterie)\\.py"], capture_output=True, text=True).stdout.split()
        if autres:
            raise SystemExit(f"[arrêt] une charge de vision ou le régulateur de batterie tourne déjà : {autres}")
    sha = git_tete()
    ko = autre_commit(racine, sha)
    if ko:
        raise SystemExit(f"[arrêt] fichiers d'EXP-013 d'un autre commit (tout recommencer) : {ko[:10]}")
    configs = configurations()
    f_ordre = racine / "exp013-ordre.json"
    ordres = ordres_contrebalances(len(configs), np.random.default_rng(13), configs)
    if f_ordre.exists():                               # reprise : même plan, sinon arrêt
        o = json.loads(f_ordre.read_text())
        if o["configs"] != [nom(*c) for c in configs] or o["ordres"] != ordres:
            raise SystemExit("[arrêt] exp013-ordre.json décrit un autre plan que le code")
    else:
        ecrire_json(f_ordre, {"configs": [nom(*c) for c in configs], "ordres": ordres, "git": sha})
    etat_machine(racine, "debut", sha)
    compteur = energie_puce.Compteur()
    for r, ordre in enumerate(ordres, start=1):
        repos(racine, f"r{r}-avant" if r == 1 else f"r{r - 1}-r{r}")
        for i in ordre:
            m, c, mode = configs[i]
            if m.startswith("seule"):
                vision_seule(racine, mode, r, compteur, sha)
                continue
            run(racine, f"exp013-{nom(m, c, mode)}-r{r}", m, c, mode=mode, etiquette=f"{nom(m, c, mode)}-r{r}")
    repos(racine, f"r{REPETITIONS}-apres")
    for m, variante, k in SENSIBILITE_OMP:            # entre le repos d'après la répétition 3 et le suivant
        run(racine, f"exp013-sens-{m}-omp-{variante}-{k}", m, "fp16",
            env={"OMP_WAIT_POLICY": "PASSIVE"} if variante == "passif" else None)
    repos(racine, "sens-apres")
    m, graines = REUSSITE
    run(racine, f"exp013-{m}-reussite", m, "fp16", n=P["n_reussite"], graines=graines, controle=False)
    run(racine, "exp013-fin", "torch", "fp16")       # la garde de batterie sort ici (son dernier fichier attendu)
    f_lot = racine / "exp013-balayage-lot.json"
    if not f_lot.exists():                             # appels isolés du prédicteur, une vingtaine de minutes
        dire("[balayage] taille du lot")
        b = subprocess.run(["caffeinate", "-d", "-i", PY, "balayage_lot.py", str(f_lot)] + (["--essai"] if a.essai else []))
        if b.returncode != 0:
            dire(f"[balayage] ÉCHEC (code {b.returncode}) : le reste de l'expérience est gardé, contrôles finaux faits")
    etat_machine(racine, "fin", sha)
    ref = racine / "pilote-exp012" / "torch" / "fp32_s42.npz" if not a.essai else racine / "exp013-torch-r1" / "fp32_s42.npz"
    torch_runs = [racine / f"exp013-torch-r{r}" / "fp32_s42.npz" for r in range(1, REPETITIONS + 1)] + [racine / "exp013-fin" / "fp32_s42.npz"]
    ko = [str(f.parent.name) for f in torch_runs if not identiques(f, ref)]
    dire("[contrôle] runs torch identiques au pilote d'EXP-012" if not ko else f"[arrêt] runs torch DIFFÉRENTS du pilote : {ko}")
    sys.exit(1 if ko else 0)


if __name__ == "__main__":
    main()
