"""EXP-013 (PREREG-EXP013.md, amendement 2) : enchaîne les mesures de vitesse et d'énergie.

- 3 répétitions ; ordre de la répétition 1 tiré au hasard (default_rng(13)), répétition 2 dans l'ordre inverse
  (contrebalancement de la dérive), répétition 3 tirée au hasard ; la vision seule jamais en premier.
- Mesures au repos par composant (repos_puce.py) avant la répétition 1, entre les répétitions et après la 3 ; jamais
  refaites à la reprise.
- Chaque configuration est un appel de run_grid.py sur les 50 épisodes de la sonde, graine 42, avec l'énergie de la
  puce ; avec charge, charge_vision.py tourne pendant le run, à pleine vitesse ou à 30 images par seconde.
- Vision seule (pleine vitesse et 30 images par seconde) : 45 s par mode et par répétition, énergie de la puce mesurée.
- Un run allongé par une pause de la garde de batterie (fil de lecture gelé) est refait, 3 essais au plus.
- Ensuite : contrôle de réussite du planificateur sur le Neural Engine (500 épisodes, 2 graines), balayage de la
  taille du lot (balayage_lot.py), run torch final, contrôles bloquants (runs torch identiques au pilote d'EXP-012).
Reprend là où il s'est arrêté.

Usage (depuis code-exp013, par le lanceur) :
    python vitesse.py --racine ../resultats
"""
import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import energie_puce

PY = sys.executable
SANS_CHARGE = [("torch", "fp16"), ("torch-fp16", "fp16"), ("coreml-gpu", "fp16"), ("coreml-ne", "fp16"),
               ("coreml-gpu", "int8"), ("coreml-gpu", "int4b32"), ("coreml-gpu", "pal4"),
               ("coreml-ne", "int8"), ("coreml-ne", "int4b32"), ("coreml-ne", "pal4"), ("ne-complet", "fp16"),
               ("coreml-cpu", "fp16")]
AVEC_CHARGE = [("torch", "fp16"), ("coreml-gpu", "fp16"), ("coreml-ne", "fp16"), ("ne-complet", "fp16")]
MODES_VISION = {"vision": 0.0, "vision30": 30.0}      # pleine vitesse ; 30 images par seconde
REUSSITE = ("ne-complet", (42, 43))                    # contrôle de réussite du planificateur hors du GPU, 500 épisodes
REPETITIONS = 3
SUR_CPU = ("ne-complet", "coreml-cpu")


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


def pause_pendant(dossier, g=42):
    """Vrai si le fil de lecture du run a été gelé plus de 12,5 s (pause de la garde de batterie)."""
    t = json.loads((dossier / f"fp32_s{g}.temps.json").read_text())
    ts = [x["t"] for x in t["lectures"] if "t" in x]
    return any(b - a > 12.5 for a, b in zip(ts, ts[1:]))


def run(racine, dossier, moteur, codage, n=50, graines=(42,)):
    if all((racine / dossier / f"fp32_s{g}.npz").exists() for g in graines):
        dire(f"[saut] {dossier}")
        return
    cmd = [PY, "run_grid.py", "--out", str(racine / dossier), "--n", str(n), "--chunk", "50", "--specs", "fp32",
           "--seeds", *map(str, graines), "--moteur", moteur, "--energie-puce"] + \
        ([] if moteur.startswith("torch") else ["--codage", codage]) + (["--device", "cpu"] if moteur in SUR_CPU else [])
    for essai in (1, 2, 3):
        dire(f"[run] {dossier}" + (f" (essai {essai})" if essai > 1 else ""))
        subprocess.run(["caffeinate", "-d", "-i"] + cmd, check=True, stdout=subprocess.DEVNULL)
        if not any(pause_pendant(racine / dossier, g) for g in graines):
            return
        dire(f"[pause] {dossier} : pause de la garde pendant le run, run refait")
        for g in graines:
            for suffixe in (".npz", ".temps.json"):
                (racine / dossier / f"fp32_s{g}{suffixe}").unlink(missing_ok=True)
    raise SystemExit(f"[arrêt] {dossier} : pause pendant 3 essais")


def charge_vision(racine, etiquette, ips):
    return subprocess.Popen([PY, "charge_vision.py", str(racine / f"exp013-vision-{etiquette}.jsonl"), str(4 * 3600), str(ips)])


def arreter(p):
    p.send_signal(signal.SIGTERM)
    try:
        p.wait(timeout=30)
    except subprocess.TimeoutExpired:
        p.kill()


def vision_seule(racine, mode, r, compteur):
    f = racine / f"exp013-vision-seule-{mode}-r{r}.json"
    if f.exists():
        dire(f"[saut] vision seule {mode} r{r}")
        return
    dire(f"[vision seule] {mode} r{r}")
    p = charge_vision(racine, f"seule-{mode}-r{r}", MODES_VISION[mode])
    try:
        time.sleep(10)                                 # chauffe, hors mesure
        e0 = compteur.lire()
        time.sleep(35)
        j, dt = compteur.ecart(e0, compteur.lire())
    finally:
        arreter(p)
    f.write_text(json.dumps({"mode": mode, "debut": e0[0], "duree_s": dt, "joules": j}, indent=1))


def repos(racine, etiquette):
    if list(racine.glob(f"exp013-repos-{etiquette}-*.json")):
        return
    # 30 s de stabilisation et 60 s de mesure : moins de 2 min sans calcul, la garde ne croit pas à un arrêt
    subprocess.run([PY, "repos_puce.py", str(racine), etiquette, "60", "30"], check=True)


def identiques(fa, fb):
    a, b = np.load(fa), np.load(fb)
    if set(a.files) != set(b.files):
        return False
    return all(np.array_equal(a[k], b[k], equal_nan=a[k].dtype.kind == "f") for k in a.files if k not in ("git", "git_sale"))


def ordres_contrebalances(n, rng, configs):
    def sans_seule_en_tete(o):
        if configs[o[0]][0].startswith("seule"):
            i = next(k for k, v in enumerate(o) if not configs[v][0].startswith("seule"))
            o[0], o[i] = o[i], o[0]
        return o
    o1 = sans_seule_en_tete(rng.permutation(n).tolist())
    o2 = sans_seule_en_tete(o1[::-1])
    o3 = sans_seule_en_tete(rng.permutation(n).tolist())
    return [o1, o2, o3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--racine", required=True)
    a = ap.parse_args()
    racine = Path(a.racine).resolve()
    configs = configurations()
    f_ordre = racine / "exp013-ordre.json"
    if f_ordre.exists():                               # reprise : même ordre
        ordres = json.loads(f_ordre.read_text())["ordres"]
    else:
        ordres = ordres_contrebalances(len(configs), np.random.default_rng(13), configs)
        f_ordre.write_text(json.dumps({"configs": [nom(*c) for c in configs], "ordres": ordres}, indent=1))
    compteur = energie_puce.Compteur()
    for r, ordre in enumerate(ordres, start=1):
        repos(racine, f"r{r}-avant" if r == 1 else f"r{r - 1}-r{r}")
        for i in ordre:
            m, c, mode = configs[i]
            if m.startswith("seule"):
                vision_seule(racine, mode, r, compteur)
                continue
            dossier = f"exp013-{nom(m, c, mode)}-r{r}"
            if mode is None:
                run(racine, dossier, m, c)
                continue
            if (racine / dossier / "fp32_s42.npz").exists():
                dire(f"[saut] {dossier}")
                continue
            p = charge_vision(racine, f"{nom(m, c, mode)}-r{r}", MODES_VISION[mode])
            try:
                time.sleep(15)                         # chauffe de la vision ; ses 15 s servent de référence appariée
                if p.poll() is not None:
                    raise SystemExit(f"[arrêt] la charge de vision s'est arrêtée avant {dossier}")
                run(racine, dossier, m, c)
            finally:
                arreter(p)
    repos(racine, f"r{REPETITIONS}-apres")
    m, graines = REUSSITE
    run(racine, f"exp013-{m}-reussite", m, "fp16", n=500, graines=graines)
    run(racine, "exp013-fin", "torch", "fp16")       # la garde de batterie sort ici (son dernier fichier attendu)
    if not (racine / "exp013-balayage-lot.json").exists():  # appels isolés du prédicteur, une dizaine de minutes
        dire("[balayage] taille du lot")
        subprocess.run(["caffeinate", "-d", "-i", PY, "balayage_lot.py", str(racine / "exp013-balayage-lot.json")], check=True)
    ref = racine / "pilote-exp012" / "torch" / "fp32_s42.npz"
    torch_runs = [racine / f"exp013-torch-r{r}" / "fp32_s42.npz" for r in range(1, REPETITIONS + 1)] + [racine / "exp013-fin" / "fp32_s42.npz"]
    ko = [str(f.parent.name) for f in torch_runs if not identiques(f, ref)]
    dire("[contrôle] runs torch identiques au pilote d'EXP-012" if not ko else f"[arrêt] runs torch DIFFÉRENTS du pilote : {ko}")
    sys.exit(1 if ko else 0)


if __name__ == "__main__":
    main()
