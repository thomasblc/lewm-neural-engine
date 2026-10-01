"""EXP-013 (PREREG-EXP013.md) : enchaîne les runs de vitesse. Par répétition (3) : énergie au repos par composant, puis les
configurations dans un ordre tiré au hasard (default_rng(13)), chacune un appel de run_grid.py sur les 50 épisodes de la
sonde, graine 42, avec l'énergie par composant ; les configurations « avec charge » démarrent charge_vision.py 30 s avant
le run et l'arrêtent après. Une mesure de la charge seule (60 s) par répétition. Run torch final, puis contrôles bloquants
(runs torch identiques entre eux et au pilote d'EXP-012). Reprend là où il s'est arrêté : un run dont le fichier existe
est sauté.

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

PY = sys.executable
SANS_CHARGE = [("torch", "fp16"), ("torch-fp16", "fp16"), ("coreml-gpu", "fp16"), ("coreml-ne", "fp16"),
               ("coreml-gpu", "int8"), ("coreml-gpu", "int4b32"), ("coreml-gpu", "pal4"),
               ("coreml-ne", "int8"), ("coreml-ne", "int4b32"), ("coreml-ne", "pal4"), ("ne-complet", "fp16")]
AVEC_CHARGE = [("torch", "fp16"), ("coreml-gpu", "fp16"), ("coreml-ne", "fp16"), ("ne-complet", "fp16")]
REUSSITE = ("ne-complet", (42, 43))              # contrôle de réussite du planificateur hors du GPU, 500 épisodes
REPETITIONS = 3


def nom(moteur, codage, charge):
    if moteur == "charge-seule":
        return "charge-seule"
    return f"{moteur}{'' if codage == 'fp16' or moteur.startswith('torch') else '-' + codage}{'-vision' if charge else ''}"


def dire(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def run(racine, dossier, moteur, codage, n=50, graines=(42,)):
    if all((racine / dossier / f"fp32_s{g}.npz").exists() for g in graines):
        dire(f"[saut] {dossier}")
        return
    cmd = [PY, "run_grid.py", "--out", str(racine / dossier), "--n", str(n), "--chunk", "50", "--specs", "fp32",
           "--seeds", *map(str, graines), "--moteur", moteur, "--energie-puce"] + \
        ([] if moteur.startswith("torch") else ["--codage", codage]) + (["--device", "cpu"] if moteur == "ne-complet" else [])
    dire(f"[run] {dossier}")
    subprocess.run(["caffeinate", "-i"] + cmd, check=True, stdout=subprocess.DEVNULL)


def charge_vision(racine, etiquette):
    return subprocess.Popen([PY, "charge_vision.py", str(racine / f"exp013-vision-{etiquette}.jsonl"), "1800"])


def arreter(p):
    p.send_signal(signal.SIGTERM)
    try:
        p.wait(timeout=30)
    except subprocess.TimeoutExpired:
        p.kill()


def identiques(fa, fb):
    a, b = np.load(fa), np.load(fb)
    if set(a.files) != set(b.files):
        return False
    return all(np.array_equal(a[k], b[k], equal_nan=a[k].dtype.kind == "f") for k in a.files if k not in ("git", "git_sale"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--racine", required=True)
    a = ap.parse_args()
    racine = Path(a.racine).resolve()
    rng = np.random.default_rng(13)
    configs = [(m, c, False) for m, c in SANS_CHARGE] + [(m, c, True) for m, c in AVEC_CHARGE] + [("charge-seule", "", True)]
    ordres = [rng.permutation(len(configs)).tolist() for _ in range(REPETITIONS)]
    (racine / "exp013-ordre.json").write_text(json.dumps({"configs": [nom(*c) for c in configs], "ordres": ordres}, indent=1))
    for r, ordre in enumerate(ordres, start=1):
        # repos : 45 s de stabilisation et 75 s de mesure (moins de 3 min sans calcul : la garde ne croit pas à un arrêt)
        subprocess.run([PY, "repos_puce.py", str(racine), f"r{r}", "75", "45"], check=True)
        for i in ordre:
            m, c, charge = configs[i]
            if m == "charge-seule":                    # vitesse de référence de la vision, placée au hasard dans l'ordre
                if not (racine / f"exp013-vision-seule-r{r}.jsonl").exists():
                    dire(f"[charge seule] r{r}")
                    p = charge_vision(racine, f"seule-r{r}")
                    time.sleep(60)
                    arreter(p)
                continue
            dossier = f"exp013-{nom(m, c, charge)}-r{r}"
            if charge:
                if (racine / dossier / "fp32_s42.npz").exists():
                    dire(f"[saut] {dossier}")
                    continue
                p = charge_vision(racine, f"{nom(m, c, charge)}-r{r}")
                time.sleep(30)
                try:
                    run(racine, dossier, m, c)
                finally:
                    arreter(p)
            else:
                run(racine, dossier, m, c)
    m, graines = REUSSITE
    run(racine, f"exp013-{m}-reussite", m, "fp16", n=500, graines=graines)
    run(racine, "exp013-fin", "torch", "fp16")
    ref = racine / "pilote-exp012" / "torch" / "fp32_s42.npz"
    torch_runs = [racine / f"exp013-torch-r{r}" / "fp32_s42.npz" for r in range(1, REPETITIONS + 1)] + [racine / "exp013-fin" / "fp32_s42.npz"]
    ko = [str(f.parent.name) for f in torch_runs if not identiques(f, ref)]
    dire("[contrôle] runs torch identiques au pilote d'EXP-012" if not ko else f"[arrêt] runs torch DIFFÉRENTS du pilote : {ko}")
    sys.exit(1 if ko else 0)


if __name__ == "__main__":
    main()
