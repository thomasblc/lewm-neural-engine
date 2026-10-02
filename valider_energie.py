"""Validation des compteurs d'énergie IOReport contre la batterie (EXP-013, amendements 3 point 19, 6 point 15 et 7
point 7) : obligatoire avant qu'un chiffre d'énergie entre dans le papier. Chargeur débranché, Mac au repos sinon,
luminosité inchangée, personne ne touche au Mac pendant la mesure (environ 25 min).

Segments, dans cet ordre, chacun mesuré SEGMENT secondes après sa mise en route : repos, torch (PyTorch float32 sur le
GPU : calibration), coreml-gpu, ne-complet, ne-complet avec la vision ResNet-50 à 30 images/s, repos. Une charge est un
run_grid de PushT (500 épisodes, graine 42, sortie jetable) ; la mesure commence 30 s après le premier plan (la vision :
20 s après sa ligne « prête »), et le run est arrêté à la fin du segment.

Mesures, toutes les secondes : puissance tirée de la batterie (- InstantAmperage x Voltage), capacité brute
(AppleRawCurrentCapacity, recoupement), chargeur (ExternalConnected), énergie IOReport par canal. Puissance de la puce
P_p : somme des canaux PUCE d'EXP-013 (sans l'affichage).

Règles fixées avant la mesure (PROTOCOLE-VALIDATION-ENERGIE.md) :
- repos = moyenne des deux segments de repos (dérive entre les deux rapportée) ; Δ = puissance du segment moins repos ;
- facteur c = ΔP_b(torch) / ΔP_p(torch) (pertes de conversion proportionnelles à la charge) ;
- V1, pour coreml-gpu, ne-complet, ne-complet-vision30 : |c ΔP_p - ΔP_b| / ΔP_b au plus 10 % ;
- V2, rapports (les affirmations du papier sont des rapports d'énergie au-dessus du repos), pour les paires
  (coreml-gpu, ne-complet), (torch, ne-complet), (torch, coreml-gpu) : |(ΔP_p,i / ΔP_p,j) / (ΔP_b,i / ΔP_b,j) - 1| au plus
  10 % (indépendant de c) ;
- validée si V1 et V2 passent toutes, segments valides (chargeur débranché tout du long, aucun échantillon manquant à
  plus de 5 %). Sinon : aucun chiffre d'énergie dans le papier. La validation ne couvre ni le partage par composant, ni
  l'énergie marginale sous charge de vision répartie par moteur.

Usage (depuis code-exp015, environnement de le-wm, aucun autre calcul en cours) :
    python valider_energie.py ../resultats/validation-energie-AAAA-MM-JJ [--segment 180] [--essai]
--essai : segments de 20 s, chargeur permis (contrôle du code, pas une mesure).
"""
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import energie_puce

SOC = ("CPU Energy", "GPU Energy", "ANE", "DRAM", "AMCC", "DCS", "FAB", "AFR", "ISP", "AVE", "MSR", "DISP", "DISPEXT")
PUCE = tuple(k for k in SOC if k not in ("DISP", "DISPEXT"))        # comme analyse_exp013.PUCE
PY = sys.executable
CHARGES = {"torch": ["--moteur", "torch"], "coreml-gpu": ["--moteur", "coreml-gpu"],
           "ne-complet": ["--moteur", "ne-complet", "--device", "cpu"], "ne-complet-vision30": ["--moteur", "ne-complet", "--device", "cpu"]}
ORDRE = ("repos-1", "torch", "coreml-gpu", "ne-complet", "ne-complet-vision30", "repos-2")
SEUIL = 0.10


def batterie():
    out = subprocess.run(["ioreg", "-rn", "AppleSmartBattery"], capture_output=True, text=True, timeout=10).stdout
    v = {}
    for k in ("InstantAmperage", "Voltage", "AppleRawCurrentCapacity"):
        m = re.search(rf'"{k}" = (\d+)', out)
        v[k] = int(m.group(1)) if m else None
    if v["InstantAmperage"] is not None and v["InstantAmperage"] >= 2 ** 63:
        v["InstantAmperage"] -= 2 ** 64                               # entier signé sur 64 bits
    m = re.search(r'"ExternalConnected" = (Yes|No)', out)
    v["chargeur"] = None if m is None else m.group(1) == "Yes"
    return v


class Echantillonneur(threading.Thread):
    """Un échantillon par seconde : batterie et énergie IOReport par canal depuis l'échantillon précédent."""

    def __init__(self, f):
        super().__init__(daemon=True)
        self.f, self.arret, self.segment = f, threading.Event(), None
        self.c = energie_puce.Compteur()

    def run(self):
        e0 = self.c.lire()
        with open(self.f, "a") as out:
            while not self.arret.wait(1.0):
                e1 = self.c.lire()
                j, duree = energie_puce.Compteur.ecart(e0, e1)
                e0 = e1
                out.write(json.dumps({"t": e1[0], "duree_s": duree, "segment": self.segment, "batterie": batterie(),
                                      "joules": {k: j.get(k, 0.0) for k in SOC}}) + "\n")
                out.flush()


def demarrer_charge(nom, dossier):
    cmd = ["caffeinate", "-d", "-i", PY, "run_grid.py", "--out", str(dossier), "--n", "500", "--chunk", "50", "--specs", "fp32",
           "--seeds", "42", *CHARGES[nom]] + ([] if nom == "torch" else ["--codage", "fp16"])
    # sortie non tamponnée : sinon les lignes « CEM solve time » n'arrivent qu'à la fin du run (essai du 2 octobre)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                         env={**os.environ, "PYTHONUNBUFFERED": "1"})
    premier = threading.Event()

    def lire():
        for ligne in p.stdout:
            if "CEM solve time" in ligne:
                premier.set()
    threading.Thread(target=lire, daemon=True).start()
    return p, premier


def arreter(p):
    if p is not None and p.poll() is None:
        p.send_signal(signal.SIGTERM)
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()


def segment(ech, nom, duree, racine, journal):
    """Met la charge en route, puis marque les échantillons du segment pendant duree secondes."""
    p = vision = None
    try:
        if nom in CHARGES:
            p, premier = demarrer_charge(nom, racine / "runs" / nom)
            if not premier.wait(600):
                raise SystemExit(f"[arrêt] {nom} : aucun plan après 10 min")
            time.sleep(30)
            if nom.endswith("vision30"):
                fv = racine / "runs" / "vision30.jsonl"
                fv.unlink(missing_ok=True)
                vision = subprocess.Popen([PY, "charge_vision.py", str(fv), str(3600), "30"])
                t0 = time.time()
                while not (fv.exists() and fv.read_text().strip()):
                    if vision.poll() is not None or time.time() - t0 > 180:
                        raise SystemExit("[arrêt] la vision n'a pas démarré")
                    time.sleep(0.5)
                time.sleep(20)
            if p.poll() is not None:
                raise SystemExit(f"[arrêt] {nom} : run_grid arrêté avant la mesure (code {p.returncode})")
        journal(f"[mesure] {nom} : {duree} s à partir de {time.strftime('%H:%M:%S')}")
        ech.segment = nom
        time.sleep(duree)
        ech.segment = None
        if p is not None and p.poll() is not None:
            raise SystemExit(f"[arrêt] {nom} : run_grid arrêté pendant la mesure (code {p.returncode})")
    finally:
        ech.segment = None
        arreter(vision)
        arreter(p)


def resumer(f, duree, essai):
    L = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    seg = {}
    for nom in ORDRE:
        x = [l for l in L if l["segment"] == nom]
        b = [l["batterie"] for l in x]
        ok = [y for y in b if y["InstantAmperage"] is not None and y["Voltage"] is not None]
        pb = [-y["InstantAmperage"] * y["Voltage"] / 1e6 for y in ok]          # mA x mV = µW ; décharge positive
        d = sum(l["duree_s"] for l in x)
        cap = [y["AppleRawCurrentCapacity"] for y in b if y["AppleRawCurrentCapacity"] is not None]
        tension = sum(y["Voltage"] for y in ok) / max(1, len(ok)) / 1000
        seg[nom] = {"echantillons": len(x), "duree_s": d, "P_b": sum(pb) / max(1, len(pb)),
                    "P_b_capacite": (cap[0] - cap[-1]) * tension * 3.6 / d if len(cap) > 1 and d > 0 else None,   # mAh x V = 3,6 J
                    "P_p": sum(sum(l["joules"][k] for k in PUCE) for l in x) / d if d > 0 else None,
                    "P_affichage": sum(l["joules"]["DISP"] + l["joules"]["DISPEXT"] for l in x) / d if d > 0 else None,
                    "chargeur_branche": any(y["chargeur"] for y in b),
                    "valide": len(x) >= 0.95 * duree and len(ok) >= 0.95 * duree and (essai or not any(y["chargeur"] for y in b))}
    repos = {k: (seg["repos-1"][k] + seg["repos-2"][k]) / 2 for k in ("P_b", "P_p")}
    D = {n: {k: seg[n][k] - repos[k] for k in ("P_b", "P_p")} for n in CHARGES}
    c = D["torch"]["P_b"] / D["torch"]["P_p"]
    v1 = {n: abs(c * D[n]["P_p"] - D[n]["P_b"]) / D[n]["P_b"] for n in ("coreml-gpu", "ne-complet", "ne-complet-vision30")}
    v2 = {f"{i} / {j}": abs((D[i]["P_p"] / D[j]["P_p"]) / (D[i]["P_b"] / D[j]["P_b"]) - 1)
          for i, j in (("coreml-gpu", "ne-complet"), ("torch", "ne-complet"), ("torch", "coreml-gpu"))}
    # un écart au repos nul ou négatif (charge qui consommerait moins que le repos) est un échec, pas un écart relatif
    positifs = all(D[n]["P_b"] > 0 and D[n]["P_p"] > 0 for n in CHARGES)
    valides = all(s["valide"] for s in seg.values()) and positifs
    return {"segments": seg, "repos": repos, "derive_repos": {k: seg["repos-2"][k] - seg["repos-1"][k] for k in ("P_b", "P_p")},
            "au_dessus_du_repos": D, "ecarts_positifs": positifs, "facteur_c": c, "V1_ecart_relatif": v1, "V2_ecart_relatif": v2, "segments_valides": valides,
            "seuil": SEUIL, "validee": bool(valides and all(e <= SEUIL for e in [*v1.values(), *v2.values()])) and not essai}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sortie")
    ap.add_argument("--segment", type=int, default=180)
    ap.add_argument("--essai", action="store_true")
    a = ap.parse_args()
    duree = 20 if a.essai else a.segment
    racine = Path(a.sortie)
    racine.mkdir(parents=True, exist_ok=True)
    jf = racine / "journal.txt"
    journal = lambda m: (print(m, flush=True), open(jf, "a").write(m + "\n"))
    b = batterie()
    if not a.essai and (b["chargeur"] or b["chargeur"] is None):
        raise SystemExit("[arrêt] chargeur branché : le débrancher avant la mesure")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    sale = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip())
    if sale and not a.essai:
        raise SystemExit("[arrêt] arbre git modifié")
    journal(f"[info] commit {git[:7]}, batterie {b}, segments de {duree} s, début {time.strftime('%H:%M:%S')}")
    f = racine / "echantillons.jsonl"
    f.unlink(missing_ok=True)
    ech = Echantillonneur(f)
    ech.start()
    caf = subprocess.Popen(["caffeinate", "-d", "-i", "-w", str(os.getpid())])   # écran allumé, pas de veille
    try:
        for nom in ORDRE:
            segment(ech, nom, duree, racine, journal)
    finally:
        ech.arret.set()
        ech.join(5)
        caf.terminate()
    r = resumer(f, duree, a.essai)
    r.update({"git": git, "git_sale": sale, "essai": a.essai, "segment_s": duree})
    (racine / "resultats.json").write_text(json.dumps(r, indent=1, ensure_ascii=False))
    journal(json.dumps({k: r[k] for k in ("facteur_c", "V1_ecart_relatif", "V2_ecart_relatif", "segments_valides", "validee")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
