"""Vérification bloquante d'EXP-009, avant la grille : sur le vrai modèle et ses statistiques de calibration, GPTQ
doit faire moins d'erreur de reconstruction des couches que RTN (somme sur les couches de trace((W-Q) H (W-Q)^T)),
à chaque nombre de bits, sur la calibration et hors échantillon (GPTQ réglé sur les paquets pairs, erreur
mesurée sur les impairs). Sinon, c'est une erreur de code : arrêt. Affiche aussi la part des couches améliorées et
les couches calibrées avec moins de lignes que d'entrées.

Usage :
    python verifier_gptq.py --tache pusht --calib ../resultats/exp009-calib-pusht/calib.pt
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("STABLEWM_HOME", str(Path(os.environ.get("LEWM_DIR", "/Users/thomasblanc/Documents/PRO/world-models-demos/le-wm")).parent / "stable-wm"))

import torch

import gptq
from quant import GROUPES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tache", required=True)
    ap.add_argument("--calib", required=True)
    a = ap.parse_args()
    import stable_worldmodel as swm
    calib = torch.load(a.calib, map_location="cpu")
    assert calib["tache"] == a.tache, "calibration d'une autre tâche"
    model = swm.wm.utils.load_pretrained(f"{a.tache}/lewm")
    couches = {nom: m for nom, m in model.named_modules()
               if nom.split(".")[0] in GROUPES["all"] and isinstance(m, (torch.nn.Linear, torch.nn.Conv2d))}
    assert set(couches) == set(calib["H"]), "couches de la calibration différentes de celles de quant.py"
    pauvres = [nom for nom, m in couches.items() if calib["n"][nom] < m.weight[0].numel()]
    ok = True
    for bits in (4, 3):
        tg = tr = 0.0
        mieux = 0
        for nom, m in couches.items():
            w = m.weight.detach().reshape(m.weight.shape[0], -1).double()
            h = calib["H"][nom] / calib["n"][nom]
            eg, er = gptq.perte(w, gptq.gptq_couche(w, h, bits), h), gptq.perte(w, gptq.rtn(w, bits), h)
            tg, tr, mieux = tg + eg, tr + er, mieux + (eg < er)
        print(f"[vérification GPTQ] {a.tache}, {bits} bits : erreur totale GPTQ / RTN = {tg / tr:.3f}, "
              f"{mieux} couches améliorées sur {len(couches)}", flush=True)
        ok &= tg < tr
    if calib.get("H_b"):                              # hors échantillon : réglé sur une moitié, mesuré sur l'autre
        for bits in (4, 3):
            tg = tr = 0.0
            for nom, m in couches.items():
                w = m.weight.detach().reshape(m.weight.shape[0], -1).double()
                ha, hb = calib["H_a"][nom] / calib["n_a"][nom], calib["H_b"][nom] / calib["n_b"][nom]
                tg += gptq.perte(w, gptq.gptq_couche(w, ha, bits), hb)
                tr += gptq.perte(w, gptq.rtn(w, bits), hb)
            print(f"[vérification GPTQ] {a.tache}, {bits} bits, hors échantillon : erreur GPTQ / RTN = {tg / tr:.3f}", flush=True)
            for nom in pauvres:                          # couches à calibration mince : rapport couche par couche
                m = couches[nom]
                w = m.weight.detach().reshape(m.weight.shape[0], -1).double()
                ha, hb = calib["H_a"][nom] / calib["n_a"][nom], calib["H_b"][nom] / calib["n_b"][nom]
                rg = gptq.perte(w, gptq.gptq_couche(w, ha, bits), hb) / gptq.perte(w, gptq.rtn(w, bits), hb)
                print(f"[vérification GPTQ]   {nom} ({calib['n'][nom]} lignes), hors échantillon : {rg:.3f}", flush=True)
            ok &= tg < tr
    else:
        print("[vérification GPTQ] une seule moitié de calibration : pas de contrôle hors échantillon", flush=True)
    print(f"[vérification GPTQ] couches calibrées avec moins de lignes que d'entrées : {pauvres or 'aucune'}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
