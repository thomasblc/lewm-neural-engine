"""Erreur de reconstruction des couches, rapportée à celle de tout mettre à zéro : somme sur les couches de
trace((W-Q) H (W-Q)^T), divisée par la même somme pour Q = 0, avec H la statistique de calibration d'EXP-009. Pour
RTN et GPTQ (mêmes grille et échelles), et pour RTN et GPTQ avec l'échelle de chaque couche réduite au meilleur facteur
pour RTN (gptq.facteur_reduit, 0,20 à 1,00 par pas de 0,05), comme les versions ":clip" de quant.py. Donne aussi la part des poids
que RTN met à zéro. Aucun épisode d'évaluation n'est utilisé. Chiffres cités dans PREREG-EXP011.md.

Usage (depuis code-exp011, environnement de le-wm, CPU) :
    python erreur_reconstruction.py --tache pusht --groupe all --bits 4 3 2
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("STABLEWM_HOME", "/Users/thomasblanc/Documents/PRO/world-models-demos/stable-wm")
import torch

import gptq
from quant import GROUPES

R = Path(__file__).resolve().parent.parent / "resultats"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tache", required=True)
    ap.add_argument("--groupe", default="all")
    ap.add_argument("--bits", nargs="+", type=int, default=[4, 3, 2])
    ap.add_argument("--hors-echantillon", action="store_true",
                    help="échelles et GPTQ réglés sur la moitié a de la calibration, erreur mesurée sur la moitié b")
    a = ap.parse_args()
    torch.set_num_threads(4)
    import stable_worldmodel as swm
    cal = torch.load(R / f"exp009-calib-{a.tache}" / "calib.pt", map_location="cpu")
    assert cal["tache"] == a.tache, "calibration d'une autre tâche"
    m = swm.wm.utils.load_pretrained(f"{a.tache}/lewm")
    couches = {n: x for n, x in m.named_modules()
               if n.split(".")[0] in GROUPES[a.groupe] and isinstance(x, (torch.nn.Linear, torch.nn.Conv2d))}
    for bits in a.bits:
        qmax = 2 ** (bits - 1) - 1
        e = {"rtn": 0.0, "gptq": 0.0, "rtn_echelle_reduite": 0.0, "gptq_echelle_reduite": 0.0, "zero": 0.0}
        zeros = total = 0
        for n, x in couches.items():
            w = x.weight.detach().reshape(x.weight.shape[0], -1).double()
            if a.hors_echantillon:                        # réglage sur a, mesure sur b
                h, hm = cal["H_a"][n] / cal["n_a"][n], cal["H_b"][n] / cal["n_b"][n]
            else:
                h = hm = cal["H"][n] / cal["n"][n]
            r = gptq.rtn(w, bits)
            e["rtn"] += gptq.perte(w, r, hm)
            e["zero"] += gptq.perte(w, torch.zeros_like(w), hm)
            e["gptq"] += gptq.perte(w, gptq.gptq_couche(w, h, bits), hm)
            s0 = w.abs().amax(1, keepdim=True).clamp_min(1e-12) / qmax
            s = s0 * gptq.facteur_reduit(w, h, bits)      # même choix que quant.py (":clip")
            e["rtn_echelle_reduite"] += gptq.perte(w, (w / s).round().clamp(-qmax, qmax) * s, hm)
            e["gptq_echelle_reduite"] += gptq.perte(w, gptq.gptq_couche(w, h, bits, s=s), hm)
            zeros, total = zeros + int((r == 0).sum()), total + r.numel()
        print(f"{a.tache} {a.groupe} ({len(couches)} couches), {bits} bits{', hors échantillon' if a.hors_echantillon else ''} : erreur rapportée à tout mettre à zéro : "
              f"RTN {e['rtn'] / e['zero']:.3f}, GPTQ {e['gptq'] / e['zero']:.3f}, RTN à échelle réduite "
              f"{e['rtn_echelle_reduite'] / e['zero']:.3f}, GPTQ à échelle réduite {e['gptq_echelle_reduite'] / e['zero']:.3f} ; "
              f"poids mis à zéro par RTN {100 * zeros / total:.0f} %", flush=True)


if __name__ == "__main__":
    main()
