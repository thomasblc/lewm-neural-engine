"""Erreur de reconstruction des couches, rapportée à celle de tout mettre à zéro : somme sur les couches de
trace((W-Q) H (W-Q)^T), divisée par la même somme pour Q = 0, avec H la statistique de calibration d'EXP-009. Pour
RTN et GPTQ (mêmes grille et échelles), et pour RTN avec l'échelle de chaque couche réduite au meilleur facteur
(0,20 à 1,00 par pas de 0,05, en clair : ce que gagnerait un réglage des échelles). Donne aussi la part des poids
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
        e = {"rtn": 0.0, "gptq": 0.0, "rtn_echelle_reduite": 0.0, "zero": 0.0}
        zeros = total = 0
        for n, x in couches.items():
            w = x.weight.detach().reshape(x.weight.shape[0], -1).double()
            h = cal["H"][n] / cal["n"][n]
            r = gptq.rtn(w, bits)
            e["rtn"] += gptq.perte(w, r, h)
            e["zero"] += gptq.perte(w, torch.zeros_like(w), h)
            e["gptq"] += gptq.perte(w, gptq.gptq_couche(w, h, bits), h)
            s0 = w.abs().amax(1, keepdim=True).clamp_min(1e-12) / qmax
            e["rtn_echelle_reduite"] += min(gptq.perte(w, (w / (s0 * f)).round().clamp(-qmax, qmax) * s0 * f, h)
                                            for f in [i / 20 for i in range(4, 21)])
            zeros, total = zeros + int((r == 0).sum()), total + r.numel()
        print(f"{a.tache} {a.groupe} ({len(couches)} couches), {bits} bits : erreur rapportée à tout mettre à zéro : "
              f"RTN {e['rtn'] / e['zero']:.3f}, GPTQ {e['gptq'] / e['zero']:.3f}, RTN à échelle réduite "
              f"{e['rtn_echelle_reduite'] / e['zero']:.3f} ; poids mis à zéro par RTN {100 * zeros / total:.0f} %", flush=True)


if __name__ == "__main__":
    main()
