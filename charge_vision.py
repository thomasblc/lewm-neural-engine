"""EXP-013 : charge de vision sur le GPU, pour mesurer la planification pendant que le GPU voit (un robot voit et agit en
même temps). ResNet-50 à poids aléatoires (le temps de calcul ne dépend pas des poids ; rien n'est téléchargé), images
224 x 224, lot de 1, en boucle sur le GPU (MPS, float16), aussi vite que possible. Écrit toutes les 5 s le nombre
d'images traitées dans un fichier JSON lignes ; s'arrête sur SIGTERM ou après DUREE_MAX secondes.

Usage :
    python charge_vision.py SORTIE.jsonl [DUREE_MAX]
"""
import json
import signal
import sys
import time
from pathlib import Path

import torch
import torchvision

sortie, duree_max = Path(sys.argv[1]), float(sys.argv[2]) if len(sys.argv) > 2 else 4 * 3600
arret = False


def stop(*a):
    global arret
    arret = True


signal.signal(signal.SIGTERM, stop)
torch.manual_seed(0)
m = torchvision.models.resnet50(weights=None).eval().to("mps", torch.float16)
x = torch.randn(1, 3, 224, 224, device="mps", dtype=torch.float16)
debut, t0, n = time.time(), time.time(), 0
with torch.no_grad(), sortie.open("a") as f:
    while not arret and time.time() - debut < duree_max:
        m(x)
        torch.mps.synchronize()
        n += 1
        if time.time() - t0 >= 5:
            t1 = time.time()
            f.write(json.dumps({"t0": t0, "t1": t1, "images": n}) + "\n")
            f.flush()
            t0, n = t1, 0
