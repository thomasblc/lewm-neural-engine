"""EXP-013 : charge de vision sur le GPU, pour mesurer la planification pendant que le GPU voit (un robot voit et agit en
même temps). ResNet-50 à poids aléatoires (le temps de calcul ne dépend pas des poids ; rien n'est téléchargé), images
224 x 224, lot de 1, float16 sur le GPU (MPS). Deux modes :
- IPS = 0 : aussi vite que possible (test de saturation) ;
- IPS > 0 : cadence fixe (par exemple 30 images par seconde, une caméra) ; une image commence à chaque top d'horloge,
  ou dès la fin de la précédente si elle est en retard (sans rattrapage) ; elle a manqué son échéance si elle finit
  après le top suivant.
Fichier neuf à chaque lancement. Première ligne {"pret": t} une fois le modèle chargé et une image calculée ; puis toutes
les 5 s une ligne : début, fin, images, latences médiane et 99e centile de la ligne (ms, pour suivre), échéances
manquées, cadence visée, et pour chaque image sa fin (horloge time.time, comme la planification), sa latence (ms) et
les indices des images en retard ; l'analyse regroupe les images des fenêtres de planification.
S'arrête sur SIGTERM, après DUREE_MAX secondes, ou si son parent disparaît (jamais de charge orpheline).

Usage :
    python charge_vision.py SORTIE.jsonl [DUREE_MAX] [IPS]
"""
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision

sortie = Path(sys.argv[1])
duree_max = float(sys.argv[2]) if len(sys.argv) > 2 else 4 * 3600
ips = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
parent = os.getppid()
arret = False


def stop(*a):
    global arret
    arret = True


signal.signal(signal.SIGTERM, stop)
torch.manual_seed(0)
m = torchvision.models.resnet50(weights=None).eval().to("mps", torch.float16)
x = torch.randn(1, 3, 224, 224, device="mps", dtype=torch.float16)
with torch.no_grad():
    y = m(x)
    torch.mps.synchronize()
    assert torch.isfinite(y).all(), "sortie de la charge de vision non finie"
periode = 1 / ips if ips > 0 else 0.0
with torch.no_grad(), sortie.open("w") as f:
    f.write(json.dumps({"pret": time.time(), "ips_cible": ips}) + "\n")
    f.flush()
    debut = time.time()
    t0, lat, fins, retard = time.time(), [], [], []
    prochain = time.perf_counter()
    while not arret and time.time() - debut < duree_max:
        if periode:
            attente = prochain - time.perf_counter()
            if attente > 0:
                time.sleep(attente)
        a = time.perf_counter()
        m(x)
        torch.mps.synchronize()
        b = time.perf_counter()
        lat.append(b - a)
        fins.append(time.time())
        if periode:
            if b > prochain + periode:
                retard.append(len(lat) - 1)
                prochain = b                           # en retard : la suivante part tout de suite, sans rattrapage
            else:
                prochain += periode
        if time.time() - t0 >= 5:
            t1 = time.time()
            l = np.array(lat) * 1000
            f.write(json.dumps({"t0": t0, "t1": t1, "images": len(lat), "latence_p50_ms": float(np.median(l)),
                                "latence_p99_ms": float(np.percentile(l, 99)), "echeances_manquees": len(retard),
                                "ips_cible": ips, "fins": [round(v, 4) for v in fins],
                                "lat_ms": [round(float(v), 3) for v in l], "retard": retard}) + "\n")
            f.flush()
            t0, lat, fins, retard = t1, [], [], []
            if os.getppid() != parent:                 # parent disparu (arrêt brutal de vitesse.py) : on s'arrête
                break
