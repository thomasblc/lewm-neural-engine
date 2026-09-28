"""Sonde de faisabilité (pilote, pas un résultat d'article) : quantification simulée des poids de LeWM.

Arrondi au plus proche (RTN), symétrique, une échelle par canal de sortie. Les poids restent en float
mais ne prennent que 2^b - 1 valeurs par canal. Réglage par variable d'environnement :
LEWM_QUANT="all:4", "encoder:4", "predictor:4", ou vide (pleine précision).
"""
import os
import torch

GROUPES = {
    "all": ("encoder", "predictor", "action_encoder", "projector", "pred_proj"),
    "encoder": ("encoder",),
    "predictor": ("predictor",),
}


@torch.no_grad()
def rtn_par_canal(w, bits):
    qmax = 2 ** (bits - 1) - 1
    flat = w.reshape(w.shape[0], -1)
    scale = flat.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / qmax
    q = (flat / scale).round().clamp(-qmax, qmax) * scale
    return q.reshape(w.shape)


@torch.no_grad()
def quantifier(model, spec=None):
    spec = spec if spec is not None else os.environ.get("LEWM_QUANT", "")
    if not spec:
        return model, []
    groupe, bits = spec.split(":")
    bits = int(bits)
    touches = []
    for nom, mod in model.named_modules():
        if nom.split(".")[0] in GROUPES[groupe] and isinstance(mod, (torch.nn.Linear, torch.nn.Conv2d)):
            mod.weight.copy_(rtn_par_canal(mod.weight, bits))
            touches.append(nom)
    print(f"[quant] {spec} : {len(touches)} couches quantifiées")
    return model, touches
