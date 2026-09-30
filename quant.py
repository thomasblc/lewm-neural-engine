"""Sonde de faisabilité (pilote, pas un résultat d'article) : quantification simulée des poids de LeWM.

Arrondi au plus proche (RTN), symétrique, une échelle par canal de sortie. Les poids restent en float
mais ne prennent que 2^b - 1 valeurs par canal. Réglage par variable d'environnement :
LEWM_QUANT="all:4", "encoder:4", "predictor:4", ou vide (pleine précision).

EXP-004 ajoute deux groupes (« autres » : encodeur d'action et projecteurs ; « encpred » : encodeur et
prédicteur) et la quantification par groupes de poids : "all:4:g64" donne une échelle à chaque groupe
de 64 poids consécutifs d'une ligne. Une couche dont l'entrée n'est pas divisible par la taille du groupe
reste quantifiée par canal (2 couches de LeWM : la projection des images en carrés, 588 entrées, et une
couche de l'encodeur d'action, 10 entrées). Les versions sans ":gN" passent par le code d'origine.

EXP-009 ajoute GPTQ : "all:4:gptq" arrondit les mêmes couches, sur la même grille et avec les mêmes échelles que
"all:4", mais choisit les arrondis avec gptq.py, à partir des statistiques de calibration désignées par la variable
LEWM_GPTQ_CALIB (fichier écrit par run_grid.py --calibrer-gptq). Les autres versions ne changent pas.
"""
import os
import torch

GROUPES = {
    "all": ("encoder", "predictor", "action_encoder", "projector", "pred_proj"),
    "encoder": ("encoder",),
    "predictor": ("predictor",),
    "autres": ("action_encoder", "projector", "pred_proj"),
    "encpred": ("encoder", "predictor"),
}


@torch.no_grad()
def rtn_par_canal(w, bits):
    qmax = 2 ** (bits - 1) - 1
    flat = w.reshape(w.shape[0], -1)
    scale = flat.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / qmax
    q = (flat / scale).round().clamp(-qmax, qmax) * scale
    return q.reshape(w.shape)


@torch.no_grad()
def rtn_par_groupe(w, bits, taille):
    """Une échelle par groupe de `taille` poids consécutifs d'une ligne. Renvoie (poids, par_groupe)."""
    flat = w.reshape(w.shape[0], -1)
    if flat.shape[1] % taille:
        return rtn_par_canal(w, bits), False
    qmax = 2 ** (bits - 1) - 1
    g = flat.reshape(flat.shape[0], -1, taille)
    scale = g.abs().amax(dim=2, keepdim=True).clamp_min(1e-12) / qmax
    q = (g / scale).round().clamp(-qmax, qmax) * scale
    return q.reshape(w.shape), True


def lire_spec(spec):
    """"all:4" -> ("all", 4, None) ; "all:4:g64" -> ("all", 4, 64)."""
    morceaux = spec.split(":")
    if len(morceaux) not in (2, 3) or morceaux[0] not in GROUPES:
        raise ValueError(f"spécification inconnue : {spec}")
    taille = None
    if len(morceaux) == 3:
        if not morceaux[2].startswith("g"):
            raise ValueError(f"spécification inconnue : {spec}")
        taille = int(morceaux[2][1:])
    return morceaux[0], int(morceaux[1]), taille


@torch.no_grad()
def quantifier(model, spec=None):
    spec = spec if spec is not None else os.environ.get("LEWM_QUANT", "")
    if not spec:
        return model, []
    avec_gptq = spec.endswith(":gptq")
    groupe, bits, taille = lire_spec(spec[:-len(":gptq")] if avec_gptq else spec)
    if avec_gptq:
        import gptq
        if taille is not None or not os.environ.get("LEWM_GPTQ_CALIB"):
            raise ValueError(f"{spec} : GPTQ demande une version par canal et LEWM_GPTQ_CALIB")
        calib = torch.load(os.environ["LEWM_GPTQ_CALIB"], map_location="cpu")
    touches, par_canal = [], []
    for nom, mod in model.named_modules():
        if nom.split(".")[0] in GROUPES[groupe] and isinstance(mod, (torch.nn.Linear, torch.nn.Conv2d)):
            if avec_gptq:
                h = calib["H"][nom] / calib["n"][nom]
                w32 = mod.weight.reshape(mod.weight.shape[0], -1)
                q = gptq.gptq_couche(w32, h, bits)
                # retour exact sur la grille float32 de RTN : mêmes échelles, entiers de GPTQ
                s32 = w32.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / (2 ** (bits - 1) - 1)
                k = (q / s32.double()).round().clamp(-(2 ** (bits - 1) - 1), 2 ** (bits - 1) - 1).to(w32.dtype)
                mod.weight.copy_((k * s32).reshape(mod.weight.shape))
            elif taille is None:
                mod.weight.copy_(rtn_par_canal(mod.weight, bits))
            else:
                q, ok = rtn_par_groupe(mod.weight, bits, taille)
                mod.weight.copy_(q)
                if not ok:
                    par_canal.append(nom)
            touches.append(nom)
    print(f"[quant] {spec} : {len(touches)} couches quantifiées"
          + (f", dont {len(par_canal)} par canal faute de divisibilité : {par_canal}" if par_canal else ""))
    return model, touches
