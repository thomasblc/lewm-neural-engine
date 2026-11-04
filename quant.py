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

EXP-011 ajoute l'échelle réduite : "predictor:2:clip" est un RTN par canal dont l'échelle de chaque ligne vaut
f x max|W_ligne| / qmax, avec un facteur f par couche choisi sur la calibration (gptq.facteur_reduit) ;
"predictor:2:clip:gptq" garde exactement ces échelles et choisit les arrondis avec GPTQ. Mémoire identique.

EXP-019 (pilote) ajoute les poids gardés : "predictor:2:clip:gptq:k1:obs" garde 1 % des poids de chaque couche hors de la
grille, en 16 bits, choisis par le critère « obs » ou « mag » de garder.py ; avec GPTQ, leur valeur est celle que GPTQ
leur donne à leur colonne (compensations comprises), arrondie en float16 ; sans GPTQ (":clip:k1:obs"), le poids
d'origine arrondi en float16. Mêmes échelles qu'EXP-011. Sans suffixe ":k", rien ne change.

EXP-019 (pilote, réglage fin) : chaque appel range dans DETAILS, pour chaque couche à échelle réduite, ses échelles
(s), ses entiers (k) et son masque de poids gardés (G, ou None) ; affinage.py s'en sert. Les poids ne changent pas.
"""
import os
import re
import torch

GROUPES = {
    "all": ("encoder", "predictor", "action_encoder", "projector", "pred_proj"),
    "encoder": ("encoder",),
    "predictor": ("predictor",),
    "autres": ("action_encoder", "projector", "pred_proj"),
    "encpred": ("encoder", "predictor"),
}
DETAILS = {}


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
    DETAILS.clear()
    if not spec:
        return model, []
    garde = re.search(r":k([0-9.]+):([a-z]+)$", spec)
    sans_garde = spec[:garde.start()] if garde else spec
    avec_gptq = sans_garde.endswith(":gptq")
    base = sans_garde[:-len(":gptq")] if avec_gptq else sans_garde
    avec_clip = base.endswith(":clip")
    groupe, bits, taille = lire_spec(base[:-len(":clip")] if avec_clip else base)
    if garde and not avec_clip:
        raise ValueError(f"{spec} : les poids gardés (EXP-019) demandent l'échelle réduite (:clip)")
    if garde:
        import garder
        fraction, critere = float(garde.group(1)) / 100, garde.group(2)
    if avec_gptq or avec_clip:
        import gptq
        if taille is not None or not os.environ.get("LEWM_GPTQ_CALIB"):
            raise ValueError(f"{spec} : GPTQ et l'échelle réduite demandent une version par canal et LEWM_GPTQ_CALIB")
        calib = torch.load(os.environ["LEWM_GPTQ_CALIB"], map_location="cpu")
    qmax = 2 ** (bits - 1) - 1
    touches, par_canal, facteurs, gardes, n_poids = [], [], [], 0, 0
    for nom, mod in model.named_modules():
        if nom.split(".")[0] in GROUPES[groupe] and isinstance(mod, (torch.nn.Linear, torch.nn.Conv2d)):
            if avec_clip:
                h = calib["H"][nom] / calib["n"][nom]
                w32 = mod.weight.reshape(mod.weight.shape[0], -1)
                f = gptq.facteur_reduit(w32, h, bits)
                s32 = w32.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / qmax * f
                G = garder.masque(w32, h, s32.double(), qmax, fraction, critere) if garde else None
                if avec_gptq:
                    q = gptq.gptq_couche(w32, h, bits, s=s32.double(), garder=G)
                    k = (q / s32.double()).round().clamp(-qmax, qmax).to(w32.dtype)
                    haute = q
                else:
                    k = (w32 / s32).round().clamp(-qmax, qmax)
                    haute = w32
                val = k * s32
                if G is not None:                             # EXP-019 : poids gardés, en float16
                    val = torch.where(G, haute.to(torch.float16).to(w32.dtype), val)
                    gardes += int(G.sum())
                n_poids += w32.numel()
                mod.weight.copy_(val.reshape(mod.weight.shape))
                facteurs.append(f)
                DETAILS[nom] = {"s": s32.detach().clone(), "k": k.detach().clone(), "G": None if G is None else G.clone()}
            elif avec_gptq:
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
          + (f", dont {len(par_canal)} par canal faute de divisibilité : {par_canal}" if par_canal else "")
          + (f", facteurs d'échelle de {min(facteurs):.2f} à {max(facteurs):.2f}" if facteurs else "")
          + (f", {gardes} poids gardés en 16 bits sur {n_poids} ({100 * gardes / n_poids:.3f} %)" if garde else ""))
    return model, touches
