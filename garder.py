"""EXP-019 (pilote) : choisir les poids qu'on garde en haute précision quand le reste passe en ternaire.

Deux critères, une même fraction de poids par couche :
- « mag » : les plus grands |w| ;
- « obs » : la sensibilité de SpQR (arXiv:2306.03078) et d'OBS, (w - q(w))^2 / [H^-1]_jj, q l'arrondi sur la grille de
  la couche (échelles imposées s), H = moyenne de x x^T sur la calibration, amortie comme dans gptq.py.
Les colonnes jamais actives à la calibration (diag(H) = 0) ne sont jamais gardées : leurs poids n'ont aucun effet.
Classement stable : en cas d'égalité, l'ordre des indices décide (résultat déterministe).
"""
import torch

from gptq import AMORTISSEMENT

CRITERES = ("mag", "obs")


@torch.no_grad()
def masque(w, h, s, qmax, fraction, critere, amortissement=AMORTISSEMENT):
    """w : (sorties, entrées) ; h : (entrées, entrées) ; s : échelles (sorties, 1). Renvoie un masque booléen de w."""
    if critere not in CRITERES:
        raise ValueError(f"critère inconnu : {critere}")
    W = w.detach().to(torch.float64)
    actif = torch.diag(h.to(torch.float64)) > 0
    if critere == "mag":
        score = W.abs()
    else:
        H = h.detach().to(torch.float64).clone()
        mort = ~actif
        H[mort, mort] = 1
        H += amortissement * torch.mean(torch.diag(H)) * torch.eye(H.shape[0], dtype=H.dtype)
        hinv_diag = torch.diag(torch.cholesky_inverse(torch.linalg.cholesky(H)))
        S = s.detach().to(torch.float64).reshape(-1, 1)
        q = (W / S).round().clamp(-qmax, qmax) * S
        score = (W - q) ** 2 / hinv_diag.unsqueeze(0)
    score = score.clone()
    score[:, ~actif] = -float("inf")
    n = int(round(fraction * W.numel()))
    n = min(n, int(actif.sum()) * W.shape[0])
    m = torch.zeros(W.numel(), dtype=torch.bool)
    if n > 0:
        ordre = torch.argsort(score.flatten(), descending=True, stable=True)
        m[ordre[:n]] = True
    return m.reshape(W.shape)
