"""GPTQ (Frantar et al., arXiv:2210.17323), variante « poids seuls » pour EXP-009.

Même grille que le RTN par canal de quant.py : symétrique, une échelle par ligne s = max|W_ligne| / (2^(b-1) - 1),
calculée sur les poids d'origine, valeurs entières dans [-qmax, qmax]. La mémoire est donc identique à RTN ; seul
change le choix des arrondis : colonne par colonne, l'erreur d'un poids est compensée sur les colonnes suivantes
à l'aide de l'inverse de H = X^T X (X : entrées de la couche vues pendant la calibration). Calcul en float64 sur
CPU, déterministe. Amortissement 1 % de la moyenne de la diagonale, blocs de 128 colonnes, ordre naturel des
colonnes, comme l'implémentation de référence sans « act-order ».
"""
import torch

AMORTISSEMENT, BLOC = 0.01, 128


def echelle(w, bits):
    qmax = 2 ** (bits - 1) - 1
    return w.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / qmax, qmax


@torch.no_grad()
def rtn(w, bits):
    s, qmax = echelle(w, bits)
    return (w / s).round().clamp(-qmax, qmax) * s


@torch.no_grad()
def gptq_couche(w, h, bits, amortissement=AMORTISSEMENT, bloc=BLOC):
    """w : (sorties, entrées) ; h : (entrées, entrées), moyenne de x x^T. Renvoie les poids quantifiés (float64)."""
    W = w.detach().to(torch.float64).clone()
    H = h.detach().to(torch.float64).clone()
    s, qmax = echelle(W, bits)                       # échelles de RTN, sur les poids d'origine
    mort = torch.diag(H) == 0                        # entrée jamais active : poids sans effet, mis à 0
    H[mort, mort] = 1
    W[:, mort] = 0
    H += amortissement * torch.mean(torch.diag(H)) * torch.eye(H.shape[0], dtype=H.dtype)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    Q = torch.zeros_like(W)
    n = W.shape[1]
    for i1 in range(0, n, bloc):
        i2 = min(i1 + bloc, n)
        W1, Hinv1 = W[:, i1:i2].clone(), Hinv[i1:i2, i1:i2]
        Q1, Err1 = torch.zeros_like(W1), torch.zeros_like(W1)
        for i in range(i2 - i1):
            col = W1[:, i]
            q = (col.unsqueeze(1) / s).round().clamp(-qmax, qmax).mul(s).squeeze(1)
            Q1[:, i] = q
            err = (col - q) / Hinv1[i, i]
            W1[:, i:] -= err.unsqueeze(1) @ Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err
        Q[:, i1:i2] = Q1
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    return Q


def perte(w, q, h):
    """Erreur de sortie moyenne de la couche : trace((W - Q) H (W - Q)^T), H moyenne de x x^T."""
    d = (w.to(torch.float64) - q.to(torch.float64))
    return float(torch.einsum("oi,ij,oj->", d, h.to(torch.float64), d))
