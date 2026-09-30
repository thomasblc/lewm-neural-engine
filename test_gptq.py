"""Tests unitaires de gptq.py, sur des matrices construites (CPU, quelques secondes).

Usage :
    python test_gptq.py
"""
import torch

import gptq
import quant


def test_identite_egale_rtn():
    """Avec H proportionnelle à l'identité, GPTQ ne propage aucune erreur : il doit redonner RTN exactement."""
    g = torch.Generator().manual_seed(0)
    for bits in (3, 4, 8):
        w = torch.randn(64, 300, generator=g, dtype=torch.float64)
        q = gptq.gptq_couche(w, 3.7 * torch.eye(300, dtype=torch.float64), bits)
        assert torch.equal(q, gptq.rtn(w, bits)), bits
    # et gptq.rtn est bien le RTN de quant.py (même grille, même échelle)
    w32 = torch.randn(64, 300, generator=g)
    assert torch.equal(gptq.rtn(w32, 4), quant.rtn_par_canal(w32, 4))
    print("ok H = identité : GPTQ redonne RTN exactement (3, 4, 8 bits) ; même RTN que quant.py")


def test_grille():
    """Les poids de GPTQ restent sur la grille de RTN : q / s entiers dans [-qmax, qmax], échelles de RTN."""
    g = torch.Generator().manual_seed(1)
    w = torch.randn(48, 200, generator=g, dtype=torch.float64)
    x = torch.randn(5000, 200, generator=g, dtype=torch.float64) @ torch.randn(200, 200, generator=g, dtype=torch.float64)
    q = gptq.gptq_couche(w, x.T @ x / len(x), 4)
    s, qmax = gptq.echelle(w, 4)
    k = q / s
    assert torch.allclose(k, k.round(), atol=1e-9) and k.abs().max() <= qmax + 1e-9
    assert len(torch.unique(k.round())) <= 2 * qmax + 1
    print("ok grille : valeurs entières dans [-7, 7] × échelle de RTN (mémoire identique à RTN)")


def test_erreur_plus_faible():
    """Sur des entrées corrélées, GPTQ fait nettement moins d'erreur de sortie que RTN, sur la calibration et sur
    des entrées tenues à part tirées de la même loi."""
    g = torch.Generator().manual_seed(2)
    A = torch.randn(256, 256, generator=g, dtype=torch.float64) * torch.linspace(0.05, 2, 256, dtype=torch.float64)
    xc = torch.randn(20000, 256, generator=g, dtype=torch.float64) @ A
    xt = torch.randn(5000, 256, generator=g, dtype=torch.float64) @ A
    w = torch.randn(128, 256, generator=g, dtype=torch.float64)
    for bits in (3, 4):
        hc, ht = xc.T @ xc / len(xc), xt.T @ xt / len(xt)
        qg, qr = gptq.gptq_couche(w, hc, bits), gptq.rtn(w, bits)
        rc, rt = gptq.perte(w, qg, hc) / gptq.perte(w, qr, hc), gptq.perte(w, qg, ht) / gptq.perte(w, qr, ht)
        assert rc < 0.9 and rt < 0.9, (bits, rc, rt)
    print(f"ok erreur : GPTQ sous RTN sur la calibration et sur des entrées tenues à part (dernier rapport {rt:.2f})")


def test_perte_egale_calcul_direct():
    g = torch.Generator().manual_seed(3)
    x = torch.randn(1000, 40, generator=g, dtype=torch.float64)
    w, q = torch.randn(10, 40, generator=g, dtype=torch.float64), torch.randn(10, 40, generator=g, dtype=torch.float64)
    direct = float(((x @ (w - q).T) ** 2).sum() / len(x))
    assert abs(gptq.perte(w, q, x.T @ x / len(x)) - direct) < 1e-9 * direct
    print("ok perte : trace((W-Q) H (W-Q)^T) égale à l'erreur de sortie moyenne calculée directement")


def test_entree_morte():
    """Une entrée jamais active (colonne de H nulle) : GPTQ ne plante pas et met ce poids à 0."""
    g = torch.Generator().manual_seed(4)
    x = torch.randn(3000, 30, generator=g, dtype=torch.float64)
    x[:, 7] = 0
    q = gptq.gptq_couche(torch.randn(8, 30, generator=g, dtype=torch.float64), x.T @ x / len(x), 4)
    assert torch.isfinite(q).all() and (q[:, 7] == 0).all()
    print("ok entrée morte : pas d'erreur, poids correspondant à 0")


def gptq_naif(w, h, bits):
    """Référence indépendante : quantification colonne par colonne, avec l'inverse de H recalculé à chaque étape
    sur les colonnes restantes (équation de GPTQ, sans Cholesky ni blocs)."""
    W = w.clone()
    H = h.clone() + gptq.AMORTISSEMENT * torch.mean(torch.diag(h)) * torch.eye(h.shape[0], dtype=h.dtype)
    s, qmax = gptq.echelle(w, bits)
    Q = torch.zeros_like(W)
    for i in range(W.shape[1]):
        Hinv = torch.linalg.inv(H[i:, i:])                      # inverse sur les colonnes pas encore quantifiées
        q = (W[:, i:i + 1] / s).round().clamp(-qmax, qmax).mul(s).squeeze(1)
        Q[:, i] = q
        err = (W[:, i] - q) / Hinv[0, 0]
        W[:, i + 1:] -= err.unsqueeze(1) @ Hinv[0, 1:].unsqueeze(0)
    return Q


def test_reference_naive():
    """La version rapide (Cholesky, blocs de 128 ou de 16) donne les mêmes poids que la référence naïve."""
    g = torch.Generator().manual_seed(5)
    x = torch.randn(4000, 300, generator=g, dtype=torch.float64) @ torch.randn(300, 300, generator=g, dtype=torch.float64)
    w, h = torch.randn(32, 300, generator=g, dtype=torch.float64), x.T @ x / len(x)
    ref = gptq_naif(w, h, 4)
    for bloc in (128, 16, 300):
        q = gptq.gptq_couche(w, h, 4, bloc=bloc)
        assert torch.allclose(q, ref, atol=1e-9, rtol=0), (bloc, float((q - ref).abs().max()))
    print("ok référence : version rapide identique à une implémentation naïve indépendante (blocs 128, 16, 300)")


if __name__ == "__main__":
    test_reference_naive()
    test_identite_egale_rtn()
    test_grille()
    test_erreur_plus_faible()
    test_perte_egale_calcul_direct()
    test_entree_morte()
    print("tous les tests passent")
