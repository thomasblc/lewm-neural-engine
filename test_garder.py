"""Tests d'EXP-019 (pilote) : poids gardés hors de la grille ternaire (garder.py, gptq.py, quant.py).

Usage :
    python test_garder.py            # matrices construites (secondes)
    python test_garder.py --vrai     # en plus, le vrai checkpoint PushT et la calibration d'EXP-009 (CPU, quelques minutes)
"""
import os
import sys

import torch

import garder
import gptq


def donnees(seed, n_out=96, n_in=256, mort=(3, 77)):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n_in, n_in, generator=g, dtype=torch.float64) * torch.linspace(0.05, 2, n_in, dtype=torch.float64)
    x = torch.randn(20000, n_in, generator=g, dtype=torch.float64) @ A
    x[:, list(mort)] = 0                                       # entrées jamais actives
    w = torch.randn(n_out, n_in, generator=g, dtype=torch.float64)
    w[5, 10] = 9.0                                              # un poids aberrant
    return w, x.T @ x / len(x)


def test_masque():
    w, h = donnees(0)
    s, qmax = gptq.echelle(w, 2)
    for crit in garder.CRITERES:
        for frac in (0.0, 0.005, 0.01, 0.02):
            m = garder.masque(w, h, s * 0.5, qmax, frac, crit)
            assert int(m.sum()) == round(frac * w.numel()), (crit, frac)
            assert not m[:, [3, 77]].any(), "poids d'une entrée morte gardé"
        m1 = garder.masque(w, h, s * 0.5, qmax, 0.01, crit)
        assert torch.equal(m1, garder.masque(w, h, s * 0.5, qmax, 0.01, crit)), "non déterministe"
    assert garder.masque(w, h, s * 0.5, qmax, 0.01, "mag")[5, 10], "le poids aberrant n'est pas gardé par la taille"
    print("ok masque : nombre exact, entrées mortes exclues, déterministe, aberrant gardé par la taille")


def test_obs_calcul_direct():
    """Le critère obs classe selon (w - q)^2 / [H^-1]_jj, calculé ici à la main."""
    w, h = donnees(1, n_out=8, n_in=32, mort=())
    s, qmax = gptq.echelle(w, 2)
    s = s * 0.6
    H = h.clone() + gptq.AMORTISSEMENT * torch.mean(torch.diag(h)) * torch.eye(32, dtype=torch.float64)
    d = torch.diag(torch.linalg.inv(H))
    q = (w / s).round().clamp(-qmax, qmax) * s
    score = (w - q) ** 2 / d
    attendu = torch.zeros(w.numel(), dtype=torch.bool)
    attendu[torch.argsort(score.flatten(), descending=True, stable=True)[:13]] = True
    m = garder.masque(w, h, s, qmax, 13 / w.numel(), "obs")
    assert torch.equal(m, attendu.reshape(w.shape))
    print("ok obs : même classement que le calcul direct de (w - q)^2 / [H^-1]_jj")


def test_gptq_masque():
    w, h = donnees(2)
    s, qmax = gptq.echelle(w, 2)
    s = s * 0.5
    sans = gptq.gptq_couche(w, h, 2, s=s)
    vide = gptq.gptq_couche(w, h, 2, s=s, garder=torch.zeros_like(w, dtype=torch.bool))
    assert torch.equal(sans, vide), "un masque vide change GPTQ"
    m = garder.masque(w, h, s, qmax, 0.02, "obs")
    q = gptq.gptq_couche(w, h, 2, s=s, garder=m)
    k = q / s
    assert torch.allclose(k[~m], k[~m].round(), atol=1e-9) and k[~m].abs().max() <= qmax + 1e-9, "poids non gardé hors grille"
    # H identité : aucune compensation, les poids gardés valent exactement leur valeur d'origine, les autres RTN
    qi = gptq.gptq_couche(w, torch.eye(w.shape[1], dtype=torch.float64), 2, s=s, garder=m)
    rtn = (w / s).round().clamp(-qmax, qmax) * s
    assert torch.equal(qi[m], w[m]) and torch.equal(qi[~m], rtn[~m])
    # garder des poids fait baisser l'erreur de sortie, et obs plus que mag (critère fait pour cette erreur)
    e0 = gptq.perte(w, sans, h)
    e_obs = gptq.perte(w, q, h)
    e_mag = gptq.perte(w, gptq.gptq_couche(w, h, 2, s=s, garder=garder.masque(w, h, s, qmax, 0.02, "mag")), h)
    assert e_obs < e0 and e_obs < e_mag, (e0, e_obs, e_mag)
    print(f"ok GPTQ masqué : masque vide = EXP-011 ; grille respectée hors masque ; H = I redonne w et RTN ; erreur "
          f"relative à 2 % gardés : obs {e_obs / e0:.2f}, mag {e_mag / e0:.2f}")


def test_vrai_modele():
    """Vrai checkpoint PushT : « k0 » redonne exactement la version d'EXP-011 ; « k1 » garde 1 % de chaque couche, sur la
    grille ailleurs, et ses poids gardés tiennent en float16."""
    os.environ["LEWM_GPTQ_CALIB"] = os.path.abspath("../resultats/exp009-calib-pusht/calib.pt")
    import run_grid  # noqa: F401  (variables d'environnement de stable-worldmodel)
    import stable_worldmodel as swm
    import quant
    cfg = run_grid.charger_config(50, "cpu", "pusht")
    charger = lambda: swm.wm.utils.load_pretrained(cfg.policy).eval()
    ref, _ = quant.quantifier(charger(), "predictor:2:clip:gptq")
    k0, _ = quant.quantifier(charger(), "predictor:2:clip:gptq:k0:obs")
    for (n1, p1), (n2, p2) in zip(ref.named_parameters(), k0.named_parameters()):
        assert n1 == n2 and torch.equal(p1, p2), n1
    k1, touches = quant.quantifier(charger(), "predictor:2:clip:gptq:k1:obs")
    mods = dict(k1.named_modules())
    total, gardes = 0, 0
    for nom in touches:
        w_new = mods[nom].weight.reshape(mods[nom].weight.shape[0], -1).double()
        w_ref = dict(ref.named_modules())[nom].weight.reshape(w_new.shape[0], -1).double()
        diff = (w_new != w_ref)
        total += w_new.numel()
        gardes += int(diff.sum())
        assert torch.equal(w_new[diff].float().half().float().double(), w_new[diff].float().double())
    frac = gardes / total
    assert 0.002 < frac <= 0.0101, frac        # au plus 1 % ; un poids gardé peut retomber sur sa valeur de grille
    print(f"ok vrai modèle : k0 identique à EXP-011 ({len(touches)} couches) ; k1 change {100 * frac:.2f} % des poids, en float16")


if __name__ == "__main__":
    test_masque()
    test_obs_calcul_direct()
    test_gptq_masque()
    if "--vrai" in sys.argv:
        test_vrai_modele()
