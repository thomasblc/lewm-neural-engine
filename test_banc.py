"""Tests du banc d'essai d'EXP-020 (CPU, une minute environ).

Usage :
    python test_banc.py
"""
import copy

import torch

import banc as B
import gptq


def test_monde():
    g = torch.Generator().manual_seed(5)
    s = B.etats_aleatoires(50_000, g)
    a = 2 * torch.rand(50_000, 2, generator=g) - 1
    s2 = B.pas(s, a)
    assert ((s2[:, :2] >= B.RA - 1e-6) & (s2[:, :2] <= 1 - B.RA + 1e-6)).all()
    assert ((s2[:, 2:] >= B.RB - 1e-6) & (s2[:, 2:] <= 1 - B.RB + 1e-6)).all()
    loin = ((s[:, :2] - s[:, 2:]).norm(dim=1) > B.RA + B.RB + 2 * B.VMAX) \
        & (s[:, :2] > B.RA + B.VMAX).all(1) & (s[:, :2] < 1 - B.RA - B.VMAX).all(1)
    assert torch.allclose(s2[loin, :2], s[loin, :2] + B.VMAX * a[loin], atol=1e-6)    # déplacement libre exact
    assert torch.equal(s2[loin, 2:], s[loin, 2:])                                        # bloc immobile sans contact
    pousse = (s2[:, 2:] - s[:, 2:]).norm(dim=1) > 1e-9
    assert 0.03 < pousse.float().mean() < 0.3
    print("ok monde : murs, déplacement libre exact, bloc immobile sans contact, "
          f"{100 * pousse.float().mean():.1f} % de transitions avec poussée")


def petit_modele():
    torch.manual_seed(3)
    m = B.Modele().eval()
    return m


def test_compact():
    m = petit_modele()
    c0 = B.quantifier(m, 2, "gptq", n=4000)
    assert B.valider(c0, bits=2, budget=0) == 0
    for c in c0["couches"]:
        assert set(c["codes"].unique().tolist()) <= {-1, 0, 1}
    # sans poids gardés, la couche 0 est exactement le GPTQ de gptq.py avec les mêmes échelles
    s, a, _ = B.transitions(4000, B.GRAINE_CALIB_Q)
    x = B.Modele.entrees(s, a).double()
    h = x.T @ x / x.shape[0]
    w = m.couches[0].weight.detach().double()
    ech = c0["couches"][0]["echelle"].double()[:, None]
    attendu = gptq.gptq_couche(w, h, 2, s=ech)
    obtenu = B.depuis_compact(c0).couches[0].weight.detach().double()
    assert torch.allclose(obtenu, attendu, atol=1e-6)
    ck = B.quantifier(m, 2, "gptq", B.FRACTION, "obs", n=4000)
    n = B.valider(ck, bits=2, budget=B.budget_gardes())
    assert abs(n - B.budget_gardes()) <= len(ck["couches"])                # 1 % par couche, arrondis compris
    for c, lin in zip(ck["couches"], B.depuis_compact(ck).couches):
        w = lin.weight.detach().flatten()
        assert torch.equal(w[c["garde_idx"]], c["garde_val"].float())
        reste = torch.ones_like(w, dtype=torch.bool)
        reste[c["garde_idx"]] = False
        e = c["echelle"].float()[:, None].expand(lin.weight.shape).flatten()
        assert torch.allclose(w[reste].abs(), (w[reste].abs() > 0) * e[reste])         # grille ternaire ailleurs
    print(f"ok format compact : grille, couche 0 identique à gptq.py, {n} poids gardés (budget {B.budget_gardes()})")


def test_valider():
    ck = B.quantifier(petit_modele(), 2, "gptq", B.FRACTION, "obs", n=4000)
    mauvais = []
    c = copy.deepcopy(ck)
    c["couches"][1]["codes"][0, 0] = 2
    mauvais.append(c)
    c = copy.deepcopy(ck)
    c["couches"][1]["garde_idx"] = torch.cat([c["couches"][1]["garde_idx"], c["couches"][1]["garde_idx"][:1]])
    c["couches"][1]["garde_val"] = torch.cat([c["couches"][1]["garde_val"], c["couches"][1]["garde_val"][:1]])
    mauvais.append(c)
    c = copy.deepcopy(ck)
    extra = torch.arange(0, c["couches"][2]["codes"].numel(), 7)       # dans les limites, budget dépassé
    c["couches"][2]["garde_idx"] = extra
    c["couches"][2]["garde_val"] = torch.zeros(extra.numel(), dtype=torch.float16)
    mauvais.append(c)
    c = copy.deepcopy(ck)
    c["couches"][0]["codes"] = c["couches"][0]["codes"].float()
    mauvais.append(c)
    for i, c in enumerate(mauvais):
        try:
            B.valider(c, bits=2, budget=B.budget_gardes())
            raise AssertionError(f"cas {i} accepté")
        except ValueError:
            pass
    print("ok validation : code hors grille, indice en double, budget dépassé, codes non entiers refusés")


def test_affiner():
    ck = B.quantifier(petit_modele(), 2, "gptq", B.FRACTION, "obs", n=4000)
    ca = B.affiner(ck, pas_=20, n=5000)
    B.valider(ca, bits=2, budget=B.budget_gardes())
    for c0, c1 in zip(ck["couches"], ca["couches"]):
        assert torch.equal(c0["codes"], c1["codes"]) and torch.equal(c0["garde_idx"], c1["garde_idx"])
    print("ok affinage : codes et positions gardées inchangés")


if __name__ == "__main__":
    test_monde()
    test_compact()
    test_valider()
    test_affiner()
    print("tous les tests passent")
