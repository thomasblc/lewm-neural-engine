"""Tests d'affinage.py et de quant.DETAILS (EXP-019, pilote), sur le vrai checkpoint de LeWM, CPU.

Usage :
    python test_affinage.py
"""
import os

import torch

import affinage as AF

CALIB = "/Users/thomasblanc/Documents/PRO/these-ia-thomas/papier-corl-2026/resultats/exp009-calib-pusht/calib.pt"
SPEC = "predictor:2:clip:gptq:k1:obs"


def main():
    prof, eleve, details = AF.charger_modeles(SPEC, CALIB, "cpu")
    n_g = sum(int(d["G"].sum()) for d in details.values())
    n_p = sum(d["k"].numel() for d in details.values())
    print(f"ok chargement : {len(details)} couches du prédicteur, {n_g} poids gardés sur {n_p} ({100 * n_g / n_p:.2f} %)")
    for nom, d in details.items():
        W = eleve.get_submodule(nom).weight.detach().reshape(d["k"].shape)
        hors = ~d["G"]
        assert set(d["k"][hors].unique().tolist()) <= {-1.0, 0.0, 1.0}, nom
        assert torch.equal(W[hors], (d["k"] * d["s"])[hors]), nom
    print("ok DETAILS : entiers ternaires, poids hors masque = k x s exactement")

    libres = AF.depart(eleve, details, "cpu")                       # vérifie aussi poids(m=1, K) == W
    etat = AF.exporter(eleve, details, libres)
    AF.verifier_format(etat, details)
    with torch.no_grad():
        for nom, d in details.items():
            libres[nom]["m"].mul_(1.03)
    AF.verifier_format(AF.exporter(eleve, details, libres), details)
    print("ok export : format accepté au départ et avec des échelles changées")

    nom = next(iter(details))
    d = details[nom]
    for cas in ("hors_grille", "echelle_double", "zero"):
        mauvais = {k: v.clone() for k, v in etat.items()}
        W = mauvais[f"{nom}.weight"].reshape(d["k"].shape)
        if cas == "hors_grille":
            i = (~d["G"] & (d["k"] != 0)).nonzero()[0]
            W[i[0], i[1]] *= 1.5
        elif cas == "echelle_double":
            i = (~d["G"] & (d["k"] != 0)).nonzero()[0]
            W[i[0], i[1]] = W[i[0], i[1]] + 2 ** -10 * W[i[0], i[1]].sign() * W[i[0], i[1]].abs()
        else:
            i = (~d["G"] & (d["k"] == 0)).nonzero()[0]
            W[i[0], i[1]] = 0.01
        try:
            AF.verifier_format(mauvais, details)
            raise AssertionError(f"cas {cas} accepté")
        except ValueError:
            pass
    print("ok vérification : poids hors grille, seconde échelle sur une ligne, entier nul devenu non nul refusés")

    D = prof.predictor.pos_embedding.shape[-1]
    g = torch.Generator().manual_seed(0)
    X = {T: torch.randn(64, T, D, generator=g).half() for T in (1, 3)}
    C = {T: torch.randn(64, T, D, generator=g).half() for T in (1, 3)}
    l0, p0 = AF.affiner(prof, eleve, details, X, C, 0, 8, 1e-3, 0, "cpu")
    assert all(torch.equal(a, b) for a, b in zip(AF.exporter(eleve, details, l0).values(), etat.values()))
    l1, p1 = AF.affiner(prof, eleve, details, X, C, 30, 16, 1e-3, 0, "cpu")
    e1 = AF.exporter(eleve, details, l1)
    AF.verifier_format(e1, details)
    x, c = X[3][:16].float(), C[3][:16].float()
    with torch.no_grad():
        cible = prof.predict(x, c)[:, -1]
        avant = ((AF.sortie_eleve(eleve, details, l0, x, c) - cible) ** 2).mean().item()
        apres = ((AF.sortie_eleve(eleve, details, l1, x, c) - cible) ** 2).mean().item()
        eleve2 = eleve
        for k, v in e1.items():
            eleve2.get_parameter(k).copy_(v)
        charge = ((eleve2.predict(x, c)[:, -1] - AF.sortie_eleve(eleve, details, l1, x, c)) ** 2).mean().item()
    assert apres < avant, (avant, apres)
    assert charge < 1e-4 * max(1.0, cible.pow(2).mean().item()), charge
    print(f"ok réglage : 0 pas = départ ; 30 pas font baisser la perte ({avant:.3e} -> {apres:.3e}) ; "
          f"les poids exportés redonnent la même sortie (écart {charge:.1e}, arrondis float16)")
    print("tous les tests passent")


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    main()
