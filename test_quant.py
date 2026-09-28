"""Tests de quant.py (EXP-004), sur des poids aléatoires et sur le vrai checkpoint de LeWM (CPU).

Le test principal : pour toutes les versions d'EXP-003, le nouveau quant.py donne exactement les mêmes
poids que celui du commit c5703db, qui a produit la grille.

Usage :
    python test_quant.py
"""
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

import quant as Q

ICI = Path(__file__).resolve().parent


def test_groupes_aleatoires():
    torch.manual_seed(0)
    w = torch.randn(96, 192)
    for b in (8, 6, 5, 4, 3):
        canal = Q.rtn_par_canal(w, b)
        entier, ok = Q.rtn_par_groupe(w, b, 192)             # un seul groupe par ligne = par canal
        assert ok and torch.equal(entier, canal)
        g64, ok = Q.rtn_par_groupe(w, b, 64)
        assert ok
        assert (g64 - w).norm() <= (canal - w).norm()         # une échelle plus fine n'aggrave rien
        qmax = 2 ** (b - 1) - 1
        for ligne_q, ligne in zip(g64.reshape(96, 3, 64), w.reshape(96, 3, 64)):
            for gq, gw in zip(ligne_q, ligne):                # au plus 2^b - 1 valeurs par groupe
                assert len(torch.unique(gq)) <= 2 * qmax + 1
                assert gq.abs().max() <= gw.abs().max() * (1 + 1e-6)
    w10 = torch.randn(8, 10)                                  # entrée de 10 : repli par canal
    q, ok = Q.rtn_par_groupe(w10, 4, 64)
    assert not ok and torch.equal(q, Q.rtn_par_canal(w10, 4))
    conv = torch.randn(192, 3, 14, 14)                        # 588 entrées : repli par canal
    q, ok = Q.rtn_par_groupe(conv, 4, 64)
    assert not ok and torch.equal(q, Q.rtn_par_canal(conv, 4))
    print("ok groupes sur poids aléatoires (égalité avec par canal pour un groupe par ligne, erreur plus faible, repli)")


def test_lire_spec():
    assert Q.lire_spec("all:4") == ("all", 4, None)
    assert Q.lire_spec("autres:3") == ("autres", 3, None)
    assert Q.lire_spec("all:3:g64") == ("all", 3, 64)
    for faux in ("tout:4", "all", "all:4:64", "all:4:g64:x"):
        try:
            Q.lire_spec(faux)
            raise AssertionError(faux)
        except ValueError:
            pass
    print("ok lecture des spécifications")


def ancien_quant():
    """quant.py du commit c5703db, chargé comme module à part."""
    src = subprocess.run(["git", "-C", str(ICI), "show", "c5703db:quant.py"], capture_output=True, text=True,
                         check=True).stdout
    f = Path(tempfile.mkdtemp()) / "quant_c5703db.py"
    f.write_text(src)
    spec = importlib.util.spec_from_file_location("quant_c5703db", f)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_vrai_modele():
    import run_grid as RG
    import stable_worldmodel as swm
    cfg = RG.charger_config(50, "cpu")

    def charger():
        return swm.wm.utils.load_pretrained(cfg.policy).eval()

    ancien = ancien_quant()
    for spec in ("all:8", "all:6", "all:5", "all:4", "all:3", "encoder:4", "encoder:3", "predictor:4", "predictor:3"):
        a, ta = ancien.quantifier(charger(), spec)
        n, tn = Q.quantifier(charger(), spec)
        assert ta == tn, spec
        pa, pn = dict(a.named_parameters()), dict(n.named_parameters())
        assert all(torch.equal(pa[k], pn[k]) for k in pa), spec
    print("ok vrai modèle : les 9 versions d'EXP-003 donnent les mêmes poids qu'avec quant.py de c5703db")

    attendu = {"autres": 6, "encpred": 103, "all": 109}
    for groupe, n_couches in attendu.items():
        _, t = Q.quantifier(charger(), f"{groupe}:4")
        assert len(t) == n_couches and len(Q.quantifier(charger(), f"{groupe}:4:g64")[1]) == n_couches, groupe
    _, t = Q.quantifier(charger(), "autres:4")
    assert sorted({x.split(".")[0] for x in t}) == ["action_encoder", "pred_proj", "projector"]

    ref = charger()
    poids = {nom: m.weight.detach().clone() for nom, m in ref.named_modules()
             if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d))}
    print("erreur relative des poids de tout le modèle, par canal contre groupes de 64 :")
    for b in (4, 3):
        erreurs = []
        for spec in (f"all:{b}", f"all:{b}:g64"):
            m = Q.quantifier(charger(), spec)[0]
            num = sum(((mod.weight - poids[nom]) ** 2).sum().item() for nom, mod in m.named_modules() if nom in poids)
            den = sum((w ** 2).sum().item() for w in poids.values())
            erreurs.append((num / den) ** 0.5)
        assert erreurs[1] < erreurs[0]
        print(f"   {b} bits : {100 * erreurs[0]:.2f} % par canal, {100 * erreurs[1]:.2f} % par groupes de 64")
    print("ok vrai modèle : nombre de couches par groupe, groupes de 64 plus précis")


if __name__ == "__main__":
    test_groupes_aleatoires()
    test_lire_spec()
    test_vrai_modele()
    print("tous les tests passent")
