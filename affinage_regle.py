"""EXP-020, amendement 2 : affinage par gradient à codes figés, mieux réglé, contre la recette d'Opus.

Grille de 8 réglages (pas 6 000 ou 20 000, taux 3e-4 ou 1e-3, perte à un pas seule ou plus perte sur 8 pas en boucle
ouverte le long des trajectoires de `methode.donnees`), sur les 3 graines de données d'amendement 1. Choix sur un jeu
de sélection (300 épisodes, graine 2, plan 11), jamais sur le test.

Usage :
    python affinage_regle.py tests
    python affinage_regle.py lancer
"""
import argparse
import itertools
import json
import time

import torch

import banc as B
import references as R

GRAINES = (1111, 2222, 3333)
GRILLE = list(itertools.product((6000, 20000), (3e-4, 1e-3), (0, 8)))
SELECTION = (2, 300, 11)                 # graine des épisodes, nombre, graine de planification
N_IID, N_TRAJ, PAS_TRAJ = 200_000, 1000, 40   # disposition de methode.donnees


def indices_traj(t, i):
    return N_IID + t * N_TRAJ + i


def verifier_disposition(s, s2):
    """Les trajectoires de methode.donnees sont rangées pas par pas : l'état t+1 de la trajectoire i suit l'état t."""
    i = torch.arange(N_TRAJ)
    assert s.shape[0] == N_IID + N_TRAJ * PAS_TRAJ
    for t in range(PAS_TRAJ - 1):
        assert torch.equal(s[indices_traj(t + 1, i)], s2[indices_traj(t, i)]), t


def affiner_regle(compact, s, a, s2, graine, pas_, lr, horizon, lot=2048, lot_traj=256):
    libres = [{k: c[k].float().clone().requires_grad_(True) for k in ("echelle", "garde_val", "biais")}
              for c in compact["couches"]]
    opt = torch.optim.Adam([t for d in libres for t in d.values()], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, pas_)
    g = torch.Generator().manual_seed(graine)

    def avant(ss, aa):
        x = B.Modele.entrees(ss, aa)
        for k, (c, d) in enumerate(zip(compact["couches"], libres)):
            w = (c["codes"].float() * d["echelle"][:, None]).flatten()
            w = w.scatter(0, c["garde_idx"].long(), d["garde_val"]).reshape(c["codes"].shape)
            x = x @ w.T + d["biais"]
            if k < len(libres) - 1:
                x = torch.relu(x)
        return ss + B.VMAX * x

    for _ in range(pas_):
        j = torch.randint(0, s.shape[0], (lot,), generator=g)
        perte = (((avant(s[j], a[j]) - s2[j]) / B.VMAX) ** 2).mean()
        if horizon:
            i = torch.randint(0, N_TRAJ, (lot_traj,), generator=g)
            t0 = torch.randint(0, PAS_TRAJ - horizon + 1, (lot_traj,), generator=g)
            x = s[indices_traj(t0, i)]
            pertes = []
            for h in range(horizon):
                k = indices_traj(t0 + h, i)
                x = avant(x, a[k])
                pertes.append((((x - s2[k]) / B.VMAX) ** 2).mean())
            perte = perte + torch.stack(pertes).mean()
        opt.zero_grad()
        perte.backward()
        opt.step()
        sched.step()
    sortie = {"bits": compact["bits"], "couches": []}
    for c, d in zip(compact["couches"], libres):
        sortie["couches"].append({"codes": c["codes"].clone(), "garde_idx": c["garde_idx"].clone(),
                                  **{k: v.detach().to(torch.float16) for k, v in d.items()}})
    return sortie


def nom(pas_, lr, horizon):
    return f"affine_p{pas_ // 1000}k_lr{lr:.0e}_h{horizon}"


def cmd_lancer(_):
    R.SORTIE.mkdir(parents=True, exist_ok=True)
    f32, _ = B.charger(B.sortie() / "fp32.pt")
    journal = R.SORTIE / "affinage_regle.jsonl"
    ge, ne, gp = SELECTION
    for graine in GRAINES:
        s, a, s2 = R.M.donnees(graine=graine)
        verifier_disposition(s, s2)
        depart = B.quantifier(f32, 2, "gptq", B.FRACTION, "obs", donnees=(s, a))
        ref = torch.load(R.SORTIE / f"gptq_k1_{graine}.pt", weights_only=True)
        assert all(torch.equal(x[k], y[k]) for x, y in zip(depart["couches"], ref["couches"]) for k in x)
        opus = B.depuis_compact(torch.load(R.SORTIE / f"opus_{graine}.pt", weights_only=True))
        r = {"methode": "opus", "graine": graine, "selection": B.mesurer(opus, ge, ne, gp)}
        with journal.open("a") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"=== graine {graine} ; opus sélection {r['selection']['reussite']:.3f}", flush=True)
        for pas_, lr, horizon in GRILLE:
            t0 = time.time()
            c = affiner_regle(depart, s, a, s2, graine, pas_, lr, horizon)
            duree = time.time() - t0
            B.valider(c, bits=2, budget=B.budget_gardes())
            chemin = R.SORTIE / f"{nom(pas_, lr, horizon)}_{graine}.pt"
            torch.save(c, chemin)
            m = B.depuis_compact(c)
            r = {"methode": nom(pas_, lr, horizon), "graine": graine, "pas": pas_, "lr": lr, "horizon": horizon,
                 "selection": B.mesurer(m, ge, ne, gp), "erreurs": B.erreurs(m),
                 "secondes_compression": round(duree, 1), "empreinte": B.empreinte(chemin)}
            for p in R.PLANS:
                r[f"test_plan{p}"] = B.mesurer(m, R.GRAINE_TEST, R.N_TEST, p)
            with journal.open("a") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"{r['methode']:26s} sélection {r['selection']['reussite']:.3f}  erreur 8 pas "
                  f"{r['erreurs']['8_pas']:.3f}  ({duree:.0f} s)", flush=True)
    print("FINI", flush=True)


def cmd_tests(_):
    s, a, s2 = R.M.donnees(n_trans=N_IID, graine=7)
    verifier_disposition(s, s2)
    f32, _ = B.charger(B.sortie() / "fp32.pt")
    depart = B.quantifier(f32, 2, "gptq", B.FRACTION, "obs", donnees=(s[:5000], a[:5000]))
    for horizon in (0, 8):
        c = affiner_regle(depart, s, a, s2, 1, 3, 1e-3, horizon)
        B.valider(c, bits=2, budget=B.budget_gardes())
        assert all(torch.equal(x["codes"], y["codes"]) and torch.equal(x["garde_idx"], y["garde_idx"])
                   for x, y in zip(c["couches"], depart["couches"]))
    c0 = affiner_regle(depart, s, a, s2, 1, 0, 1e-3, 0)
    assert all(torch.equal(x[k], y[k]) for x, y in zip(c0["couches"], depart["couches"]) for k in x)
    print("ok : disposition des trajectoires, format valide, codes et positions figés, zéro pas = départ")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(required=True)
    sp.add_parser("tests").set_defaults(f=cmd_tests)
    sp.add_parser("lancer").set_defaults(f=cmd_lancer)
    x = p.parse_args()
    x.f(x)
