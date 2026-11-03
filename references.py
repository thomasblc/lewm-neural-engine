"""EXP-020, amendement 1 : références au même budget de mémoire que la méthode d'Opus (ternaire, 194 poids gardés en
float16, échelles et biais en float16), sur les mêmes données (`methode.donnees` d'agent-opus-1, 240 000 transitions).

- gptq_k1 : GPTQ ternaire et poids gardés SpQR (banc.quantifier), calibré sur ces données ;
- dfq : égalisation entre couches (Nagel et al., arXiv:1906.04721), GPTQ et poids gardés SpQR, puis correction de
  biais empirique couche par couche ;
- adaquant : reconstruction couche par couche par gradient (Hubara et al., arXiv:2006.10518), codes réappris par
  « straight-through », cible : pré-activations du float32 ;
- affine : banc.affiner (codes figés), 6 000 pas, depuis gptq_k1 ;
- qat : réseau entier, codes réappris par « straight-through », perte à un pas contre le vrai simulateur ;
- opus : recette C' de la méthode d'Opus, rejouée sans LLM.

Usage :
    python references.py tests                      # contrôles rapides
    python references.py lancer --graines 1111 2222 3333
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch

import banc as B
import garder
import gptq

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI / "agent-opus-1"))
import methode as M  # noqa: E402

SORTIE = B.RESULTATS / "c96" / "references-2"
GRAINE_TEST, N_TEST, PLANS = 10485, 300, (11, 12)
LR_LATENT, LR_LIBRE = 1e-4, 1e-3


def ste(x):
    return x + (x.round() - x).detach()


# --- dfq ---------------------------------------------------------------------------------------------------------------

@torch.no_grad()
def egaliser(f32, tours=50):
    """Égalisation entre couches de DFQ : pour chaque neurone caché i entre W1 et W2, s_i = sqrt(r1_i / r2_i) avec
    r1_i = max|W1[i, :]| et r2_i = max|W2[:, i]| ; W1[i, :] et b1[i] divisés par s_i, W2[:, i] multiplié par s_i.
    La ReLU étant positivement homogène, la fonction du réseau ne change pas."""
    m = copy.deepcopy(f32)
    L = m.couches
    for _ in range(tours):
        for k in range(len(L) - 1):
            W1, b1, W2 = L[k].weight, L[k].bias, L[k + 1].weight
            r1, r2 = W1.abs().amax(1), W2.abs().amax(0)
            s = torch.where((r1 > 0) & (r2 > 0), (r1 / r2.clamp_min(1e-30)).sqrt(), torch.ones_like(r1))
            W1.div_(s[:, None])
            b1.div_(s)
            W2.mul_(s[None, :])
    return m


@torch.no_grad()
def corriger_biais(compact, ref, s, a):
    """Correction de biais empirique : à chaque couche, dans l'ordre, le biais compressé reçoit l'écart moyen entre la
    pré-activation du modèle `ref` et celle du réseau compressé (entrées déjà compressées)."""
    out = copy.deepcopy(compact)
    Xf = B.Modele.entrees(s, a).double()
    Xq = Xf.clone()
    n = len(ref.couches)
    for k, (c, lin) in enumerate(zip(out["couches"], ref.couches)):
        Wq = B.depuis_couche(c, lin.weight.shape).double()
        Yf = Xf @ lin.weight.double().T + lin.bias.double()
        c["biais"] = (c["biais"].double() + (Yf - (Xq @ Wq.T + c["biais"].double())).mean(0)).to(torch.float16)
        Yq = Xq @ Wq.T + c["biais"].double()
        Xf, Xq = (torch.relu(Yf), torch.relu(Yq)) if k < n - 1 else (Yf, Yq)
    return out


def dfq(f32, s, a):
    eq = egaliser(f32)
    return corriger_biais(B.quantifier(eq, 2, "gptq", B.FRACTION, "obs", donnees=(s, a)), eq, s, a)


# --- straight-through : couche par couche et réseau entier -------------------------------------------------------------

def depart(W, h):
    """Échelle réduite d'EXP-011 et masque SpQR, comme banc.quantifier, pour une couche (W en float64)."""
    f = gptq.facteur_reduit(W, h, 2)
    s0, qmax = gptq.echelle(W, 2)
    ech = (s0 * f).to(torch.float16).double()
    return ech.squeeze(1).float(), garder.masque(W, h, ech, qmax, B.FRACTION, "obs")


def poids_ste(V, e, G, kv):
    idx = G.flatten().nonzero().squeeze(1)
    q = e[:, None] * ste((V / e[:, None]).clamp(-1, 1))
    return q.flatten().scatter(0, idx, kv).reshape(V.shape)


def vers_compact(V, e, G, kv, b):
    with torch.no_grad():
        idx = G.flatten().nonzero().squeeze(1)
        codes = torch.where(G, torch.zeros_like(V), (V / e[:, None]).clamp(-1, 1).round()).to(torch.int8)
        return {"codes": codes, "echelle": e.to(torch.float16), "garde_idx": idx, "garde_val": kv.to(torch.float16),
                "biais": b.to(torch.float16)}


def optimiseur(latents, libres, pas_):
    opt = torch.optim.Adam([{"params": latents, "lr": LR_LATENT}, {"params": libres, "lr": LR_LIBRE}])
    return opt, torch.optim.lr_scheduler.CosineAnnealingLR(opt, pas_)


def adaquant(f32, s, a, graine, pas_=2000, lot=4096):
    g = torch.Generator().manual_seed(graine)
    Xf = B.Modele.entrees(s, a)
    Xq = Xf.clone()
    out = {"bits": 2, "couches": []}
    n = len(f32.couches)
    for k, lin in enumerate(f32.couches):
        W, b = lin.weight.detach(), lin.bias.detach()
        Yf = Xf @ W.T + b
        h = Xq.double().T @ Xq.double() / Xq.shape[0]
        e0, G = depart(W.double(), h)
        V = W.clone().requires_grad_(True)
        e = e0.clone().requires_grad_(True)
        bb = b.clone().requires_grad_(True)
        kv = W[G].clone().requires_grad_(True)
        opt, sched = optimiseur([V, e], [bb, kv], pas_)
        for _ in range(pas_):
            j = torch.randint(0, Xq.shape[0], (lot,), generator=g)
            perte = ((Xq[j] @ poids_ste(V, e, G, kv).T + bb - Yf[j]) ** 2).mean()
            opt.zero_grad()
            perte.backward()
            opt.step()
            sched.step()
        c = vers_compact(V, e, G, kv, bb)
        out["couches"].append(c)
        with torch.no_grad():
            Yq = Xq @ B.depuis_couche(c, W.shape).T + c["biais"].float()
        Xf, Xq = (torch.relu(Yf), torch.relu(Yq)) if k < n - 1 else (Yf, Yq)
    return out


def qat(f32, s, a, s2, graine, depart_compact, pas_=6000, lot=2048):
    """Réseau entier ; masques gardés et échelles de départ pris dans `depart_compact` (gptq_k1 de la même graine)."""
    g = torch.Generator().manual_seed(graine)
    params = []
    for lin, c in zip(f32.couches, depart_compact["couches"]):
        G = torch.zeros(lin.weight.numel(), dtype=torch.bool)
        G[c["garde_idx"]] = True
        G = G.reshape(lin.weight.shape)
        params.append({"V": lin.weight.detach().clone().requires_grad_(True),
                       "e": c["echelle"].float().clone().requires_grad_(True), "G": G,
                       "kv": lin.weight.detach()[G].clone().requires_grad_(True),
                       "b": lin.bias.detach().clone().requires_grad_(True)})
    opt, sched = optimiseur([t for p in params for t in (p["V"], p["e"])],
                            [t for p in params for t in (p["b"], p["kv"])], pas_)

    def avant(ss, aa):
        x = B.Modele.entrees(ss, aa)
        for k, p in enumerate(params):
            x = x @ poids_ste(p["V"], p["e"], p["G"], p["kv"]).T + p["b"]
            if k < len(params) - 1:
                x = torch.relu(x)
        return ss + B.VMAX * x

    for _ in range(pas_):
        j = torch.randint(0, s.shape[0], (lot,), generator=g)
        perte = (((avant(s[j], a[j]) - s2[j]) / B.VMAX) ** 2).mean()
        opt.zero_grad()
        perte.backward()
        opt.step()
        sched.step()
    return {"bits": 2, "couches": [vers_compact(p["V"], p["e"], p["G"], p["kv"], p["b"]) for p in params]}


# --- recette d'Opus ----------------------------------------------------------------------------------------------------

def opus(graine):
    rep = M.repartition_globale(equilibrer=True, relu=True, graine=graine)
    return M.compresser(gardes=rep, equilibrer=True, relu=True, imp=True, graine=graine)


# --- mesures -----------------------------------------------------------------------------------------------------------

def mesures(compact):
    B.valider(compact, bits=2, budget=B.budget_gardes())
    m = B.depuis_compact(compact)
    r = {"calibration": B.mesurer(m), "erreurs": B.erreurs(m)}
    for gp in PLANS:
        r[f"test_plan{gp}"] = B.mesurer(m, GRAINE_TEST, N_TEST, gp)
    return r


def cmd_lancer(args):
    SORTIE.mkdir(parents=True, exist_ok=True)
    f32, _ = B.charger(B.sortie() / "fp32.pt")
    journal = SORTIE / "resultats.jsonl"
    for graine in args.graines:
        s, a, s2 = M.donnees(graine=graine)
        print(f"=== graine {graine} : {s.shape[0]} transitions", flush=True)
        faits = {}
        etapes = [("gptq_k1", lambda: B.quantifier(f32, 2, "gptq", B.FRACTION, "obs", donnees=(s, a))),
                  ("dfq", lambda: dfq(f32, s, a)),
                  ("adaquant", lambda: adaquant(f32, s, a, graine)),
                  ("affine", lambda: B.affiner(faits["gptq_k1"], pas_=6000, donnees=(s, a, s2), graine=graine)),
                  ("qat", lambda: qat(f32, s, a, s2, graine, faits["gptq_k1"])),
                  ("opus", lambda: opus(graine))]
        for nom, faire in etapes:
            if args.methodes and nom not in args.methodes and nom != "gptq_k1":
                continue
            t0 = time.time()
            faits[nom] = faire()
            duree = time.time() - t0
            chemin = SORTIE / f"{nom}_{graine}.pt"
            torch.save(faits[nom], chemin)
            r = mesures(faits[nom])
            r.update({"methode": nom, "graine": graine, "secondes_compression": round(duree, 1),
                      "empreinte": B.empreinte(chemin)})
            with journal.open("a") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"{nom:9s} calib {r['calibration']['reussite']:.2f}  test {r['test_plan11']['reussite']:.3f} / "
                  f"{r['test_plan12']['reussite']:.3f}  erreur 1 pas {r['erreurs']['un_pas']:.3f}  "
                  f"({duree:.0f} s)", flush=True)
    print("FINI", flush=True)


def cmd_tests(_):
    f32, _ = B.charger(B.sortie() / "fp32.pt")
    s, a, s2 = B.transitions(5000, 99)
    eq = egaliser(f32)
    with torch.no_grad():
        ecart = (eq(s, a) - f32(s, a)).abs().max().item()
        r1 = eq.couches[1].weight.abs().amax(1)
        r2 = eq.couches[2].weight.abs().amax(0)
    assert ecart < 1e-5, ecart
    assert torch.allclose(r1, r2, rtol=1e-3), (r1 - r2).abs().max()
    print(f"ok égalisation : même fonction (écart max {ecart:.1e}), plages égales entre couches 1 et 2")
    c0 = B.quantifier(f32, 2, "gptq", B.FRACTION, "obs", donnees=(s, a))
    cb = corriger_biais(c0, f32, s, a)
    with torch.no_grad():
        x = B.Modele.entrees(s, a)
        y0 = B.depuis_compact(cb).couches[0](x)
        yf = f32.couches[0](x)
    assert (y0 - yf).mean(0).abs().max() < 1e-2
    print("ok correction de biais : écart moyen de la couche 0 ramené à zéro (à float16 près)")
    for nom, c in [("dfq", dfq(f32, s, a)), ("adaquant", adaquant(f32, s, a, 1, pas_=5)),
                   ("qat", qat(f32, s, a, s2, 1, c0, pas_=5))]:
        n = B.valider(c, bits=2, budget=B.budget_gardes())
        assert all(set(x["codes"].unique().tolist()) <= {-1, 0, 1} for x in c["couches"])
        print(f"ok {nom} : format valide, {n} poids gardés")
    ca = adaquant(f32, s, a, 1, pas_=0)
    with torch.no_grad():
        for x, lin, cq in zip(ca["couches"], f32.couches, c0["couches"]):
            W = lin.weight.detach()
            attendu = (W / x["echelle"].float()[:, None]).clamp(-1, 1).round()
            garde = torch.zeros(W.numel(), dtype=torch.bool)
            garde[x["garde_idx"]] = True
            assert torch.equal(x["codes"].flatten()[~garde].float(), attendu.flatten()[~garde])
    print("ok adaquant sans pas : arrondi au plus proche des poids d'origine (départ RTN)")
    print("tous les tests passent")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(required=True)
    sp.add_parser("tests").set_defaults(f=cmd_tests)
    lc = sp.add_parser("lancer")
    lc.add_argument("--graines", type=int, nargs="+", default=[1111, 2222, 3333])
    lc.add_argument("--methodes", nargs="*")
    lc.set_defaults(f=cmd_lancer)
    x = p.parse_args()
    x.f(x)
