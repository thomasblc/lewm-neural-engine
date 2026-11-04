"""EXP-019 (pilote) : réglage fin d'un prédicteur ternaire avec poids gardés, leçon d'EXP-020.

Après quant.quantifier (version à échelle réduite, par exemple "predictor:2:clip:gptq:k1:obs"), chaque couche du
prédicteur vaut W = k x s hors des poids gardés, et la valeur gardée sur le masque G. On garde les entiers k et le
masque, et on règle par gradient : un facteur par ligne sur l'échelle (m), les valeurs gardées (K) et le biais. Cible :
la sortie du modèle float32 (prédicteur puis pred_proj, dernière position, celle que lit la planification), sur des
entrées du prédicteur enregistrées pendant une planification en pleine précision sur des épisodes de calibration
(run_grid.py --collecter-affinage). Le fichier produit ne contient que les poids et biais des couches réglées ;
l'échelle réglée s x m et les valeurs gardées passent par float16 (la mémoire annoncée), les biais restent en float32
comme dans toutes les versions.

Usage :
    python affinage.py --spec predictor:2:clip:gptq:k1:obs --calib CALIB --donnees DONNEES --out AFFINAGE \
        --pas 5000 --lot 256 --lr 1e-3 --graine 0 --device mps
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch

import quant


def charger_modeles(spec, calib, device):
    import run_grid as RG
    import stable_worldmodel as swm
    os.environ["LEWM_GPTQ_CALIB"] = str(Path(calib).resolve())
    cfg = RG.charger_config(50, device)
    prof = swm.wm.utils.load_pretrained(cfg.policy).eval()
    eleve, _ = quant.quantifier(swm.wm.utils.load_pretrained(cfg.policy), spec)
    details = {n: {k: (v.clone() if v is not None else None) for k, v in d.items()} for n, d in quant.DETAILS.items()}
    assert details and all(n.startswith("predictor.") for n in details), "réglage fin : prédicteur seul"
    return prof.to(device).requires_grad_(False), eleve.eval().to(device).requires_grad_(False), details


def poids(d, m, K):
    """W = k x s x m hors du masque, K sur le masque (même ordre que le masque à plat)."""
    W = (d["k"] * d["s"] * m[:, None]).flatten()
    if d["G"] is not None:
        W = W.scatter(0, d["G"].flatten().nonzero().squeeze(1), K)
    return W.reshape(d["k"].shape)


def depart(eleve, details, device):
    libres = {}
    for nom, d in details.items():
        mod = eleve.get_submodule(nom)
        dd = {k: (v.to(device) if v is not None else None) for k, v in d.items()}
        details[nom] = dd
        W = mod.weight.detach().reshape(dd["k"].shape)
        libres[nom] = {"m": torch.ones(W.shape[0], device=device, requires_grad=True),
                       "K": (W[dd["G"]].clone() if dd["G"] is not None else W.new_zeros(0)).requires_grad_(True)}
        if mod.bias is not None:
            libres[nom]["b"] = mod.bias.detach().clone().requires_grad_(True)
        assert torch.equal(poids(dd, libres[nom]["m"].detach(), libres[nom]["K"].detach()), W), nom
    return libres


def sortie_eleve(eleve, details, libres, x, c):
    params = {}
    for nom, d in details.items():
        court = nom[len("predictor."):]
        params[f"{court}.weight"] = poids(d, libres[nom]["m"], libres[nom]["K"]).reshape(
            eleve.get_submodule(nom).weight.shape)
        if "b" in libres[nom]:
            params[f"{court}.bias"] = libres[nom]["b"]
    y = torch.func.functional_call(eleve.predictor, params, (x, c), strict=False)
    return eleve.pred_proj(y[:, -1])


def affiner(prof, eleve, details, X, C, pas_, lot, lr, graine, device):
    """X, C : dictionnaires {longueur d'historique T : tenseur (N_T, T, ...)} ; à chaque pas, un groupe tiré au
    prorata de sa taille, puis un lot dans ce groupe."""
    libres = depart(eleve, details, device)
    groupes = sorted(X)
    poids_g = torch.tensor([float(X[T].shape[0]) for T in groupes])
    opt = torch.optim.Adam([t for d in libres.values() for t in d.values()], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(pas_, 1))
    g = torch.Generator().manual_seed(graine)
    pertes = []
    for i in range(pas_):
        T = groupes[int(torch.multinomial(poids_g, 1, generator=g))]
        j = torch.randint(0, X[T].shape[0], (lot,), generator=g)
        x, c = X[T][j].to(device).float(), C[T][j].to(device).float()
        with torch.no_grad():
            cible = prof.predict(x, c)[:, -1]
        perte = ((sortie_eleve(eleve, details, libres, x, c) - cible) ** 2).mean()
        opt.zero_grad()
        perte.backward()
        opt.step()
        sched.step()
        pertes.append(perte.item())
        if i % 500 == 0 or i == pas_ - 1:
            print(f"[affinage] pas {i + 1}/{pas_} : perte {sum(pertes[-50:]) / len(pertes[-50:]):.4e}", flush=True)
    return libres, pertes



@torch.no_grad()
def perte_controle(prof, eleve, details, libres, Xv, Cv, device, lot=512):
    """Perte moyenne (par ligne) sur les lignes tenues à l'écart, tous groupes T confondus."""
    somme, n = 0.0, 0
    for T in Xv:
        for i in range(0, Xv[T].shape[0], lot):
            x, c = Xv[T][i:i + lot].to(device).float(), Cv[T][i:i + lot].to(device).float()
            e = (sortie_eleve(eleve, details, libres, x, c) - prof.predict(x, c)[:, -1]) ** 2
            somme += e.mean(1).sum().item()
            n += x.shape[0]
    return somme / max(n, 1)


@torch.no_grad()
def exporter(eleve, details, libres):
    """Poids finaux : échelle s x m et valeurs gardées arrondies en float16, entiers et masque inchangés."""
    etat = {}
    for nom, d in details.items():
        mod = eleve.get_submodule(nom)
        s16 = (d["s"] * libres[nom]["m"][:, None]).to(torch.float16).to(d["s"].dtype)
        W = (d["k"] * s16).flatten()
        if d["G"] is not None:
            W = W.scatter(0, d["G"].flatten().nonzero().squeeze(1),
                          libres[nom]["K"].to(torch.float16).to(W.dtype))
        etat[f"{nom}.weight"] = W.reshape(mod.weight.shape).cpu()
        if "b" in libres[nom]:
            etat[f"{nom}.bias"] = libres[nom]["b"].detach().cpu()
    return etat


@torch.no_grad()
def verifier_format(etat, details):
    """Refuse un fichier dont les poids ne respectent pas la mémoire annoncée : hors du masque, W = k x s' avec les
    mêmes entiers k et une échelle s' par ligne (float16) ; au plus les poids gardés du masque d'origine."""
    noms = {n[:-len(".weight")] for n in etat if n.endswith(".weight")}
    if noms != set(details):
        raise ValueError(f"couches du fichier différentes des couches quantifiées : {sorted(noms ^ set(details))[:5]}")
    for nom, d in details.items():
        k = d["k"].cpu()
        W = etat[f"{nom}.weight"].reshape(k.shape).double()
        G = d["G"].cpu() if d["G"] is not None else torch.zeros_like(k, dtype=torch.bool)
        hors = ~G
        if (W[hors & (k == 0)] != 0).any():
            raise ValueError(f"{nom} : poids non nul là où l'entier vaut 0")
        nz = hors & (k != 0)
        ratio = torch.where(nz, W / k.double(), torch.zeros_like(W))
        ref = ratio.abs().amax(1, keepdim=True)
        if not torch.allclose(ratio.abs()[nz], ref.expand_as(ratio)[nz], rtol=0, atol=0):
            raise ValueError(f"{nom} : plusieurs échelles sur une même ligne")
        if not torch.equal(ref.squeeze(1).to(torch.float16).double(), ref.squeeze(1)):
            raise ValueError(f"{nom} : échelle hors float16")
        if G.any() and not torch.equal(W[G].to(torch.float16).double(), W[G]):
            raise ValueError(f"{nom} : valeur gardée hors float16")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--donnees", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pas", type=int, default=5000)
    ap.add_argument("--lot", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--graine", type=int, default=0)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--validation", type=float, default=0.05, help="part des lignes tenue à l'écart (perte de contrôle)")
    a = ap.parse_args()
    t0 = time.time()
    prof, eleve, details = charger_modeles(a.spec, a.calib, a.device)
    D = torch.load(a.donnees, map_location="cpu")
    g = torch.Generator().manual_seed(a.graine + 1)
    X, C, Xv, Cv = {}, {}, {}, {}
    for T in D["X"]:                                  # lignes tirées au hasard, tenues à l'écart du réglage
        ordre = torch.randperm(D["X"][T].shape[0], generator=g)
        nv = int(round(a.validation * len(ordre)))
        Xv[T], Cv[T] = D["X"][T][ordre[:nv]], D["C"][T][ordre[:nv]]
        X[T], C[T] = D["X"][T][ordre[nv:]], D["C"][T][ordre[nv:]]
    print(f"[affinage] {len(details)} couches ; exemples par longueur d'historique : "
          + ", ".join(f"T={T} : {X[T].shape[0]}" for T in sorted(X)), flush=True)
    libres0 = depart(eleve, details, a.device)          # place aussi les détails sur l'appareil
    controle_avant = perte_controle(prof, eleve, details, libres0, Xv, Cv, a.device)
    libres, pertes = affiner(prof, eleve, details, X, C, a.pas, a.lot, a.lr, a.graine, a.device)
    controle_apres = perte_controle(prof, eleve, details, libres, Xv, Cv, a.device)
    print(f"[affinage] perte de contrôle (lignes tenues à l'écart) : {controle_avant:.4e} -> {controle_apres:.4e}", flush=True)
    etat = exporter(eleve, details, libres)
    verifier_format(etat, details)
    sha = hashlib.sha256(Path(a.donnees).read_bytes()).hexdigest()
    torch.save({"etat": etat, "spec": a.spec, "pas": a.pas, "lot": a.lot, "lr": a.lr, "graine": a.graine,
                "donnees_sha256": sha, "perte_debut": sum(pertes[:50]) / max(1, len(pertes[:50])),
                "perte_fin": sum(pertes[-50:]) / max(1, len(pertes[-50:])), "validation": a.validation,
                "controle_avant": controle_avant, "controle_apres": controle_apres}, a.out)
    print(f"[affinage] écrit {a.out} ({time.time() - t0:.0f} s)", flush=True)
    print(json.dumps({"perte_debut": pertes[0] if pertes else None, "perte_fin": pertes[-1] if pertes else None}))


if __name__ == "__main__":
    main()
