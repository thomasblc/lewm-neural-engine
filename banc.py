"""EXP-020 (essai exploratoire) : banc d'essai minuscule pour compresser très fort un world model qui planifie, et pour
voir si un LLM sait réparer les poids gardés.

Monde « Pousse » : un disque agent (rayon RA) pousse un disque bloc (rayon RB) dans le carré [0, 1]^2. À chaque pas,
l'agent avance de VMAX x a (a dans [-1, 1]^2) ; s'il chevauche le bloc, le bloc est poussé le long de la normale
jusqu'au contact ; un mur arrête le bloc, et l'agent recule alors jusqu'au contact. État : positions de l'agent et du
bloc (4 nombres). Réussite : le bloc passe à moins de TOL du but en au plus T pas.

World model : MLP 6 -> 96 -> 96 -> 96 -> 4 (19 392 poids), entraîné sur des transitions tirées au hasard, qui prédit
le déplacement de l'état. Planification : CEM en horizon glissant (comme LeWM, en plus petit), coût sur tous les pas
de l'horizon : distance bloc-but, plus la distance de l'agent au point de poussée derrière le bloc.

Format compressé (« compact ») : pour chaque couche, des codes entiers dans [-qmax, qmax] (ternaire : qmax = 1), une
échelle float16 par ligne, des poids gardés hors grille (indices à plat et valeurs float16) et le biais en float16.
Mémoire : bits x poids + 16 par ligne + 32 par poids gardé (16 de valeur, 16 d'adresse) + 16 par biais.

Tout tourne en float32 sur CPU, déterministe à graine fixée. Usage : voir `python banc.py -h`.
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch

import garder
import gptq

ICI = Path(__file__).resolve().parent
RESULTATS = ICI.parent / "resultats" / "exp020-banc"

RA, RB, VMAX, TOL, T = 0.04, 0.08, 0.04, 0.03, 80
CACHE = 96          # 256 et 128 : l'affinage par gradient atteint le float32, rien à battre
H_PLAN, N_PLAN, ITERS, ELITES, K_EXEC, SIGMA0 = 8, 128, 3, 16, 8, 0.7
GRAINE_DONNEES, GRAINE_INIT, GRAINE_CALIB_Q, GRAINE_ERREUR = 0, 0, 123, 7
GRAINE_EPISODES_CALIB, GRAINE_PLAN = 1, 11
N_CALIB = 100
FRACTION = 0.01
BUDGET_ESSAIS = 30

torch.set_num_threads(max(1, torch.get_num_threads()))


# --- monde -----------------------------------------------------------------------------------------------------------

def pas(s, a):
    """s : (n, 4) positions agent puis bloc ; a : (n, 2). Renvoie l'état suivant."""
    a = a.clamp(-1, 1)
    pa = (s[:, :2] + VMAX * a).clamp(RA, 1 - RA)
    pb = s[:, 2:]
    d = pb - pa
    n = d.norm(dim=1, keepdim=True)
    contact = n < RA + RB
    pb = torch.where(contact, pa + d / n.clamp_min(1e-9) * (RA + RB), pb).clamp(RB, 1 - RB)
    d = pa - pb
    n = d.norm(dim=1, keepdim=True)
    chevauche = n < RA + RB - 1e-6
    pa = torch.where(chevauche, (pb + d / n.clamp_min(1e-9) * (RA + RB)).clamp(RA, 1 - RA), pa)
    return torch.cat([pa, pb], 1)


def etats_aleatoires(n, g):
    """Moitié des agents loin du bloc, moitié à moins de deux pas du contact (pour voir des poussées)."""
    pb = RB + (1 - 2 * RB) * torch.rand(n, 2, generator=g)
    loin = RA + (1 - 2 * RA) * torch.rand(n, 2, generator=g)
    ang = 2 * math.pi * torch.rand(n, generator=g)
    r = RA + RB + 2 * VMAX * torch.rand(n, generator=g)
    pres = (pb + r[:, None] * torch.stack([ang.cos(), ang.sin()], 1)).clamp(RA, 1 - RA)
    pa = torch.where((torch.rand(n, generator=g) < 0.5)[:, None], pres, loin)
    return pas(torch.cat([pa, pb], 1), torch.zeros(n, 2))


def transitions(n, graine):
    g = torch.Generator().manual_seed(graine)
    s = etats_aleatoires(n, g)
    a = 2 * torch.rand(n, 2, generator=g) - 1
    return s, a, pas(s, a)


def episodes(n, graine):
    """Départs et buts : bloc et but dans [0,2 ; 0,8]^2, à au moins 0,25 l'un de l'autre ; agent n'importe où."""
    g = torch.Generator().manual_seed(graine)
    pb = 0.2 + 0.6 * torch.rand(n, 2, generator=g)
    but = 0.2 + 0.6 * torch.rand(n, 2, generator=g)
    for _ in range(1000):
        mauvais = (but - pb).norm(dim=1) < 0.25
        if not mauvais.any():
            break
        but[mauvais] = 0.2 + 0.6 * torch.rand(int(mauvais.sum()), 2, generator=g)
    pa = RA + (1 - 2 * RA) * torch.rand(n, 2, generator=g)
    return pas(torch.cat([pa, pb], 1), torch.zeros(n, 2)), but


# --- world model -----------------------------------------------------------------------------------------------------

def sortie():
    """Dossier des résultats pour la taille de couche cachée en vigueur."""
    return RESULTATS / f"c{CACHE}"


class Modele(torch.nn.Module):
    def __init__(self, cache=None):
        super().__init__()
        cache = cache or CACHE
        self.couches = torch.nn.ModuleList([torch.nn.Linear(6, cache), torch.nn.Linear(cache, cache),
                                            torch.nn.Linear(cache, cache), torch.nn.Linear(cache, 4)])

    @staticmethod
    def entrees(s, a):
        return torch.cat([s * 2 - 1, a], -1)

    def forward(self, s, a):
        x = self.entrees(s, a)
        for i, c in enumerate(self.couches):
            x = c(x)
            if i < len(self.couches) - 1:
                x = torch.relu(x)
        return s + VMAX * x


def entrainer(n=400_000, epoques=30, lot=1024, lr=1e-3):
    torch.manual_seed(GRAINE_INIT)
    m = Modele()
    s, a, s2 = transitions(n, GRAINE_DONNEES)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    total = epoques * (n // lot)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, total)
    g = torch.Generator().manual_seed(GRAINE_DONNEES + 1)
    for ep in range(epoques):
        perm = torch.randperm(n, generator=g)
        somme = 0.0
        for i in range(n // lot):
            j = perm[i * lot:(i + 1) * lot]
            perte = (((m(s[j], a[j]) - s2[j]) / VMAX) ** 2).mean()
            opt.zero_grad()
            perte.backward()
            opt.step()
            sched.step()
            somme += perte.item()
        if ep % 5 == 4 or ep == epoques - 1:
            print(f"   époque {ep + 1} : perte {somme / (n // lot):.2e}", flush=True)
    return m.eval()


# --- planification et évaluation -------------------------------------------------------------------------------------

@torch.no_grad()
def cout(modele, s, acts, but):
    """s (B, 4), acts (B, H, 2), but (B, 2) : coût cumulé sur tous les pas de l'horizon."""
    x, c = s, torch.zeros(s.shape[0])
    for h in range(acts.shape[1]):
        x = modele(x, acts[:, h])
        pa, pb = x[:, :2], x[:, 2:]
        v = but - pb
        dist = v.norm(dim=1)
        point = pb - (RA + RB) * v / dist.clamp_min(1e-6)[:, None]
        c = c + dist + 0.5 * (pa - point).norm(dim=1)
    return c


@torch.no_grad()
def evaluer(modele, s0, but, graine_plan=GRAINE_PLAN):
    """Réussite de chaque épisode et distance finale bloc-but. Même bruit de CEM pour toutes les versions."""
    g = torch.Generator().manual_seed(graine_plan)
    E, s = s0.shape[0], s0.clone()
    mu = torch.zeros(E, H_PLAN, 2)
    reussi = torch.zeros(E, dtype=torch.bool)
    t = 0
    while t < T:
        sigma = torch.full((E, H_PLAN, 2), SIGMA0)
        for _ in range(ITERS):
            acts = (mu[:, None] + sigma[:, None] * torch.randn(E, N_PLAN, H_PLAN, 2, generator=g)).clamp(-1, 1)
            c = cout(modele, s.repeat_interleave(N_PLAN, 0), acts.reshape(E * N_PLAN, H_PLAN, 2),
                     but.repeat_interleave(N_PLAN, 0)).reshape(E, N_PLAN)
            idx = c.topk(ELITES, 1, largest=False).indices
            elites = torch.gather(acts, 1, idx[:, :, None, None].expand(E, ELITES, H_PLAN, 2))
            mu, sigma = elites.mean(1), elites.std(1).clamp_min(0.05)
        for k in range(K_EXEC):
            if t >= T:
                break
            s = pas(s, mu[:, k])
            t += 1
            reussi |= (s[:, 2:] - but).norm(dim=1) < TOL
        mu = torch.cat([mu[:, K_EXEC:], torch.zeros(E, K_EXEC, 2)], 1)
    return reussi, (s[:, 2:] - but).norm(dim=1)


@torch.no_grad()
def erreurs(modele, n=4096, horizon=8):
    """Erreur de prédiction en pas de VMAX : à un pas (transitions tirées au hasard) et après `horizon` pas en boucle
    ouverte (actions au hasard, départs tirés comme à l'entraînement). Graine fixe, distincte des autres."""
    s, a, s2 = transitions(n, GRAINE_ERREUR)
    un = ((modele(s, a) - s2).norm(dim=1) / VMAX).mean().item()
    g = torch.Generator().manual_seed(GRAINE_ERREUR + 1)
    vrai = etats_aleatoires(n, g)
    pred = vrai.clone()
    for _ in range(horizon):
        act = 2 * torch.rand(n, 2, generator=g) - 1
        vrai, pred = pas(vrai, act), modele(pred, act)
    return {"un_pas": un, f"{horizon}_pas": ((pred - vrai).norm(dim=1) / VMAX).mean().item()}


# --- format compact et quantification --------------------------------------------------------------------------------

def tailles():
    return [tuple(c.weight.shape) for c in Modele().couches]


def n_poids():
    return sum(o * i for o, i in tailles())


def budget_gardes(fraction=FRACTION):
    return int(round(fraction * n_poids()))


def valider(compact, bits=None, budget=None):
    """Vérifie que `compact` est représentable dans la mémoire annoncée. Lève ValueError sinon."""
    if not isinstance(compact, dict) or "couches" not in compact or "bits" not in compact:
        raise ValueError("compact : dict avec les clés 'bits' et 'couches'")
    b = int(compact["bits"])
    if bits is not None and b != bits:
        raise ValueError(f"bits = {b}, attendu {bits}")
    qmax = 2 ** (b - 1) - 1
    if len(compact["couches"]) != len(tailles()):
        raise ValueError("nombre de couches")
    gardes = 0
    for k, ((o, i), c) in enumerate(zip(tailles(), compact["couches"])):
        codes, ech, idx, val, biais = (c[x] for x in ("codes", "echelle", "garde_idx", "garde_val", "biais"))
        if tuple(codes.shape) != (o, i) or codes.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError(f"couche {k} : codes entiers de forme {(o, i)}")
        if codes.abs().max() > qmax:
            raise ValueError(f"couche {k} : codes hors de [-{qmax}, {qmax}]")
        if tuple(ech.shape) != (o,) or tuple(biais.shape) != (o,) or val.ndim != 1 or idx.ndim != 1 \
                or idx.shape != val.shape:
            raise ValueError(f"couche {k} : formes des échelles, du biais ou des poids gardés")
        if idx.numel() and (idx.min() < 0 or idx.max() >= o * i or torch.unique(idx).numel() != idx.numel()):
            raise ValueError(f"couche {k} : indices gardés hors limites ou en double")
        for nom, x in (("echelle", ech), ("garde_val", val), ("biais", biais)):
            if not torch.isfinite(x.to(torch.float16).float()).all():
                raise ValueError(f"couche {k} : {nom} non fini en float16")
        gardes += idx.numel()
    if budget is not None and gardes > budget:
        raise ValueError(f"{gardes} poids gardés, budget {budget}")
    return gardes


def depuis_compact(compact):
    """Construit le Modele ; échelles, poids gardés et biais passent par float16 (la mémoire annoncée)."""
    valider(compact)
    m = Modele()
    with torch.no_grad():
        for lin, c in zip(m.couches, compact["couches"]):
            w = c["codes"].float() * c["echelle"].to(torch.float16).float()[:, None]
            w = w.flatten()
            w[c["garde_idx"].long()] = c["garde_val"].to(torch.float16).float()
            lin.weight.copy_(w.reshape(lin.weight.shape))
            lin.bias.copy_(c["biais"].to(torch.float16).float())
    return m.eval()


def bits_par_poids(compact):
    n = n_poids()
    lignes = sum(o for o, _ in tailles())
    gardes = sum(c["garde_idx"].numel() for c in compact["couches"])
    return (int(compact["bits"]) * n + 16 * lignes + 32 * gardes + 16 * lignes) / n


@torch.no_grad()
def quantifier(modele, bits, methode="gptq", fraction=0.0, critere="obs", n=20_000, donnees=None):
    """Couche par couche, dans l'ordre : échelle réduite d'EXP-011 (facteur choisi sur la calibration), poids gardés
    par garder.masque, arrondis par RTN ou GPTQ. Les entrées de chaque couche passent par les couches déjà
    quantifiées (GPTQ séquentiel). `donnees` : (s, a) de calibration ; par défaut n transitions tirées au hasard."""
    s, a = donnees if donnees is not None else transitions(n, GRAINE_CALIB_Q)[:2]
    x = Modele.entrees(s, a).double()
    sortie = {"bits": bits, "couches": []}
    for k, lin in enumerate(modele.couches):
        w = lin.weight.detach().double()
        h = x.T @ x / x.shape[0]
        f = gptq.facteur_reduit(w, h, bits)
        s0, qmax = gptq.echelle(w, bits)
        ech = (s0 * f).to(torch.float16).double()
        G = garder.masque(w, h, ech, qmax, fraction, critere) if fraction > 0 else torch.zeros_like(w, dtype=torch.bool)
        if methode == "gptq":
            q = gptq.gptq_couche(w, h, bits, s=ech, garder=G)
        elif methode == "rtn":
            q = torch.where(G, w, (w / ech).round().clamp(-qmax, qmax) * ech)
        else:
            raise ValueError(methode)
        codes = torch.where(G, torch.zeros_like(q), (q / ech).round().clamp(-qmax, qmax)).to(torch.int8)
        idx = G.flatten().nonzero().squeeze(1)
        c = {"codes": codes, "echelle": ech.squeeze(1).to(torch.float16), "garde_idx": idx,
             "garde_val": q.flatten()[idx].to(torch.float16), "biais": lin.bias.detach().to(torch.float16)}
        sortie["couches"].append(c)
        wq = depuis_couche(c, w.shape).double()
        x = x @ wq.T + c["biais"].double()
        if k < len(modele.couches) - 1:
            x = torch.relu(x)
    return sortie


def depuis_couche(c, forme):
    w = (c["codes"].float() * c["echelle"].float()[:, None]).flatten()
    w[c["garde_idx"].long()] = c["garde_val"].float()
    return w.reshape(forme)


def affiner(compact, pas_=3000, lot=2048, lr=3e-4, n=200_000, donnees=None, graine=GRAINE_DONNEES + 2):
    """Référence par gradient : à codes et positions gardées fixes, ajuste les valeurs gardées, les échelles et les
    biais (Adam, erreur à un pas sur des transitions d'entraînement), puis repasse en float16. `donnees` : (s, a, s2) ;
    par défaut n transitions tirées au hasard."""
    s, a, s2 = donnees if donnees is not None else transitions(n, GRAINE_DONNEES)
    n = s.shape[0]
    libres = []
    for c in compact["couches"]:
        libres.append({k: c[k].float().clone().requires_grad_(True) for k in ("echelle", "garde_val", "biais")})
    opt = torch.optim.Adam([t for d in libres for t in d.values()], lr=lr)
    g = torch.Generator().manual_seed(graine)

    def avant(ss, aa):
        x = Modele.entrees(ss, aa)
        for k, (c, d) in enumerate(zip(compact["couches"], libres)):
            w = (c["codes"].float() * d["echelle"][:, None]).flatten()
            w = w.scatter(0, c["garde_idx"].long(), d["garde_val"]).reshape(c["codes"].shape)
            x = x @ w.T + d["biais"]
            if k < len(libres) - 1:
                x = torch.relu(x)
        return ss + VMAX * x

    for i in range(pas_):
        j = torch.randint(0, n, (lot,), generator=g)
        perte = (((avant(s[j], a[j]) - s2[j]) / VMAX) ** 2).mean()
        opt.zero_grad()
        perte.backward()
        opt.step()
    sortie = {"bits": compact["bits"], "couches": []}
    for c, d in zip(compact["couches"], libres):
        sortie["couches"].append({"codes": c["codes"].clone(), "garde_idx": c["garde_idx"].clone(),
                                  **{k: v.detach().to(torch.float16) for k, v in d.items()}})
    return sortie


# --- commandes -------------------------------------------------------------------------------------------------------

def charger(chemin):
    obj = torch.load(chemin, map_location="cpu", weights_only=True)
    if isinstance(obj, dict) and "couches" in obj:
        return depuis_compact(obj), obj
    m = Modele()
    m.load_state_dict(obj)
    return m.eval(), None


def empreinte(chemin):
    return hashlib.sha256(Path(chemin).read_bytes()).hexdigest()[:16]


def mesurer(modele, graine_episodes=GRAINE_EPISODES_CALIB, n=N_CALIB, graine_plan=GRAINE_PLAN):
    s0, but = episodes(n, graine_episodes)
    t0 = time.time()
    reussi, dist = evaluer(modele, s0, but, graine_plan)
    return {"reussite": reussi.float().mean().item(), "reussis": int(reussi.sum()), "episodes": n,
            "distance_finale_moyenne": dist.mean().item(), "secondes": round(time.time() - t0, 1),
            "reussite_par_episode": "".join("1" if r else "0" for r in reussi.tolist()), **erreurs(modele)}


def cmd_preparer(_):
    S = sortie()
    S.mkdir(parents=True, exist_ok=True)
    f32 = S / "fp32.pt"
    if not f32.exists():
        print("entraînement du world model float32", flush=True)
        torch.save(entrainer().state_dict(), f32)
    m, _ = charger(f32)
    versions = {
        "rtn2": lambda: quantifier(m, 2, "rtn"),
        "gptq2": lambda: quantifier(m, 2, "gptq"),
        "gptq2_k1": lambda: quantifier(m, 2, "gptq", FRACTION, "obs"),
        "gptq3": lambda: quantifier(m, 3, "gptq"),
    }
    for nom, faire in versions.items():
        torch.save(faire(), S / f"{nom}.pt")
    torch.save(affiner(torch.load(S / "gptq2_k1.pt", weights_only=True)), S / "gptq2_k1_affine.pt")
    refs = {}
    for nom in ["fp32", *versions, "gptq2_k1_affine"]:
        chemin = S / f"{nom}.pt"
        mod, comp = charger(chemin)
        r = mesurer(mod)
        r["bits_par_poids"] = 32.0 if comp is None else round(bits_par_poids(comp), 3)
        r["poids_gardes"] = 0 if comp is None else sum(c["garde_idx"].numel() for c in comp["couches"])
        r["empreinte"] = empreinte(chemin)
        refs[nom] = r
        print(f"{nom:16s} réussite {r['reussite']:.2f}  erreur 1 pas {r['un_pas']:.3f}  8 pas {r['8_pas']:.3f}  "
              f"{r['bits_par_poids']:.2f} bits/poids  ({r['secondes']} s)", flush=True)
    (S / "references.json").write_text(json.dumps(refs, indent=1, ensure_ascii=False))


def cmd_essai(args):
    """Évaluation comptée pour l'agent : valide le format et le budget, mesure sur les épisodes de calibration."""
    dossier = Path(args.dossier)
    journal = dossier / "evaluations.jsonl"
    deja = len(journal.read_text().splitlines()) if journal.exists() else 0
    if deja >= BUDGET_ESSAIS:
        raise SystemExit(f"budget épuisé : {deja} évaluations sur {BUDGET_ESSAIS}")
    compact = torch.load(args.fichier, map_location="cpu", weights_only=True)
    gardes = valider(compact, bits=2, budget=budget_gardes())
    r = mesurer(depuis_compact(compact))
    r.update({"n": deja + 1, "fichier": str(args.fichier), "empreinte": empreinte(args.fichier), "poids_gardes": gardes,
              "bits_par_poids": round(bits_par_poids(compact), 3), "heure": time.strftime("%Y-%m-%dT%H:%M:%S")})
    with journal.open("a") as f:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in r.items() if k != "reussite_par_episode"}, ensure_ascii=False))
    print(f"évaluations utilisées : {deja + 1} sur {BUDGET_ESSAIS}")


def cmd_erreur(args):
    """Erreurs de prédiction seulement (pas de planification) : libre, non comptée."""
    compact = torch.load(args.fichier, map_location="cpu", weights_only=True)
    valider(compact, bits=2, budget=budget_gardes())
    print(json.dumps(erreurs(depuis_compact(compact))))


def cmd_test(args):
    """Test final (pas pour l'agent) : épisodes neufs tirés avec une graine choisie après l'essai."""
    res = {}
    for chemin in args.fichiers:
        mod, comp = charger(chemin)
        if comp is not None:
            valider(comp, budget=budget_gardes() if comp["bits"] == 2 else None)
        r = mesurer(mod, args.graine, args.n, args.graine_plan)
        r["bits_par_poids"] = 32.0 if comp is None else round(bits_par_poids(comp), 3)
        r["empreinte"] = empreinte(chemin)
        res[str(chemin)] = r
        print(f"{Path(chemin).name:28s} réussite {r['reussite']:.3f} ({r['reussis']}/{r['episodes']})  "
              f"{r['bits_par_poids']:.2f} bits/poids", flush=True)
    if args.sortie:
        Path(args.sortie).write_text(json.dumps({"graine": args.graine, "graine_plan": args.graine_plan, "n": args.n, "resultats": res}, indent=1,
                                                ensure_ascii=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = p.add_subparsers(required=True)
    sp.add_parser("preparer", help="entraîne le modèle float32 et construit les références").set_defaults(f=cmd_preparer)
    e = sp.add_parser("essai", help="évaluation comptée d'un fichier compact (agent)")
    e.add_argument("dossier")
    e.add_argument("fichier")
    e.set_defaults(f=cmd_essai)
    r = sp.add_parser("erreur", help="erreurs de prédiction d'un fichier compact (libre)")
    r.add_argument("fichier")
    r.set_defaults(f=cmd_erreur)
    t = sp.add_parser("test", help="test final sur des épisodes neufs")
    t.add_argument("fichiers", nargs="+")
    t.add_argument("--graine", type=int, required=True)
    t.add_argument("--n", type=int, default=300)
    t.add_argument("--graine-plan", type=int, default=GRAINE_PLAN)
    t.add_argument("--sortie")
    t.set_defaults(f=cmd_test)
    a = p.parse_args()
    a.f(a)
