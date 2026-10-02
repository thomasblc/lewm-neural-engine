"""EXP-003 : grille de quantification de LeWM sur PushT, avec plusieurs graines du planificateur.

Pour chaque version du modèle (pleine précision ou quantifiée) et chaque graine de la CEM, fait
planifier LeWM sur les mêmes N épisodes et enregistre, pour chaque épisode et chaque pas, l'état
exact du simulateur (7 nombres : agent x, y, T x, y, angle, vitesse de l'agent x, y), la pose du T,
la position de l'agent et les contacts. Un fichier .npz par (version, graine) ; un lancement
interrompu reprend là où il s'est arrêté, après avoir vérifié que la config n'a pas changé.

Hasard du planificateur (amendement 1 du pré-enregistrement) : chaque épisode a son propre tirage,
fixé par (graine, position de l'épisode dans l'échantillon, numéro du plan). Le hasard reçu par un
épisode ne dépend donc ni des autres épisodes du paquet ni de la version du modèle. La config
officielle (un générateur partagé, tiré épisode après épisode) couplait les épisodes d'un même paquet.

Le tirage des épisodes reproduit celui de le-wm/eval.py (graine 42 de la config). EXP-005 ajoute deux
options, sans effet quand elles sont absentes : --graine-episodes (autre tirage) et --exclure (meta.json
d'expériences dont les couples (épisode, pas de départ) sont écartés du tirage).

Usage :
    python run_grid.py --out ../resultats/exp003 --n 500 --chunk 50 --specs fp32 all:4 --seeds 42 43
"""
import argparse
import hashlib
import platform
import resource
import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

LEWM = Path(os.environ.get("LEWM_DIR", "/Users/thomasblanc/Documents/PRO/world-models-demos/le-wm"))
os.environ.setdefault("STABLEWM_HOME", str(LEWM.parent / "stable-wm"))
os.environ.setdefault("MUJOCO_GL", "glfw" if sys.platform == "darwin" else "egl")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import stable_pretraining as spt  # noqa: E402
import stable_worldmodel as swm  # noqa: E402
import torch  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from sklearn import preprocessing  # noqa: E402
from stable_worldmodel.solver.cem import CEMSolver  # noqa: E402
from stable_worldmodel.world.world import _apply_callables, _extract_init_goal  # noqa: E402
from torchvision.transforms import v2 as transforms  # noqa: E402

from quant import GROUPES, quantifier  # noqa: E402
import coreml_pred  # noqa: E402

# ce que chaque tâche enregistre : dimension de l'état, clés de départ et de but (configs de le-wm/config/eval)
TACHES = {"pusht": {"dim": 7, "depart": "state", "but": "goal_state", "pusht": True},
          "tworoom": {"dim": 2, "depart": "proprio", "but": "goal_proprio", "pusht": False}}
CLE = "ep_key"  # position de l'épisode dans l'échantillon, glissée dans les infos du monde


class CEMParEpisode(CEMSolver):
    """CEM de stable-worldmodel, mais avec un tirage propre à chaque épisode.

    Chaque appel est découpé épisode par épisode (comme batch_size = 1). Avant chaque épisode, le
    générateur est réinitialisé à partir de (graine, position de l'épisode, numéro du plan).
    """

    def __init__(self, *args, graine=0, **kwargs):
        super().__init__(*args, seed=graine, **kwargs)
        self.graine = int(graine)
        self._compte = {}

    def solve(self, info_dict, init_action=None):
        cles = info_dict.pop(CLE, None)
        if cles is None:
            raise RuntimeError("clé d'épisode absente : le hasard ne serait plus apparié")
        cles = np.asarray(cles.cpu() if torch.is_tensor(cles) else cles).reshape(len(cles), -1)[:, -1]
        sorties = []
        for i, k in enumerate(cles.tolist()):
            k = int(k)
            c = self._compte.get(k, 0)
            self._compte[k] = c + 1
            s = int(np.random.SeedSequence([self.graine, k, c]).generate_state(1)[0])
            self.torch_gen.manual_seed(s)
            sub = {}
            for kk, v in info_dict.items():
                if torch.is_tensor(v) or isinstance(v, np.ndarray):
                    sub[kk] = v[i:i + 1]
                elif isinstance(v, list):
                    sub[kk] = v[i:i + 1]
                else:
                    sub[kk] = v
            init = init_action[i:i + 1] if init_action is not None else None
            sorties.append(super().solve(sub, init))
        return {
            "actions": torch.cat([o["actions"] for o in sorties]),
            "costs": sum((list(o["costs"]) for o in sorties), []),
            "mean": [torch.cat([o["mean"][0] for o in sorties])],
            "var": [torch.cat([o["var"][0] for o in sorties])],
        }


def img_transform(cfg):
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=cfg.eval.img_size),
    ])


def charger_config(chunk, device, tache="pusht"):
    with initialize_config_dir(config_dir=str(LEWM / "config" / "eval"), version_base=None):
        cfg = compose(tache, overrides=[
            f"policy={tache}/lewm", f"solver.device={device}", f"eval.num_eval={chunk}"])
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    return cfg


def preparer_donnees(cfg):
    """Jeu de données, normalisations et épisodes valides, exactement comme le-wm/eval.py."""
    dataset = swm.data.HDF5Dataset(cfg.eval.dataset_name, keys_to_cache=cfg.dataset.keys_to_cache,
                                   cache_dir=swm.data.utils.get_cache_dir())
    col = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(dataset.get_col_data(col), return_index=True)
    process = {}
    for c in cfg.dataset.keys_to_cache:
        if c == "pixels":
            continue
        sc = preprocessing.StandardScaler()
        d = dataset.get_col_data(c)
        sc.fit(d[~np.isnan(d).any(axis=1)])
        process[c] = sc
        if c != "action":
            process[f"goal_{c}"] = process[c]
    step_idx = dataset.get_col_data("step_idx")
    ep_col = dataset.get_col_data(col)
    lengths = np.array([np.max(step_idx[ep_col == e]) + 1 for e in ep_indices])
    max_start = dict(zip(ep_indices, lengths - cfg.eval.goal_offset_steps - 1))
    valid = np.nonzero(step_idx <= np.array([max_start[e] for e in ep_col]))[0]
    return dataset, process, col, valid


def tirer_episodes(cfg, dataset, col, valid, n, graine=None, exclus=frozenset()):
    if graine is None and not exclus:                     # tirage d'origine, identique à c5703db
        g = np.random.default_rng(cfg.seed)
        idx = np.sort(valid[g.choice(len(valid) - 1, size=n, replace=False)])
        rows = dataset.get_row_data(idx)
        return np.asarray(rows[col]).tolist(), np.asarray(rows["step_idx"]).tolist()
    # EXP-005 : même forme de tirage, autre graine. On écarte les trajectoires entières déjà utilisées (tout
    # pas de départ d'un épisode du jeu présent dans la liste d'exclusion) et on garde au plus une position par
    # trajectoire, dans l'ordre du tirage : les positions sont des épisodes indépendants. Le nombre de candidats
    # dépend de n et de la liste : le tirage est fixé par (graine, n, liste d'exclusion).
    g = np.random.default_rng(cfg.seed if graine is None else graine)
    cand = valid[g.choice(len(valid) - 1, size=min(len(valid) - 1, 2 * n + len(exclus) + 100), replace=False)]
    ordre = np.argsort(cand)                              # le lecteur HDF5 veut des indices croissants
    rows = dataset.get_row_data(cand[ordre])
    lues = list(zip(np.asarray(rows[col]).tolist(), np.asarray(rows["step_idx"]).tolist()))
    paires = [None] * len(cand)
    for j, i in enumerate(ordre):
        paires[i] = lues[j]
    ep_exclus, vus, garde = {a for a, _ in exclus}, set(), []
    for i, (ep, _) in enumerate(paires):
        if ep in ep_exclus or ep in vus:
            continue
        vus.add(ep)
        garde.append(i)
        if len(garde) == n:
            break
    if len(garde) < n:
        sys.exit(f"tirage : {len(garde)} épisodes disponibles au lieu de {n}")
    idx = np.sort(cand[garde])
    rows = dataset.get_row_data(idx)
    return np.asarray(rows[col]).tolist(), np.asarray(rows["step_idx"]).tolist()


def lire_exclus(chemins):
    """Couples (épisode, pas de départ) des meta.json donnés (format de run_grid ou des premiers tests)."""
    exclus = set()
    for c in chemins:
        m = json.loads(Path(c).read_text())
        m = m.get("protocole", m)
        exclus |= set(zip(m["episodes_idx"], m["start_steps"]))
    return frozenset(exclus)


def evaluer_paquet(world, dataset, episodes_idx, start_steps, positions, cfg, tache="pusht"):
    """Copie de World._evaluate_from_dataset (stable-worldmodel 0.1.1) qui enregistre chaque pas."""
    n = len(episodes_idx)
    assert n == world.num_envs
    init_state, goal_state, _ = _extract_init_goal(dataset, episodes_idx, start_steps, cfg.eval.goal_offset_steps)
    # Le jeu PushT n'a pas de colonne « seed » : sans graine, la réinitialisation tire une position au hasard
    # avant _set_state, et pymunk garde la trace des contacts de cette position. Graine fixe par épisode.
    graines_env = init_state.get("seed")
    if graines_env is None:
        graines_env = [int(np.random.SeedSequence([7, int(p)]).generate_state(1)[0]) for p in positions]
    world.reset(seed=list(graines_env))
    callables = OmegaConf.to_container(cfg.eval.callables, resolve=True)
    merged = {**init_state, **goal_state}
    for i in range(n):
        _apply_callables(world.envs.envs[i].unwrapped, deepcopy(callables), {k: v[i] for k, v in merged.items()})
    prefix = world.infos["pixels"].shape[:2]
    for src in (init_state, goal_state):
        for k, v in src.items():
            if k in world.infos or k in goal_state:
                world.infos[k] = np.broadcast_to(v[:, None, ...], prefix + v.shape[1:]).copy()
    snapshot = {k: world.infos[k].copy() for k in goal_state}
    snapshot[CLE] = np.asarray(positions, dtype=np.int64)[:, None]
    world.infos.update(deepcopy(snapshot))

    T = cfg.eval.eval_budget
    tc = TACHES[tache]
    rec = {"state": np.full((T, n, tc["dim"]), np.nan), "env_state_dist": np.full((T, n), np.nan)}
    if tc["pusht"]:
        rec.update(block_pose=np.full((T, n, 3), np.nan), agent_pos=np.full((T, n, 2), np.nan),
                   contacts=np.full((T, n), -1, dtype=np.int32))
    success = np.zeros(n, dtype=bool)
    first = np.full(n, -1, dtype=np.int32)
    t = [0]

    def on_step(w):
        vivant = ~success  # épisodes pas encore réussis avant ce pas : les autres sont gelés
        k = t[0]
        info = w.infos
        # norme de l'écart d'état 7-D renvoyée par l'env (mélange pixels, radians, vitesses) : ne pas rapporter
        rec["env_state_dist"][k, vivant] = -np.asarray(w.rewards, dtype=np.float64)[vivant]
        if "state" in info:
            rec["state"][k, vivant] = np.asarray(info["state"]).reshape(n, -1)[:, -tc["dim"]:][vivant]
        if tc["pusht"]:
            rec["block_pose"][k, vivant] = np.asarray(info["block_pose"]).reshape(n, -1)[:, -3:][vivant]
            rec["agent_pos"][k, vivant] = np.asarray(info["pos_agent"]).reshape(n, -1)[:, -2:][vivant]
            rec["contacts"][k, vivant] = np.asarray(info["n_contacts"]).reshape(n, -1)[:, -1][vivant]
        term = np.asarray(w.terminateds, dtype=bool)
        first[term & vivant] = k
        success[:] |= term
        w.infos.update(deepcopy(snapshot))
        t[0] += 1

    world._run(max_steps=T, mode="wait", on_step=on_step)
    rec.update(success=success, first_success=first, n_steps=np.int32(t[0]),
               init_state=np.asarray(init_state[tc["depart"]]), goal_state=np.asarray(goal_state[tc["but"]]))
    return rec


def calibrer_gptq(args, cfg, dataset, process, ep_all, st_all, world, transform, solver_kw, out, sha):
    """EXP-009 : planifie en pleine précision sur des épisodes de calibration et accumule, pour chaque couche que
    quant.py arrondit, H = somme des x x^T de ses entrées (Conv2d : carrés dépliés). Sommes sur l'appareil en
    float32, versées en float64 sur CPU après chaque paquet. Écrit out/calib.pt."""
    assert args.specs == ["fp32"] and len(args.seeds) == 1, "calibration : --specs fp32 et une seule graine"
    model = swm.wm.utils.load_pretrained(cfg.policy)
    cibles = {nom: mod for nom, mod in model.named_modules()
              if nom.split(".")[0] in GROUPES["all"] and isinstance(mod, (torch.nn.Linear, torch.nn.Conv2d))}
    model = model.to(args.device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    acc, npaquet, H, n = {}, {}, {}, {}
    moities = {"a": ({}, {}), "b": ({}, {})}             # paquets pairs et impairs : contrôle hors échantillon

    def crochet(nom):
        def f(mod, entree):
            x = entree[0].detach()
            if isinstance(mod, torch.nn.Conv2d):
                x = torch.nn.functional.unfold(x, mod.kernel_size, dilation=mod.dilation, padding=mod.padding,
                                               stride=mod.stride).transpose(1, 2)
            x = x.reshape(-1, x.shape[-1]).float()
            acc[nom] = acc[nom] + x.T @ x if nom in acc else x.T @ x
            n[nom] = n.get(nom, 0) + x.shape[0]
            npaquet[nom] = npaquet.get(nom, 0) + x.shape[0]
        return f

    poignees = [m.register_forward_pre_hook(crochet(nom)) for nom, m in cibles.items()]
    for c in range(args.n // args.chunk):
        sl = slice(c * args.chunk, (c + 1) * args.chunk)
        solver = CEMParEpisode(model=model, graine=args.seeds[0], **solver_kw)
        world.set_policy(swm.policy.WorldModelPolicy(solver=solver, config=swm.PlanConfig(**cfg.plan_config),
                                                     process=process, transform=transform))
        evaluer_paquet(world, dataset, ep_all[sl], st_all[sl], list(range(args.n))[sl], cfg, args.tache)
        Hm, nm = moities["a" if c % 2 == 0 else "b"]
        for nom in list(acc):
            h = acc.pop(nom).cpu().double()               # MPS ne connaît pas le float64 : CPU d'abord
            H[nom] = H.get(nom, 0) + h
            Hm[nom] = Hm.get(nom, 0) + h
            nm[nom] = nm.get(nom, 0) + npaquet.pop(nom)
        print(f"[calibration] paquet {c + 1} sur {args.n // args.chunk}", flush=True)
    for p in poignees:
        p.remove()
    manque = sorted(set(cibles) - set(H))
    assert not manque, f"calibration : couches jamais appelées {manque}"
    tmp = out / "calib.tmp.pt"
    torch.save({"H": H, "n": n, "H_a": moities["a"][0], "n_a": moities["a"][1], "H_b": moities["b"][0],
                "n_b": moities["b"][1], "episodes_idx": list(ep_all), "start_steps": list(st_all), "graine": args.seeds[0],
                "git": sha, "tache": args.tache}, tmp)
    os.replace(tmp, out / "calib.pt")
    print(f"[calibration] {len(H)} couches, de {min(n.values())} à {max(n.values())} lignes", flush=True)


def git_info():
    ici = Path(__file__).resolve().parent
    sha = subprocess.run(["git", "-C", str(ici), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    sale = subprocess.run(["git", "-C", str(ici), "status", "--porcelain", "--untracked-files=no"],
                          capture_output=True, text=True).stdout.strip() != ""
    return sha, sale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--specs", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--graine-episodes", type=int, default=None)
    ap.add_argument("--exclure", nargs="*", default=[])
    ap.add_argument("--tache", choices=sorted(TACHES), default="pusht")
    ap.add_argument("--replan", type=int, default=None, help="blocs de 5 actions exécutés avant de replanifier (défaut 5)")
    ap.add_argument("--cem-iter", type=int, default=None, help="itérations de la CEM par plan (défaut 30)")
    ap.add_argument("--calibrer-gptq", action="store_true", help="EXP-009 : calibre GPTQ au lieu d'évaluer")
    ap.add_argument("--gptq-calib", default=None, help="EXP-009 : statistiques de calibration pour les versions :gptq")
    ap.add_argument("--moteur", choices=coreml_pred.MOTEURS, default="torch",
                    help="EXP-012 : prédicteur en PyTorch (défaut) ou en Core ML 16 bits sur le GPU ou le Neural Engine ; "
                         "EXP-013 : torch-fp16 (autocast)")
    ap.add_argument("--codage", choices=coreml_pred.CODAGES + coreml_pred.CODAGES_PORTEE, default="fp16",
                    help="EXP-013 : codage des poids Core ML ; EXP-015 : portée -sp (sans pred_proj), -pp (pred_proj seule), -bl (blocs seuls)")
    ap.add_argument("--go", default=None, help="EXP-013 : après la mise en route, écrire GO.pret puis attendre le fichier GO "
                    "(la charge de vision démarre entre les deux, une fois le modèle chargé et converti)")
    ap.add_argument("--energie-puce", action="store_true", help="EXP-013 : énergie par composant (IOReport) par fichier")
    args = ap.parse_args()
    assert args.n % args.chunk == 0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = charger_config(args.chunk, args.device, args.tache)
    if args.replan is not None:                              # Q4 : replanifier plus souvent (EXP-007)
        cfg.plan_config.receding_horizon = args.replan
    if args.cem_iter is not None:
        cfg.solver.n_steps = args.cem_iter
    dataset, process, col, valid = preparer_donnees(cfg)
    exclus = lire_exclus(args.exclure)
    ep_all, st_all = tirer_episodes(cfg, dataset, col, valid, args.n, args.graine_episodes, exclus)
    sha, sale = git_info()
    protocole = {"n": args.n, "chunk": args.chunk, "episodes_idx": ep_all, "start_steps": st_all,
                 "episode_seed": int(cfg.seed), "device": args.device, "hasard": "par episode (amendement 1)",
                 "config": OmegaConf.to_container(cfg, resolve=False)}
    if args.graine_episodes is not None or exclus:           # absent du protocole d'EXP-003 et d'EXP-004
        protocole["graine_episodes"] = args.graine_episodes
        protocole["exclus"] = sorted(map(list, exclus))
        protocole["regle_tirage"] = "trajectoires exclues entières, une position par trajectoire"
    if args.tache != "pusht":                                # absent du protocole des expériences sur PushT
        protocole["tache"] = args.tache
    if args.gptq_calib:                                       # absent des protocoles précédents
        os.environ["LEWM_GPTQ_CALIB"] = str(Path(args.gptq_calib).resolve())
        protocole["gptq_calib"] = {"fichier": Path(args.gptq_calib).name,
                                   "sha256": hashlib.sha256(Path(args.gptq_calib).read_bytes()).hexdigest()}
    if args.calibrer_gptq:
        protocole["calibration_gptq"] = True
    if args.moteur in ("ne-complet", "coreml-cpu") and args.device != "cpu":
        sys.exit(f"{args.moteur} : la CEM tourne sur le CPU, lancer avec --device cpu")
    if args.moteur == "torch-fp16":                           # EXP-013, absent des protocoles précédents
        protocole["moteur"] = {"nom": args.moteur, "macos": platform.mac_ver()[0], "precision": "float16 (autocast)"}
    elif args.moteur != "torch":                              # absent des protocoles précédents
        import coremltools
        protocole["moteur"] = {"nom": args.moteur, "coremltools": coremltools.__version__,
                               "macos": platform.mac_ver()[0], "precision": "float16"}
        if args.codage != "fp16":                             # EXP-013
            protocole["moteur"]["codage"] = args.codage
    compteur = None
    if args.energie_puce:                                     # EXP-013 : énergie par composant, sans changer le protocole
        import energie_puce
        compteur = energie_puce.Compteur()
    if args.replan is not None or args.cem_iter is not None:  # absent des protocoles précédents
        protocole["planification"] = {"receding_horizon": int(cfg.plan_config.receding_horizon),
                                      "cem_n_steps": int(cfg.solver.n_steps)}
    f_meta = out / "meta.json"
    if f_meta.exists():
        ancien = json.loads(f_meta.read_text())
        if json.dumps(ancien["protocole"], sort_keys=True, default=str) != json.dumps(protocole, sort_keys=True, default=str):
            sys.exit(f"{f_meta} décrit un autre protocole : choisir un autre dossier --out")
    else:
        tmp_meta = f_meta.with_name("meta.json.tmp")         # jamais de meta.json partiel
        tmp_meta.write_text(json.dumps({"protocole": protocole, "git": sha, "git_sale": sale,
                                        "torch": torch.__version__}, indent=1, default=str))
        os.replace(tmp_meta, f_meta)
    print(f"[info] commit {sha[:7]}{' (modifié)' if sale else ''}, {args.n} épisodes", flush=True)

    world = swm.World(**cfg.world, image_shape=(224, 224))
    transform = {"pixels": img_transform(cfg), "goal": img_transform(cfg)}
    solver_kw = {k: v for k, v in OmegaConf.to_container(cfg.solver, resolve=True).items()
                 if k not in ("_target_", "model", "seed")}

    if args.calibrer_gptq:
        calibrer_gptq(args, cfg, dataset, process, ep_all, st_all, world, transform, solver_kw, out, sha)
        return
    for seed in args.seeds:
        for spec in args.specs:
            f = out / f"{spec.replace(':', '-')}_s{seed}.npz"
            if f.exists():
                print(f"[saut] {f.name} existe", flush=True)
                continue
            t0 = time.time()
            model = swm.wm.utils.load_pretrained(cfg.policy)
            model, touches = quantifier(model, "" if spec == "fp32" else spec)
            model = model.to(args.device).eval()
            model.requires_grad_(False)
            model.interpolate_pos_encoding = True
            if args.moteur == "torch-fp16":                   # EXP-013 : poids en float16 une fois pour toutes
                model = model.half()
            # EXP-013 : paquets Core ML dans un dossier .noindex (Spotlight ne les indexe pas pendant la mesure suivante)
            paquets = out / "modeles.noindex" if args.energie_puce else out
            paquets.mkdir(exist_ok=True)
            octets_modules = {n: sum(q.numel() * q.element_size() for q in c.parameters()) for n, c in model.named_children()}
            moteur = coreml_pred.brancher(model, args.moteur, paquets / f"{f.stem}.mlpackage", args.codage)   # {} pour torch
            recs, temps, energie_solve, fenetres = [], [], {}, []
            if args.energie_puce:                          # EXP-013 : mise en route hors mesure, compteurs à zéro
                coreml_pred.echauffer(model, args.moteur)
                coreml_pred.remise_a_zero_appels()
            if args.go:                                    # EXP-013 : la vision démarre maintenant, le run attend son signal
                Path(args.go + ".pret").write_text(str(time.time()))
                t_go = time.time()
                while not Path(args.go).exists():
                    if time.time() - t_go > 900:
                        sys.exit(f"{args.go} : signal de départ jamais reçu")
                    time.sleep(0.2)
            octets_torch = sum(q.numel() * q.element_size() for q in model.parameters()) if not moteur else None
            ru0 = resource.getrusage(resource.RUSAGE_SELF)
            e0 = compteur.lire() if compteur else None
            conditions_debut = coreml_pred.Puissance.lire() if compteur else None
            with coreml_pred.Puissance() as puissance:
                for c in range(args.n // args.chunk):
                    sl = slice(c * args.chunk, (c + 1) * args.chunk)
                    solver = CEMParEpisode(model=model, graine=seed, **solver_kw)
                    if args.moteur == "torch-fp16":
                        solver = coreml_pred.autocast_fp16(solver)
                    solver = coreml_pred.chronometrer(solver, temps, compteur, energie_solve, fenetres, args.device == "mps")
                    policy = swm.policy.WorldModelPolicy(solver=solver, config=swm.PlanConfig(**cfg.plan_config),
                                                         process=process, transform=transform)
                    world.set_policy(policy)
                    recs.append(evaluer_paquet(world, dataset, ep_all[sl], st_all[sl], list(range(args.n))[sl], cfg, args.tache))
            energie_fichier = None
            if compteur:
                j, dt = compteur.ecart(e0, compteur.lire())
                ru1 = resource.getrusage(resource.RUSAGE_SELF)
                energie_fichier = {"joules": j, "duree_s": dt, "debut": e0[0], "joules_solve": energie_solve,
                                   "cpu_s": (ru1.ru_utime + ru1.ru_stime) - (ru0.ru_utime + ru0.ru_stime),
                                   "maxrss_octets": ru1.ru_maxrss, "appels_predict": dict(coreml_pred.APPELS),
                                   "octets_modele_torch": octets_torch, "octets_modules_torch": octets_modules,
                                   "fenetres_solve": fenetres, "fils_torch": torch.get_num_threads(),
                                   "omp": {k: v for k, v in os.environ.items() if k.startswith("OMP_")},
                                   "priorite": os.getpriority(os.PRIO_PROCESS, 0),
                                   "conditions_debut": conditions_debut, "conditions_fin": coreml_pred.Puissance.lire()}
            data = {}
            for k in recs[0]:
                if k in ("init_state", "goal_state") or recs[0][k].ndim < 2:
                    data[k] = np.concatenate([np.atleast_1d(r[k]) for r in recs], axis=0)
                else:
                    data[k] = np.concatenate([r[k] for r in recs], axis=1)
            tmp = f.with_suffix(".tmp.npz")
            extra = {"moteur": args.moteur, "coreml_sha256": moteur["sha256_poids"]} if moteur else {}
            np.savez_compressed(tmp, spec=spec, seed=seed, n_couches=len(touches), git=sha, git_sale=sale,
                                positions=np.arange(args.n), episodes_idx=np.array(ep_all), start_steps=np.array(st_all),
                                chunk=args.chunk, **extra, **data)
            # temps et puissance à part : les fichiers .npz du moteur torch restent identiques à ceux d'EXP-003 ;
            # écrits avant le .npz, si bien qu'un .npz présent a toujours son fichier de temps
            ft = f.with_suffix(".temps.json")
            ft_tmp = f.with_suffix(".temps.tmp.json")
            ft_tmp.write_text(json.dumps({
                "moteur": args.moteur, "coreml": moteur, "appels_solve": temps,
                "temps_par_plan_s": [d / n for d, n in temps if n > 0], "plans": sum(n for _, n in temps),
                "puissance": puissance.resume(), "lectures": puissance.lectures, "duree_s": time.time() - t0,
                "energie_puce": energie_fichier,
                "git": sha}, indent=1))
            os.replace(ft_tmp, ft)
            os.replace(tmp, f)
            print(f"[fait] {spec} graine {seed} : réussite {100 * data['success'].mean():.1f} % "
                  f"({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
