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

from quant import quantifier  # noqa: E402

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


def charger_config(chunk, device):
    with initialize_config_dir(config_dir=str(LEWM / "config" / "eval"), version_base=None):
        cfg = compose("pusht", overrides=[
            "policy=pusht/lewm", f"solver.device={device}", f"eval.num_eval={chunk}"])
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


def evaluer_paquet(world, dataset, episodes_idx, start_steps, positions, cfg):
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
    rec = {"state": np.full((T, n, 7), np.nan), "block_pose": np.full((T, n, 3), np.nan),
           "agent_pos": np.full((T, n, 2), np.nan), "contacts": np.full((T, n), -1, dtype=np.int32),
           "env_state_dist": np.full((T, n), np.nan)}
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
            rec["state"][k, vivant] = np.asarray(info["state"]).reshape(n, -1)[:, -7:][vivant]
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
               init_state=np.asarray(init_state["state"]), goal_state=np.asarray(goal_state["goal_state"]))
    return rec


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
    args = ap.parse_args()
    assert args.n % args.chunk == 0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = charger_config(args.chunk, args.device)
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
    f_meta = out / "meta.json"
    if f_meta.exists():
        ancien = json.loads(f_meta.read_text())
        if json.dumps(ancien["protocole"], sort_keys=True, default=str) != json.dumps(protocole, sort_keys=True, default=str):
            sys.exit(f"{f_meta} décrit un autre protocole : choisir un autre dossier --out")
    else:
        f_meta.write_text(json.dumps({"protocole": protocole, "git": sha, "git_sale": sale,
                                      "torch": torch.__version__}, indent=1, default=str))
    print(f"[info] commit {sha[:7]}{' (modifié)' if sale else ''}, {args.n} épisodes", flush=True)

    world = swm.World(**cfg.world, image_shape=(224, 224))
    transform = {"pixels": img_transform(cfg), "goal": img_transform(cfg)}
    solver_kw = {k: v for k, v in OmegaConf.to_container(cfg.solver, resolve=True).items()
                 if k not in ("_target_", "model", "seed")}

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
            recs = []
            for c in range(args.n // args.chunk):
                sl = slice(c * args.chunk, (c + 1) * args.chunk)
                solver = CEMParEpisode(model=model, graine=seed, **solver_kw)
                policy = swm.policy.WorldModelPolicy(solver=solver, config=swm.PlanConfig(**cfg.plan_config),
                                                     process=process, transform=transform)
                world.set_policy(policy)
                recs.append(evaluer_paquet(world, dataset, ep_all[sl], st_all[sl], list(range(args.n))[sl], cfg))
            data = {}
            for k in recs[0]:
                if k in ("init_state", "goal_state") or recs[0][k].ndim < 2:
                    data[k] = np.concatenate([np.atleast_1d(r[k]) for r in recs], axis=0)
                else:
                    data[k] = np.concatenate([r[k] for r in recs], axis=1)
            tmp = f.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, spec=spec, seed=seed, n_couches=len(touches), git=sha, git_sale=sale,
                                positions=np.arange(args.n), episodes_idx=np.array(ep_all), start_steps=np.array(st_all),
                                chunk=args.chunk, **data)
            os.replace(tmp, f)
            print(f"[fait] {spec} graine {seed} : réussite {100 * data['success'].mean():.1f} % "
                  f"({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
