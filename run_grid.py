"""EXP-003 : grille de quantification de LeWM sur PushT, avec plusieurs graines du planificateur.

Pour chaque version du modèle (pleine précision ou quantifiée) et chaque graine de la CEM, fait
planifier LeWM sur les mêmes N épisodes et enregistre, pour chaque épisode et chaque pas, la distance
exacte au but donnée par le simulateur, la pose du T, la position de l'agent et les contacts.
Un fichier .npz par (version, graine) ; un lancement interrompu reprend là où il s'est arrêté.

Le tirage des épisodes reproduit celui de le-wm/eval.py (graine 42 de la config) : avec N = 50 et des
paquets de 50, la pleine précision à la graine 42 doit redonner exactement EXP-002 (test de sanité).

Usage :
    python run_grid.py --out ../resultats/exp003 --n 200 --chunk 50 \
        --specs fp32 all:4 encoder:4 --seeds 42 43
"""
import argparse
import json
import os
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
from stable_worldmodel.world.world import _apply_callables, _extract_init_goal  # noqa: E402
from torchvision.transforms import v2 as transforms  # noqa: E402

from quant import quantifier  # noqa: E402


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
    """Jeu de données, normalisations et tirage des épisodes, exactement comme le-wm/eval.py."""
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


def tirer_episodes(cfg, dataset, col, valid, n):
    g = np.random.default_rng(cfg.seed)
    idx = np.sort(valid[g.choice(len(valid) - 1, size=n, replace=False)])
    rows = dataset.get_row_data(idx)
    return np.asarray(rows[col]).tolist(), np.asarray(rows["step_idx"]).tolist()


def evaluer_paquet(world, dataset, episodes_idx, start_steps, cfg):
    """Copie de World._evaluate_from_dataset (stable-worldmodel 0.1.1) qui enregistre chaque pas."""
    n = len(episodes_idx)
    assert n == world.num_envs
    init_state, goal_state, _ = _extract_init_goal(dataset, episodes_idx, start_steps, cfg.eval.goal_offset_steps)
    world.reset(seed=init_state.get("seed"))
    callables = OmegaConf.to_container(cfg.eval.callables, resolve=True)
    merged = {**init_state, **goal_state}
    for i in range(n):
        _apply_callables(world.envs.envs[i].unwrapped, deepcopy(callables), {k: v[i] for k, v in merged.items()})
    prefix = world.infos["pixels"].shape[:2]
    for src in (init_state, goal_state):
        for k, v in src.items():
            if k in world.infos or k in goal_state:
                world.infos[k] = np.broadcast_to(v[:, None, ...], prefix + v.shape[1:]).copy()
    goal_snapshot = {k: world.infos[k].copy() for k in goal_state}

    T = cfg.eval.eval_budget
    rec = {"dist": np.full((T, n), np.nan), "block_pose": np.full((T, n, 3), np.nan),
           "agent_pos": np.full((T, n, 2), np.nan), "contacts": np.full((T, n), -1, dtype=np.int32)}
    success = np.zeros(n, dtype=bool)
    first = np.full(n, -1, dtype=np.int32)
    t = [0]

    def on_step(w):
        vivant = ~success  # épisodes pas encore réussis avant ce pas
        k = t[0]
        rec["dist"][k, vivant] = -np.asarray(w.rewards, dtype=np.float64)[vivant]
        info = w.infos
        bp = np.asarray(info["block_pose"]).reshape(n, -1)[:, -3:]
        ap = np.asarray(info["pos_agent"]).reshape(n, -1)[:, -2:]
        nc = np.asarray(info["n_contacts"]).reshape(n, -1)[:, -1]
        rec["block_pose"][k, vivant] = bp[vivant]
        rec["agent_pos"][k, vivant] = ap[vivant]
        rec["contacts"][k, vivant] = nc[vivant]
        term = np.asarray(w.terminateds, dtype=bool)
        nouveaux = term & vivant
        first[nouveaux] = k
        success[:] |= term
        w.infos.update(deepcopy(goal_snapshot))
        t[0] += 1

    world._run(max_steps=T, mode="wait", on_step=on_step)
    rec.update(success=success, first_success=first, n_steps=np.int32(t[0]),
               init_state=np.asarray(init_state["state"]), goal_state=np.asarray(goal_state["goal_state"]))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--specs", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    assert args.n % args.chunk == 0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = charger_config(args.chunk, args.device)
    dataset, process, col, valid = preparer_donnees(cfg)
    ep_all, st_all = tirer_episodes(cfg, dataset, col, valid, args.n)
    meta = {"n": args.n, "chunk": args.chunk, "episodes_idx": ep_all, "start_steps": st_all,
            "episode_seed": int(cfg.seed), "torch": torch.__version__, "device": args.device,
            "config": OmegaConf.to_container(cfg, resolve=False)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    world = swm.World(**cfg.world, image_shape=(224, 224))
    transform = {"pixels": img_transform(cfg), "goal": img_transform(cfg)}

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
                solver = hydra.utils.instantiate(cfg.solver, model=model, seed=seed)
                policy = swm.policy.WorldModelPolicy(solver=solver, config=swm.PlanConfig(**cfg.plan_config),
                                                     process=process, transform=transform)
                world.set_policy(policy)
                recs.append(evaluer_paquet(world, dataset, ep_all[sl], st_all[sl], cfg))
            data = {k: (np.concatenate([r[k] for r in recs], axis=1) if recs[0][k].ndim >= 2 and k not in ("init_state", "goal_state")
                        else np.concatenate([np.atleast_1d(r[k]) for r in recs], axis=0))
                    for k in recs[0]}
            np.savez_compressed(f, spec=spec, seed=seed, n_couches=len(touches), **data)
            print(f"[fait] {spec} graine {seed} : réussite {100 * data['success'].mean():.1f} % "
                  f"({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
