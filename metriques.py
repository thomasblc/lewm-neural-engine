"""Chargement des résultats d'EXP-003 et calcul des métriques M1, M2, M3 (voir PREREG.md).

Disposition de l'état PushT (stable_worldmodel, envs/pusht/env.py, _get_obs) :
    [agent x, agent y, T x, T y, angle du T (mod 2 pi), agent vx, agent vy]
`block_pose` enregistré par pas : [T x, T y, angle brut du T] (angle non réduit modulo 2 pi).
Succès de l'environnement : norme de (agent x, agent y, T x, T y) moins le but < 20, et écart d'angle
< pi / 9, à un pas quelconque de l'épisode.
"""
from pathlib import Path

import numpy as np


def ecart_angle(a, b):
    d = np.mod(a - b, 2 * np.pi)
    return np.minimum(d, 2 * np.pi - d)


def charger(dossier):
    """Renvoie {spec: {seed: dict de tableaux}}. Ignore les fichiers temporaires d'écriture."""
    res = {}
    for f in sorted(Path(dossier).glob("*.npz")):
        if ".tmp." in f.name:
            continue
        d = dict(np.load(f))
        spec, seed = str(d["spec"]), int(d["seed"])
        res.setdefault(spec, {})[seed] = d
    return res


def metriques_episode(d):
    """M2 et M3 par épisode pour une (version, graine) : minimum sur les pas enregistrés."""
    goal = d["goal_state"]                     # (n, 7)
    bp = d["block_pose"]                       # (T, n, 3)
    pos = np.linalg.norm(bp[..., :2] - goal[None, :, 2:4], axis=-1)   # (T, n)
    ang = ecart_angle(bp[..., 2], goal[None, :, 4])
    m2 = np.nanmin(pos, axis=0)
    m3 = np.nanmin(ang, axis=0)
    return m2, m3


def tableau(res, spec):
    """success (graines, n), M2 (graines, n), M3 (graines, n) pour une version."""
    graines = sorted(res[spec])
    s = np.stack([res[spec][g]["success"] for g in graines]).astype(float)
    m = [metriques_episode(res[spec][g]) for g in graines]
    return graines, s, np.stack([x[0] for x in m]), np.stack([x[1] for x in m])


def verifier_succes(d, tol=1e-6):
    """Test de sanité : au pas du premier succès, les seuils de l'environnement doivent être tenus.

    Renvoie le nombre d'épisodes réussis qui violent les seuils (attendu : 0).
    """
    goal = d["goal_state"]
    viol = 0
    for i in np.where(d["success"])[0]:
        k = int(d["first_success"][i])
        agent = d["agent_pos"][k, i]
        bp = d["block_pose"][k, i]
        pos4 = np.linalg.norm(np.concatenate([agent, bp[:2]]) - goal[i, :4])
        ang = ecart_angle(bp[2], goal[i, 4])
        if not (pos4 < 20 + tol and ang < np.pi / 9 + tol):
            viol += 1
    return viol


def prolonger(x):
    """Prolonge le dernier pas enregistré sur les pas gelés (NaN) : l'env est figé après un succès."""
    x = x.copy()
    for t in range(1, x.shape[0]):
        manque = np.isnan(x[t])
        x[t][manque] = x[t - 1][manque]
    return x


def marges(d):
    """M2, M3, M4 de l'amendement 1 pour une (version, graine) : minimum sur les 50 pas.

    Renvoie (m, m_T, m_A), chacun de forme (n,). m < 1 équivaut au succès de l'environnement.
    """
    goal = d["goal_state"]
    s = prolonger(d["state"])                                   # (T, n, 7)
    d4 = np.linalg.norm(s[..., :4] - goal[None, :, :4], axis=-1)
    dT = np.linalg.norm(s[..., 2:4] - goal[None, :, 2:4], axis=-1)
    dA = np.linalg.norm(s[..., :2] - goal[None, :, :2], axis=-1)
    dth = ecart_angle(s[..., 4], goal[None, :, 4])
    lim = np.pi / 9
    m = np.min(np.maximum(d4 / 20, dth / lim), axis=0)
    mT = np.min(np.maximum(dT / 20, dth / lim), axis=0)
    mA = np.min(dA / 20, axis=0)
    return m, mT, mA


def tableau_marges(res, spec, graines):
    """success, M2, M3, M4 de forme (graines, n) ; vérifie que toutes les graines demandées existent."""
    manque = [g for g in graines if g not in res.get(spec, {})]
    assert not manque, f"{spec} : graines manquantes {manque}"
    s = np.stack([res[spec][g]["success"] for g in graines]).astype(float)
    m = [marges(res[spec][g]) for g in graines]
    return s, np.stack([x[0] for x in m]), np.stack([x[1] for x in m]), np.stack([x[2] for x in m])
