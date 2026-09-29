"""Tests de run_grid.py (EXP-005) sur CPU : sans les nouvelles options, le calcul est celui de c5703db ; avec
elles, le tirage des épisodes est reproductible et évite les épisodes déjà utilisés.

Usage :
    python test_run_grid.py        # environ 2 min sur CPU
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ICI = Path(__file__).resolve().parent
R = ICI.parent / "resultats"
PY = sys.executable
CLES = ("state", "block_pose", "agent_pos", "contacts", "success", "first_success", "n_steps", "init_state", "goal_state")


def test_chemin_par_defaut(tmp):
    """10 épisodes fp32 graine 42 sur CPU, sans option nouvelle : identique au lancement de test de c5703db."""
    out = Path(tmp) / "defaut"
    subprocess.run([PY, "-B", "run_grid.py", "--out", str(out), "--n", "10", "--chunk", "10", "--specs", "fp32",
                    "--seeds", "42", "--device", "cpu"], cwd=ICI, check=True, capture_output=True)
    a, b = np.load(out / "fp32_s42.npz"), np.load(R / "test-c5703db-c10" / "fp32_s42.npz")
    ko = [k for k in CLES if not np.array_equal(a[k], b[k], equal_nan=a[k].dtype.kind == "f")]
    assert not ko, f"le chemin par défaut a changé : {ko}"
    pa = json.loads((out / "meta.json").read_text())["protocole"]
    pb = json.loads((R / "test-c5703db-c10" / "meta.json").read_text())["protocole"]
    assert json.dumps(pa, sort_keys=True, default=str) == json.dumps(pb, sort_keys=True, default=str)
    print("ok chemin par défaut : fichier et protocole identiques au lancement de c5703db (CPU, 10 épisodes)")


def test_tirage():
    import run_grid as RG
    cfg = RG.charger_config(50, "cpu")
    dataset, _, col, valid = RG.preparer_donnees(cfg)
    m3 = json.loads((R / "exp003" / "meta.json").read_text())["protocole"]
    assert RG.tirer_episodes(cfg, dataset, col, valid, 500) == (m3["episodes_idx"], m3["start_steps"])
    exclus = RG.lire_exclus([R / "exp003" / "meta.json", R / "sanite-n50" / "meta.json"])
    assert len(exclus) == 550
    e, s = RG.tirer_episodes(cfg, dataset, col, valid, 1000, 2026, exclus)
    assert (e, s) == RG.tirer_episodes(cfg, dataset, col, valid, 1000, 2026, exclus)
    paires = set(zip(e, s))
    assert len(e) == len(paires) == 1000 and not (paires & exclus)
    print("ok tirage : défaut identique à EXP-003 ; 1 000 épisodes reproductibles, aucun en commun avec "
          "EXP-003 ni EXP-002")


if __name__ == "__main__":
    test_tirage()
    with tempfile.TemporaryDirectory() as tmp:
        test_chemin_par_defaut(tmp)
    print("tous les tests passent")
