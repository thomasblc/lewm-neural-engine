"""Tests de coreml_pred.py : le prédicteur branché sur Core ML (GPU et Neural Engine, 16 bits) donne la même sortie que
model.predict de PyTorch en float32, à la précision du 16 bits près, pour T = 1, 2, 3 (les séquences courtes sont
complétées puis recoupées), sur des plongements réels de PushT ; les poids du modèle ne changent pas ; le moteur torch
ne touche à rien. Lance de courts calculs sur le GPU et le Neural Engine (quelques secondes).

Usage :
    python test_coreml_pred.py
"""
import os
import tempfile
from pathlib import Path

os.environ.setdefault("STABLEWM_HOME", "/Users/thomasblanc/Documents/PRO/world-models-demos/stable-wm")
import torch
import stable_worldmodel as swm

import coreml_pred as C


def entrees(model, T, g):
    with torch.no_grad():
        e = model.encode({"pixels": torch.rand(C.LOT, 1, 3, 224, 224, generator=g)})["emb"]   # plongements d'images
        emb = e.expand(-1, T, -1).contiguous() + 0.1 * torch.randn(C.LOT, T, e.shape[-1], generator=g)
        act = model.action_encoder(torch.randn(C.LOT, T, model.action_encoder.patch_embed.in_channels, generator=g))
    return emb, act


def test_moteurs():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(1)
    for tache in ("pusht", "tworoom"):
        for moteur in ("coreml-gpu", "coreml-ne"):
            model = swm.wm.utils.load_pretrained(f"{tache}/lewm").eval()
            avant = {k: v.clone() for k, v in model.state_dict().items()}
            ref = model.predict
            with tempfile.TemporaryDirectory() as d:
                info = C.brancher(model, moteur, Path(d) / "p.mlpackage")
                assert all(torch.equal(v, avant[k]) for k, v in model.state_dict().items()), "poids modifiés"
                for T in (1, 2, 3):
                    emb, act = entrees(model, T, g)
                    with torch.no_grad():
                        y0, y1 = ref(emb, act), model.predict(emb, act)
                    assert y1.shape == y0.shape and y1.dtype == y0.dtype, (y1.shape, y0.shape)
                    err = float((y1 - y0).abs().max() / y0.abs().max())
                    assert err < 3e-2, (tache, moteur, T, err)
                    print(f"ok {tache} {moteur} T = {T} : écart relatif max {err:.1e} ({info['disposition']}, "
                          f"poids {info['sha256_poids'][:8]})", flush=True)
    model = swm.wm.utils.load_pretrained("pusht/lewm").eval()
    p = model.predict
    assert C.brancher(model, "torch", "inutile") == {} and model.predict == p
    print("ok moteur torch : rien de changé")


def test_puissance():
    with C.Puissance(pas=0.5) as pw:
        import time
        time.sleep(1.6)
    assert len(pw.w) >= 2 and all(5 < w < 300 for w in pw.w), pw.w
    print(f"ok puissance : {len(pw.w)} mesures, {sum(pw.w) / len(pw.w):.1f} W en moyenne")


if __name__ == "__main__":
    test_puissance()
    test_moteurs()
    print("tous les tests passent")
