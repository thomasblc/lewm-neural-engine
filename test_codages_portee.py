"""EXP-015 : tests des codages à portée (sans pred_proj « -sp », pred_proj seule « -pp ») et du groupe simulé predproj.
1. Les codages d'EXP-013 et d'EXP-014 (int8, int4c, pal4) donnent exactement les mêmes poids qu'avant (empreinte égale à
   celle d'EXP-013) : la réécriture de comprimer ne change rien. 2. -sp code les 32 constantes hors pred_proj, -pp ses
   2 seules ; tout reste sur le Neural Engine. 3. predproj:4 (PyTorch, simulé) ne touche que les 2 couches de pred_proj,
   au même arrondi que les autres groupes.

Usage :
    python test_codages_portee.py
"""
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("STABLEWM_HOME", "/Users/thomasblanc/Documents/PRO/world-models-demos/stable-wm")
import torch
import stable_worldmodel as swm

import analyse_exp014 as A14
import coreml_pred as C
import quant

RES = Path(__file__).resolve().parent.parent / "resultats"
model = swm.wm.utils.load_pretrained("pusht/lewm").eval()
with tempfile.TemporaryDirectory() as d:
    for c in ("int8", "int4c", "pal4"):
        m, info = C.convertir(model, "coreml-ne", Path(d) / f"{c}.mlpackage", c)
        ref = json.loads((RES / f"exp013-coreml-ne-{c}-r1" / "fp32_s42.temps.json").read_text())["coreml"]["sha256_poids"]
        assert info["sha256_poids"] == ref, (c, info["sha256_poids"][:12], ref[:12])
    print("ok codages d'EXP-013 inchangés (int8, int4c, pal4 : mêmes empreintes)")
    attendu = {"sp": (32, 30, 0, 2, 0), "pp": (2, 0, 2, 0, 0)}
    for c in C.CODAGES_PORTEE:
        m, info = C.convertir(model, "coreml-ne", Path(d) / f"{c}.mlpackage", c)
        lst = A14.constantes_codees(Path(d) / f"{c}.mlpackage")
        k = A14.categories(lst)
        got = (len(lst), k["blocs"], k["pred_proj"], k["reecriture"], k["autres"])
        assert got == attendu[c.split("-")[1]], (c, got)
        assert info["placement"] == {"NeuralEngine": 361}, (c, info["placement"])
        print(f"ok {c} : {got[0]} constantes codées {k}, placement {info['placement']}, poids {info['octets_poids'] / 1e6:.2f} Mo")
q, touches = quant.quantifier(swm.wm.utils.load_pretrained("pusht/lewm").eval(), "predproj:4")
assert len(touches) == 2 and all(n.startswith("pred_proj") for n in touches), touches
ref = swm.wm.utils.load_pretrained("pusht/lewm").eval()
diff = [n for (n, a), (_, b) in zip(q.state_dict().items(), ref.state_dict().items()) if not torch.equal(a, b)]
assert diff and all(n.startswith("pred_proj") for n in diff), diff
w = dict(q.named_parameters())[diff[0]]
assert w.ndim == 2 and len(torch.unique(torch.round(w / (w.abs().amax(1, keepdim=True) / 7)))) <= 15
print(f"ok predproj:4 : couches touchées {touches}, aucun autre poids modifié, 15 niveaux par canal au plus")
print("tous les tests passent")
