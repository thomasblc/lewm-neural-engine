"""EXP-017, étape 0 (avant tout run de réussite) : les codes de GPTQ (quant.py, « predict:4:gptq », calibration
d'EXP-009) sont-ils écrits exactement dans Core ML ?
1. Codes du modèle Core ML (`coreml_pred.comprimer_gptq`) contre codes de GPTQ : nombre de différences (attendu 0) ;
   valeurs décodées contre poids GPTQ (écart maximal en part du maximum de ligne).
2. Repère : le quantificateur de coremltools appliqué aux mêmes poids (int4c) : part des codes changés.
3. Placement sur le Neural Engine, taille du fichier des poids, constantes codées par catégorie.
4. Descriptif : sortie du prédicteur sur le Neural Engine contre la sortie GPTQ simulée en float32 (3 lots), à côté du
   repère 16 bits sur le Neural Engine contre float32.

Usage (depuis code-exp017, environnement de le-wm, aucun autre calcul en cours) :
    python etape0_exp017.py ../resultats/exp017-etape0
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

import analyse_exp014 as A14
import coreml_pred as C
import etape0_exp015 as E0
import etape0b_exp015 as E0B
import quant
import stable_worldmodel as swm

CALIB = Path(__file__).resolve().parent.parent / "resultats" / "exp009-calib-pusht" / "calib.pt"


def main(sortie):
    os.environ["LEWM_GPTQ_CALIB"] = str(CALIB)
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    sale = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip())
    charger = lambda: swm.wm.utils.load_pretrained("pusht/lewm").eval()
    q, touches = quant.quantifier(charger(), "predict:4:gptq")
    res = {"git": git, "git_sale": sale, "couches_gptq": len(touches)}
    # 1. codes écrits et valeurs décodées
    ml = C.comprimer_gptq(E0.programme16(q), q)
    ops = {op.outputs[0].name: op for fn in ml._mil_program.functions.values() for op in fn.operations if op.op_type.startswith("constexpr")}
    diff = n = 0
    pire = 0.0
    for nom, (mod, f) in C.correspondance_constantes(q).items():
        op = [o for s, o in ops.items() if s.startswith(nom + "_")]
        assert len(op) == 1, nom
        data = op[0].inputs["data"].val.reshape(mod.weight.shape[0], -1).astype(np.int64)
        s = mod.echelle_gptq.reshape(-1, 1)
        kg = torch.round(mod.weight.detach() / s).numpy().astype(np.int64)
        fac = np.ones(kg.shape[0]) if f is None else f.numpy()
        diff += int((data != kg * np.sign(fac)[:, None]).sum())
        n += data.size
        dec = op[0].materialized_val_inference().reshape(kg.shape).astype(np.float64)
        att = mod.weight.detach().double().numpy() * fac[:, None]
        pire = max(pire, float((np.abs(dec - att) / np.abs(att).max(1, keepdims=True)).max()))
    res["codes_differents"], res["codes"], res["ecart_decode_max_part_ligne"] = diff, n, pire
    # 2. repère : quantificateur de coremltools sur les poids GPTQ
    k16, _ = E0.constantes(E0.programme16(q))
    _, z = E0.constantes(C.comprimer(E0.programme16(q), "int4c"), k16)
    d2 = n2 = 0
    for nom, (mod, f) in C.correspondance_constantes(q).items():
        data = z[nom].inputs["data"].val.reshape(mod.weight.shape[0], -1).astype(np.int64)
        s = mod.echelle_gptq.reshape(-1, 1)
        kg = torch.round(mod.weight.detach() / s).numpy().astype(np.int64)
        fac = np.ones(kg.shape[0]) if f is None else f.numpy()
        d2 += int((data != kg * np.sign(fac)[:, None]).sum())
        n2 += data.size
    res["coremltools_part_codes_changes"] = d2 / n2
    # 3 et 4. modèle chargé sur le Neural Engine
    m0 = charger()
    X = [E0B.entrees(m0, g) for g in E0B.LOTS]
    with torch.no_grad():
        y32 = [m0.predict(e, a).numpy() for e, a in X]
        ygs = [q.predict(e, a).numpy() for e, a in X]
    with tempfile.TemporaryDirectory() as d:
        m, info = C.convertir(q, "coreml-ne", Path(d) / "g.mlpackage", "int4c-gptq")
        k = A14.categories(A14.constantes_codees(Path(d) / "g.mlpackage"))
        y_ne = [E0B.sortie_ne(m, e, a) for e, a in X]
        m16, info16 = C.convertir(m0, "coreml-ne", Path(d) / "f.mlpackage", "fp16")
        y16 = [E0B.sortie_ne(m16, e, a) for e, a in X]
    res.update({"placement": info["placement"], "octets_poids": info["octets_poids"], "sha256_poids": info["sha256_poids"],
                "constantes": {c: k[c] for c in ("blocs", "pred_proj", "reecriture", "autres")},
                "ne_contre_gptq_simule": [E0B.err(a, b) for a, b in zip(y_ne, ygs)],
                "repere_ne16_contre_fp32": [E0B.err(a, b) for a, b in zip(y16, y32)],
                "gptq_simule_contre_fp32": [E0B.err(a, b) for a, b in zip(ygs, y32)]})
    out = Path(sortie)
    out.mkdir(parents=True, exist_ok=True)
    (out / "resultats.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in res.items()}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1])
