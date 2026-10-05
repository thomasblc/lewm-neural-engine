"""Contrôle d'EXP-018 après chaque run (appelé par lancer_exp018.sh) : le dossier du bras n'a qu'un modèle Core ML
(empreintes des poids et du programme), le même que la même version sur les 500 épisodes (EXP-014, EXP-015 ou EXP-017),
et son protocole est celui d'EXP-005, moteur et calibration exceptés. Sort avec 1 sinon.

Usage : python controle_exp018.py DOSSIER_DU_BRAS SPEC RESULTATS
"""
import json
import sys
from pathlib import Path

import numpy as np

REF = {"exp018-int4c-pusht": "exp014-int4c-pusht", "exp018-int4c-sp-pusht": "exp015-int4c-sp-pusht",
       "exp018-gptq-pusht": "exp017-gptq-pusht"}


def empreintes(d, base):
    return {(str(np.load(f)["coreml_sha256"]), json.loads(f.with_suffix(".temps.json").read_text())["coreml"].get("sha256_modele"))
            for f in sorted(d.glob(f"{base}_s*.npz"))}


def main(out, spec, R):
    out, R = Path(out), Path(R)
    ref, base = REF[out.name], spec.replace(":", "-")
    ici, la = empreintes(out, base), empreintes(R / ref, base)
    p = json.loads((out / "meta.json").read_text())["protocole"]
    p5 = json.loads((R / "exp005" / "meta.json").read_text())["protocole"]
    ok = len(ici) == 1 and ici == la and {k: v for k, v in p.items() if k not in ("moteur", "gptq_calib")} == p5
    print("[contrôle]", out.name, "modèle de", ref, "et protocole d'EXP-005 :", "conformes" if ok else f"NON CONFORMES {ici} {la}",
          flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
