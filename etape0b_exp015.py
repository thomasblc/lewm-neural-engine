"""EXP-015, étape 0 b (Neural Engine, sans GPU ni planification ; avant tout run de réussite d'EXP-015) : le Neural Engine
calcule-t-il le modèle codé aussi fidèlement que le modèle 16 bits ? Méthode du ticket apple/coreai-models n° 279
(K-028) : sortie de la puce contre une référence calculée ailleurs, sur les mêmes entrées.

Pour chaque codage (16 bits, int4c et ses portées -bl, -pp, -sp, palettes), sur 3 lots de 300 x 3 entrées tirées
comme celles de la CEM (plongements N(0, 1), que SIGReg vise ; actions N(0, 1) dans l'espace normalisé, passées par
l'encodeur d'action) :
- y_ne : le prédicteur Core ML exécuté sur le Neural Engine (même conversion que les runs, coreml_pred.convertir) ;
- y_dec : le même graphe réécrit (predicteur_ane.CoeurANE) en float32 sur CPU, avec exactement les poids décodés par
  coremltools (materialized_val_inference) et toutes les autres constantes arrondies en float16 comme dans Core ML
  (biais, normes, plongement de position), sauf la première couche de pred_proj, que Core ML fusionne avec sa
  BatchNorm (poids décodés divisés par le facteur, BatchNorm gardée en float32) : à cette couche près, seule
  l'arithmétique diffère de y_ne ;
- y_sim : la quantification simulée de quant.py (groupe de même portée : predict, predictor, predproj), float32 sur
  CPU, modèle d'origine ;
- y32 : la pleine précision float32.
Mesures (erreur relative ||a - b|| / ||b|| sur tout le lot, et la pire par échantillon) :
- e_exec = err(y_ne, y_dec) : erreur d'exécution du Neural Engine à poids égaux ; lue contre e_exec du 16 bits ;
- e_code = err(y_dec, y_sim) : codage de coremltools contre celui de quant.py, calcul float32 des deux côtés ;
- e_quant = err(y_sim, y32) : taille de l'effet de la quantification simulée.
Plus : les sorties sur le Neural Engine d'int4c-sp et d'int4c-bl sont-elles identiques bit pour bit (matrices 0/1
inertes) ? Règles fixées dans PREREG-EXP015.md (amendement 1) avant ce calcul, et écrites ici :
- validité : e_exec du 16 bits et l'écart du 16 bits sur la puce à float32 sous 1e-2 sur chaque lot, sinon invalide ;
- R1 : pour chaque codage, H2 écartée si e_exec au plus 2 fois celle du 16 bits sur chacun des 3 lots, sinon retenue ;
  matérialité rapportée (e_exec / e_quant, pire échantillon).

Usage (depuis code-exp015, environnement de le-wm ; aucun autre calcul sur le Neural Engine en même temps) :
    python etape0b_exp015.py ../resultats/exp015-etape0b
"""
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

import coreml_pred as C
import etape0_exp015 as E0
import predicteur_ane as P
import quant
import stable_worldmodel as swm

CODAGES = ("fp16", "int4c", "int4c-bl", "int4c-pp", "int4c-sp", "pal4")
SIMULE = {"int4c": "predict:4", "int4c-bl": "predictor:4", "int4c-pp": "predproj:4", "int4c-sp": "predictor:4"}
LOTS = (1, 2, 3)
PLANCHER = 1e-2


def err(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    par = np.linalg.norm((a - b).reshape(a.shape[0], -1), axis=1) / np.linalg.norm(b.reshape(b.shape[0], -1), axis=1)
    return {"relative": float(np.linalg.norm(a - b) / np.linalg.norm(b)), "pire_echantillon": float(par.max())}


def entrees(model, graine):
    g = torch.Generator().manual_seed(graine)
    D = model.predictor.pos_embedding.shape[-1]
    n_act = model.action_encoder.patch_embed.in_channels
    emb = torch.randn(C.LOT, C.T_FIXE, D, generator=g)
    with torch.no_grad():
        act = model.action_encoder(torch.randn(C.LOT, C.T_FIXE, n_act, generator=g))
    return emb, act


def poids_decodes(model, codage, k16, ref):
    """CoeurANE float32 dont les 34 constantes valent exactement les poids décodés par coremltools (ou les poids 16 bits
    pour « fp16 »). const_7 porte la BatchNorm fusionnée (étape 0) : divisée par le facteur pour la couche non fusionnée."""
    coeur = P.CoeurANE(model, "1ctb").eval()
    if codage == "fp16":
        vals = dict(k16)
    else:
        k, z = E0.constantes(C.comprimer(E0.programme16(model), codage), k16)
        vals = {**k, **{n: op.materialized_val_inference() for n, op in z.items()}}
    cible = {}
    for i, b in enumerate(coeur.blocs):
        for nom, conv in (("mod", b.mod), ("qkv", b.qkv), ("sortie", b.sortie), ("fc1", b.fc1), ("fc2", b.fc2)):
            cible[f"blocs_{i}_{nom}_weight_to_fp16"] = (conv, None)
    f_bn = ref["const_7_to_fp16"][2]
    cible["const_7_to_fp16"] = (coeur.proj[0], f_bn)
    cible["proj_3_weight_to_fp16"] = (coeur.proj[3], None)
    with torch.no_grad():
        for n, (conv, f) in cible.items():
            v = vals[n].astype(np.float64)
            if f is not None:
                v = v / f[:, None, None, None]
            assert tuple(v.shape) == tuple(conv.weight.shape), (n, v.shape, conv.weight.shape)
            conv.weight.copy_(torch.from_numpy(v).float())
        for b in coeur.blocs:                    # matrices 0/1 : décodées exactement (étape 0)
            for nom in ("somme_tetes", "etale_tetes"):
                v = vals[f"blocs_0_{nom}_to_fp16"].astype(np.float32)
                assert np.array_equal(v, getattr(b, nom).numpy()), nom
        # toutes les constantes en float16 comme dans Core ML, sauf la couche fusionnée avec sa BatchNorm (proj.0, proj.1)
        for nom, x in list(coeur.named_parameters()) + list(coeur.named_buffers()):
            if not nom.startswith(("proj.0.", "proj.1.")) and x.is_floating_point():
                x.copy_(x.half().float())
    return coeur


def sortie_coeur(coeur, emb, act):
    with torch.no_grad():
        y = coeur(P.vers_disposition(emb, "1ctb").contiguous(), P.vers_disposition(act, "1ctb").contiguous())
    return P.depuis_disposition(y, "1ctb").numpy()


def sortie_ne(m, emb, act):
    out = m.predict({"emb": np.ascontiguousarray(P.vers_disposition(emb.numpy(), "1ctb"), dtype=np.float32),
                     "act": np.ascontiguousarray(P.vers_disposition(act.numpy(), "1ctb"), dtype=np.float32)})
    assert len(out) == 1, list(out)
    return P.depuis_disposition(np.asarray(next(iter(out.values())), np.float32), "1ctb")


def main(sortie):
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    sale = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip())
    charger = lambda: swm.wm.utils.load_pretrained("pusht/lewm").eval()
    model = charger()
    k16, _ = E0.constantes(E0.programme16(model))
    ref = E0.references(model)
    X = [entrees(model, g) for g in LOTS]
    with torch.no_grad():
        y32 = [model.predict(e, a).numpy() for e, a in X]
    sims = {}
    for spec in sorted(set(SIMULE.values())):
        q, touches = quant.quantifier(charger(), spec)
        with torch.no_grad():
            sims[spec] = [q.predict(e, a).numpy() for e, a in X]
    import coremltools as ct
    res = {"git": git, "git_sale": sale, "lots": list(LOTS), "coremltools": ct.__version__, "macos": platform.mac_ver()[0],
           "codages": {}}
    ne = {}
    with tempfile.TemporaryDirectory() as d:
        for c in CODAGES:
            m, info = C.convertir(model, "coreml-ne", Path(d) / f"{c}.mlpackage", c)
            assert set(info["placement"]) == {"NeuralEngine"}, (c, info["placement"])
            coeur = poids_decodes(model, c, k16, ref)
            y_ne = [sortie_ne(m, e, a) for e, a in X]
            y_dec = [sortie_coeur(coeur, e, a) for e, a in X]
            ne[c] = y_ne
            r = {"sha256_poids": info["sha256_poids"], "placement": info["placement"],
                 "e_exec": [err(a, b) for a, b in zip(y_ne, y_dec)], "err_ne_contre_fp32": [err(a, b) for a, b in zip(y_ne, y32)],
                 "err_dec_contre_fp32": [err(a, b) for a, b in zip(y_dec, y32)]}
            if c in SIMULE:
                s = sims[SIMULE[c]]
                r.update({"simule": SIMULE[c], "e_code": [err(a, b) for a, b in zip(y_dec, s)],
                          "e_quant": [err(a, b) for a, b in zip(s, y32)], "err_ne_contre_simule": [err(a, b) for a, b in zip(y_ne, s)]})
            res["codages"][c] = r
    res["sp_et_bl_identiques_sur_le_neural_engine"] = bool(all(np.array_equal(a, b) for a, b in zip(ne["int4c-sp"], ne["int4c-bl"])))
    res["sp_contre_bl"] = [err(a, b) for a, b in zip(ne["int4c-sp"], ne["int4c-bl"])]
    resume = {c: {k: float(np.mean([x["relative"] for x in v[k]]))
                  for k in ("e_exec", "e_code", "e_quant", "err_ne_contre_fp32") if k in v} for c, v in res["codages"].items()}
    C16 = res["codages"]["fp16"]
    res["valide"] = bool(all(C16["e_exec"][i]["relative"] < PLANCHER and C16["err_ne_contre_fp32"][i]["relative"] < PLANCHER
                             for i in range(len(LOTS))))
    res["R1"] = {}
    for c, v in res["codages"].items():
        rapports = [v["e_exec"][i]["relative"] / C16["e_exec"][i]["relative"] for i in range(len(LOTS))]
        resume[c]["e_exec_sur_e_exec_16_bits_par_lot"] = rapports
        resume[c]["pire_echantillon_sur_16_bits_par_lot"] = [v["e_exec"][i]["pire_echantillon"] / C16["e_exec"][i]["pire_echantillon"]
                                                            for i in range(len(LOTS))]
        if "e_quant" in v:
            resume[c]["e_exec_sur_e_quant"] = resume[c]["e_exec"] / resume[c]["e_quant"]
        if c != "fp16":
            res["R1"][c] = "H2 écartée" if all(x <= 2 for x in rapports) else "H2 retenue"
    res["resume"] = resume
    out = Path(sortie)
    out.mkdir(parents=True, exist_ok=True)
    (out / "resultats.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps({k: res[k] for k in ("git", "git_sale", "valide", "R1", "resume", "sp_et_bl_identiques_sur_le_neural_engine")},
                     indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1])
