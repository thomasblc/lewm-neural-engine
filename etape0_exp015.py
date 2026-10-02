"""EXP-015, étape 0 (CPU seul : ni GPU ni Neural Engine, aucun modèle chargé ni compilé ; avant le pré-enregistrement des
bras) : ce que coremltools code exactement, comparé à la quantification simulée de PyTorch (quant.py), sur LeWM PushT.

1. Correspondance des 34 constantes de plus de 2 048 éléments du programme 16 bits (graphe réécrit « 1ctb ») avec les
   couches de PyTorch : 30 poids des blocs (groupe predictor), 2 de pred_proj (groupe predproj), 2 matrices 0/1 de la
   réécriture (sans équivalent dans PyTorch). Pour chacune : le poids 16 bits vaut-il float16 du poids float32 attendu
   (lignes de q multipliées par 1/8, exact ; BatchNorm de pred_proj fusionnée ou non dans const_7) ?
2. int4 par canal de coremltools contre RTN par canal de quant.py (float32) : part des codes entiers différents, écart
   maximal des codes, par constante et par catégorie. Les codes par ligne ne changent pas quand une ligne est multipliée
   par un facteur (1/8 pour q, BatchNorm fusionnée) en calcul exact ; seuls les arrondis float16 les déplacent.
3. Matrices 0/1 : valeurs décodées sous int4c et pal4.
4. Erreur relative de reconstruction ||Wq - W|| / ||W|| par catégorie : int4c (coremltools), RTN 4 bits (quant.py),
   palettes (coremltools).
5. Portées (-sp, -pp, -bl) : constantes codées, et constantes laissées bit pour bit égales au programme 16 bits.

Usage (depuis code-exp015, environnement de le-wm) :
    python etape0_exp015.py ../resultats/exp015-etape0
"""
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("STABLEWM_HOME", "/Users/thomasblanc/Documents/PRO/world-models-demos/stable-wm")
import numpy as np
import torch
import coremltools as ct
import stable_worldmodel as swm

import coreml_pred as C
import predicteur_ane as P
import quant


def programme16(model):
    """Même conversion que coreml_pred.convertir pour le Neural Engine, sans enregistrer ni charger le modèle."""
    D = model.predictor.pos_embedding.shape[-1]
    with torch.no_grad():
        A = model.action_encoder(torch.zeros(1, 1, model.action_encoder.patch_embed.in_channels)).shape[-1]
    g = torch.Generator().manual_seed(0)
    emb, act = torch.randn(C.LOT, C.T_FIXE, D, generator=g), torch.randn(C.LOT, C.T_FIXE, A, generator=g)
    ex = (P.vers_disposition(emb, "1ctb").contiguous(), P.vers_disposition(act, "1ctb").contiguous())
    with torch.no_grad():
        trace = torch.jit.trace(P.CoeurANE(model, "1ctb"), ex)
    return ct.convert(trace, inputs=[ct.TensorType(name="emb", shape=tuple(ex[0].shape)),
                                     ct.TensorType(name="act", shape=tuple(ex[1].shape))],
                      convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
                      minimum_deployment_target=ct.target.macOS15, skip_model_load=True)


def constantes(ml, noms16=None):
    """{nom: valeur} des constantes 16 bits de plus de 2 048 éléments (hors données des décompressions) et {nom: op} des
    décompressions (constexpr). coremltools nomme la sortie d'une décompression d'après la constante d'origine plus un
    suffixe (« _quantized », « _palettized ») : avec noms16, la clé redevient le nom d'origine (préfixe unique)."""
    ops = [op for fn in ml._mil_program.functions.values() for op in fn.operations]
    z = {op.outputs[0].name: op for op in ops if op.op_type.startswith("constexpr")}
    donnees = {v.name for op in z.values() for v in op.inputs.values()}
    k = {op.outputs[0].name: op.outputs[0].val for op in ops
         if op.op_type == "const" and op.outputs[0].name not in donnees and op.outputs[0].val is not None
         and getattr(op.outputs[0].val, "size", 0) > 2048}
    if noms16 is not None:
        zz = {}
        for s, op in z.items():
            m = [n for n in noms16 if s.startswith(n + "_")]
            assert len(m) == 1, (s, m)
            zz[m[0]] = op
        z = zz
    return k, z


def references(model):
    """Poids float32 de PyTorch attendus pour chaque constante (forme (sortie, entrée)), facteur par ligne compris."""
    ref = {}
    for i, b in enumerate(model.predictor.transformer.layers):
        a = b.attn
        inner = a.to_qkv.out_features // 3
        f = torch.ones(3 * inner)
        f[:inner] = a.scale
        ref[f"blocs_{i}_mod_weight_to_fp16"] = (f"predictor.transformer.layers.{i}.adaLN_modulation.1", b.adaLN_modulation[1].weight, None)
        ref[f"blocs_{i}_qkv_weight_to_fp16"] = (f"predictor.transformer.layers.{i}.attn.to_qkv", a.to_qkv.weight, f)
        ref[f"blocs_{i}_sortie_weight_to_fp16"] = (f"predictor.transformer.layers.{i}.attn.to_out.0", a.to_out[0].weight, None)
        ref[f"blocs_{i}_fc1_weight_to_fp16"] = (f"predictor.transformer.layers.{i}.mlp.net.1", b.mlp.net[1].weight, None)
        ref[f"blocs_{i}_fc2_weight_to_fp16"] = (f"predictor.transformer.layers.{i}.mlp.net.4", b.mlp.net[4].weight, None)
    net = model.pred_proj.net
    bn = net[1]
    f_bn = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach()
    ref["const_7_to_fp16"] = ("pred_proj.net.0", net[0].weight, f_bn)    # BatchNorm fusionnée : vérifié au point 1
    ref["proj_3_weight_to_fp16"] = ("pred_proj.net.3", net[3].weight, None)
    return {k: (n, w.detach().float().clone(), None if f is None else f.double().numpy()) for k, (n, w, f) in ref.items()}


def categorie(nom):
    return "tetes" if "tetes" in nom else ("blocs" if nom.startswith("blocs_") else "pred_proj")


def erreur(a, b):
    return float(np.linalg.norm((a - b).ravel()) / np.linalg.norm(b.ravel()))


@torch.no_grad()
def main(sortie):
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    sale = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip())
    model = swm.wm.utils.load_pretrained("pusht/lewm").eval()
    poids0 = {k: v.clone() for k, v in model.state_dict().items()}
    ml16 = programme16(model)
    k16, z16 = constantes(ml16)
    assert not z16 and len(k16) == 34, (len(k16), len(z16))
    ref = references(model)
    assert set(ref) | {n for n in k16 if "tetes" in n} == set(k16), sorted(set(k16) ^ set(ref))
    res = {"git": git, "git_sale": sale, "coremltools": ct.__version__, "torch": torch.__version__, "constantes": {}}
    # les couches que quant.py arrondit (predictor:4, predproj:4) sont exactement celles des constantes correspondantes
    _, t_pred = quant.quantifier(swm.wm.utils.load_pretrained("pusht/lewm").eval(), "predictor:4")
    _, t_pp = quant.quantifier(swm.wm.utils.load_pretrained("pusht/lewm").eval(), "predproj:4")
    res["correspondance"] = {
        "predictor4": sorted(t_pred) == sorted(v[0] for n, v in ref.items() if n.startswith("blocs_")),
        "predproj4": sorted(t_pp) == sorted(v[0] for n, v in ref.items() if not n.startswith("blocs_")),
        "couches_predictor4": len(t_pred), "couches_predproj4": len(t_pp)}
    assert res["correspondance"]["predictor4"] and res["correspondance"]["predproj4"], res["correspondance"]

    # 1. le poids 16 bits contre float16 du poids attendu
    for n, (nom_t, w32, f) in ref.items():
        w = w32.double().numpy()
        w16 = k16[n].reshape(k16[n].shape[0], -1).astype(np.float64)
        attendu = w if f is None else w * f[:, None]
        r = {"couche_torch": nom_t, "forme": list(w.shape), "categorie": categorie(n),
             "egal_float16_attendu": bool(np.array_equal(w16, attendu.astype(np.float16).astype(np.float64))),
             "ecart_max_relatif_au_max_ligne": float((np.abs(w16 - attendu).max(1) / np.abs(attendu).max(1)).max())}
        if n == "const_7_to_fp16":
            r["egal_float16_sans_batchnorm"] = bool(np.array_equal(w16, w.astype(np.float16).astype(np.float64)))
        res["constantes"][n] = r
    for n in k16:
        if "tetes" in n:
            v = k16[n]
            res["constantes"][n] = {"categorie": "tetes", "forme": list(v.shape), "valeurs": sorted(np.unique(v).astype(float).tolist())}

    # 2 à 4. int4 par canal et palettes de coremltools, sur tout le programme (codage d'EXP-013 et EXP-014)
    dec = {}
    for codage in ("int4c", "pal4"):
        k, z = constantes(C.comprimer(programme16(model), codage), k16)
        assert len(z) == 34 and not k, (codage, len(z), len(k))
        dec[codage] = {n: op for n, op in z.items()}
    for n in k16:
        r = res["constantes"][n]
        op = dec["int4c"][n]
        assert op.op_type == "constexpr_blockwise_shift_scale", op.op_type
        codes = op.inputs["data"].val.reshape(k16[n].shape[0], -1).astype(np.int64)
        assert op.inputs.get("offset") is None, n
        d_int4 = op.materialized_val_inference().reshape(k16[n].shape[0], -1).astype(np.float64)
        d_pal = dec["pal4"][n].materialized_val_inference().reshape(k16[n].shape[0], -1).astype(np.float64)
        w16 = k16[n].reshape(k16[n].shape[0], -1).astype(np.float64)
        r["int4c_codes_min_max"] = [int(codes.min()), int(codes.max())]
        if n in ref:
            _, w32, f = ref[n]
            w = w32.double().numpy()
            attendu = w if f is None else w * f[:, None]
            # codes de la simulation, calculés comme quant.rtn_par_canal (float32, CPU, comme run_grid avant le device)
            s32 = w32.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 7
            kt = (w32 / s32).round().clamp(-7, 7)
            assert torch.equal(kt * s32, quant.rtn_par_canal(w32, 4)), n
            k32 = kt.numpy().astype(np.int64)
            sim = (kt * s32).double().numpy()
            sim = sim if f is None else sim * f[:, None]
            diff = codes != k32
            r.update({"int4c_part_codes_differents": float(diff.mean()), "int4c_codes_differents": int(diff.sum()),
                      "int4c_ecart_max_codes": int(np.abs(codes - k32).max()), "elements": int(codes.size),
                      "erreur_int4c_coreml": erreur(d_int4, attendu), "erreur_rtn4_torch": erreur(sim, attendu),
                      "erreur_pal4_coreml": erreur(d_pal, attendu), "erreur_float16": erreur(w16, attendu),
                      "ecart_int4c_coreml_contre_rtn4_torch": erreur(d_int4, sim)})
        else:
            r.update({"int4c_valeurs_decodees": sorted(np.unique(d_int4).tolist()),
                      "pal4_valeurs_decodees": sorted(np.unique(d_pal).tolist()),
                      "int4c_ecart_max": float(np.abs(d_int4 - w16).max()), "pal4_ecart_max": float(np.abs(d_pal - w16).max())})
    par_cat = {}
    for cat in ("blocs", "pred_proj"):
        L = [v for v in res["constantes"].values() if v["categorie"] == cat]
        tot = sum(v["elements"] for v in L)
        par_cat[cat] = {"constantes": len(L), "elements": tot,
                        "part_codes_differents": sum(v["int4c_codes_differents"] for v in L) / tot,
                        "ecart_max_codes": max(v["int4c_ecart_max_codes"] for v in L),
                        **{c: float(np.mean([v[c] for v in L])) for c in ("erreur_int4c_coreml", "erreur_rtn4_torch",
                                                                           "erreur_pal4_coreml", "erreur_float16")},
                        "tous_egaux_float16_attendu": all(v["egal_float16_attendu"] for v in L)}
    res["par_categorie"] = par_cat

    # 5. portées
    res["portees"] = {}
    for codage in C.CODAGES_PORTEE:
        k, z = constantes(C.comprimer(programme16(model), codage), k16)
        codees = sorted(z)
        inchangees = all(np.array_equal(k[n], k16[n]) for n in k)
        assert set(k) | set(z) == set(k16), codage
        cats = {c: sum(categorie(n) == c for n in codees) for c in ("blocs", "pred_proj", "tetes")}
        res["portees"][codage] = {"codees": len(codees), "par_categorie": cats, "non_codees_identiques_au_16_bits": inchangees}
    assert all(torch.equal(v, poids0[k]) for k, v in model.state_dict().items()), "poids du modèle modifiés"
    out = Path(sortie)
    out.mkdir(parents=True, exist_ok=True)
    (out / "resultats.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps({k: res[k] for k in ("git", "git_sale", "correspondance", "par_categorie", "portees")}, indent=1, ensure_ascii=False))
    print(json.dumps({n: v for n, v in res["constantes"].items() if v["categorie"] != "blocs"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1])
