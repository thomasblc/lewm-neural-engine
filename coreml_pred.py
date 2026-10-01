"""EXP-012 : brancher le prédicteur de LeWM (prédicteur et pred_proj, model.predict) sur Core ML, en 16 bits, sur le
GPU ou sur le Neural Engine. L'encodeur, la CEM et le coût restent en PyTorch sur le device du calcul.

Deux moteurs :
- "coreml-gpu" : le graphe d'origine (comme essai_ne_codages.py), le plus rapide sur le GPU (1,6 ms par appel) ;
- "coreml-ne"  : le graphe réécrit, lot en dernier axe (predicteur_ane.py, disposition "1ctb"), seul rapide sur le
  Neural Engine (3,8 ms contre 36,6 ms pour l'origine, essai du 1er octobre).

Formes fixes : lot de 300 (la CEM planifie épisode par épisode, 300 candidats) et T = 3 ; les séquences plus courtes
(T = 1 et 2 aux premiers pas du déroulé) sont complétées par des zéros puis recoupées, sans effet sur les T premiers
jetons (attention causale, le reste jeton par jeton ; vérifié par test_coreml_pred.py).

Le modèle Core ML est converti au lancement depuis les poids du modèle chargé (donc après une éventuelle
quantification), enregistré à côté des résultats, et son empreinte SHA-256 (fichier des poids) est rendue.
"""
import hashlib
import platform
import shutil
import threading
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import predicteur_ane as P

MOTEURS = ("torch", "coreml-gpu", "coreml-ne")
LOT, T_FIXE = 300, 3


class _Origine(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, emb, act_emb):
        return self.m.predict(emb, act_emb)


def _empreinte(paquet):
    f = sorted(Path(paquet).rglob("weight.bin"))
    h = hashlib.sha256()
    for x in f:
        h.update(x.read_bytes())
    return h.hexdigest()


def convertir(model, moteur, chemin):
    """Convertit model.predict en Core ML (mlprogram, float16, macOS 15), formes fixes (300, 3) ; rend le MLModel
    chargé sur le moteur demandé et sa description."""
    import coremltools as ct
    assert moteur in ("coreml-gpu", "coreml-ne"), moteur
    cpu = {k: v.detach().to("cpu", torch.float32) for k, v in model.state_dict().items()}
    dev = next(model.parameters()).device
    model.to("cpu", torch.float32)                      # trace sur CPU en float32, puis retour au device
    try:
        D = model.predictor.pos_embedding.shape[-1]
        with torch.no_grad():
            A = model.action_encoder(torch.zeros(1, 1, model.action_encoder.patch_embed.in_channels)).shape[-1]
        g = torch.Generator().manual_seed(0)
        emb, act = torch.randn(LOT, T_FIXE, D, generator=g), torch.randn(LOT, T_FIXE, A, generator=g)
        if moteur == "coreml-ne":
            module, disp = P.CoeurANE(model, "1ctb"), "1ctb"
            ex = (P.vers_disposition(emb, disp).contiguous(), P.vers_disposition(act, disp).contiguous())
        else:
            module, disp = _Origine(model).eval(), None
            ex = (emb, act)
        with torch.no_grad():
            trace = torch.jit.trace(module, ex)
        ml = ct.convert(trace, inputs=[ct.TensorType(name="emb", shape=tuple(ex[0].shape)),
                                       ct.TensorType(name="act", shape=tuple(ex[1].shape))],
                        convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
                        minimum_deployment_target=ct.target.macOS15, skip_model_load=True)
    finally:
        model.to(dev)
    assert all(torch.equal(v.detach().cpu(), cpu[k]) for k, v in model.state_dict().items()), "poids modifiés"
    chemin = Path(chemin)
    shutil.rmtree(chemin, ignore_errors=True)
    ml.save(str(chemin))
    cu = ct.ComputeUnit.CPU_AND_NE if moteur == "coreml-ne" else ct.ComputeUnit.CPU_AND_GPU
    m = ct.models.MLModel(str(chemin), compute_units=cu)
    info = {"moteur": moteur, "disposition": disp or "origine", "precision": "float16", "lot": LOT, "t_fixe": T_FIXE,
            "coremltools": ct.__version__, "macos": platform.mac_ver()[0], "fichier": chemin.name,
            "sha256_poids": _empreinte(chemin), "placement": placement(m, cu)}
    return m, info


def placement(m, cu):
    """Processeur préféré par Core ML pour chaque opération du modèle chargé (plan de calcul, sans exécution) :
    nombre d'opérations par processeur. Un placement indisponible n'arrête pas le calcul."""
    try:
        from coremltools.models.compute_plan import MLComputePlan
        plan = MLComputePlan.load_from_path(path=m.get_compiled_model_path(), compute_units=cu)
        comptes = {}
        for fn in plan.model_structure.program.functions.values():
            for op in fn.block.operations:
                u = plan.get_compute_device_usage_for_mlprogram_operation(op)
                if u is not None:
                    k = type(u.preferred_compute_device).__name__.replace("ML", "").replace("ComputeDevice", "")
                    comptes[k] = comptes.get(k, 0) + 1
        return comptes
    except Exception as e:  # noqa: BLE001
        return {"indisponible": f"{type(e).__name__}: {str(e)[:120]}"}


def brancher(model, moteur, chemin):
    """Remplace model.predict par l'appel Core ML. Rend la description (vide pour le moteur torch)."""
    if moteur == "torch":
        return {}
    m, info = convertir(model, moteur, chemin)
    disp = None if info["disposition"] == "origine" else info["disposition"]

    def predict(emb, act_emb):
        B, T, _ = emb.shape
        assert B == LOT and T <= T_FIXE, (B, T)
        if T < T_FIXE:
            emb = F.pad(emb, (0, 0, 0, T_FIXE - T))
            act_emb = F.pad(act_emb, (0, 0, 0, T_FIXE - T))
        e, a = emb.detach().to("cpu", torch.float32).numpy(), act_emb.detach().to("cpu", torch.float32).numpy()
        if disp is not None:
            e, a = P.vers_disposition(e, disp), P.vers_disposition(a, disp)
        y = list(m.predict({"emb": np.ascontiguousarray(e), "act": np.ascontiguousarray(a)}).values())[0]
        y = y if disp is None else P.depuis_disposition(y, disp)
        return torch.from_numpy(np.ascontiguousarray(y[:, :T], dtype=np.float32)).to(emb.device, emb.dtype)

    model.predict = predict
    return info


class Puissance:
    """Lit la puissance consommée par le Mac (SystemLoad des capteurs de la batterie, en W) toutes les 2 s dans un
    fil à part. Mesure grossière, de tout le système : sert à comparer des moteurs dans les mêmes conditions, pas à
    donner une énergie absolue. Le capteur ne se met à jour qu'environ toutes les 20 s : les lectures successives ne
    sont pas indépendantes, et les premières d'un fichier peuvent encore refléter le calcul précédent (la médiane
    sur un fichier de plusieurs minutes n'en dépend presque pas)."""

    def __init__(self, pas=2.0):
        self.pas, self.w, self._stop = pas, [], threading.Event()
        self._fil = threading.Thread(target=self._boucle, daemon=True)

    @staticmethod
    def lire():
        s = subprocess.run(["ioreg", "-rn", "AppleSmartBattery"], capture_output=True, text=True).stdout
        i = s.find('"SystemLoad"=')
        if i < 0:
            return None
        j = i + len('"SystemLoad"=')
        k = j
        while k < len(s) and s[k].isdigit():
            k += 1
        return int(s[j:k]) / 1000 if k > j else None

    def _boucle(self):
        while not self._stop.is_set():
            v = self.lire()
            if v is not None:
                self.w.append(v)
            self._stop.wait(self.pas)

    def __enter__(self):
        self._fil.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._fil.join()


def chronometrer(solver, temps):
    """Ajoute à la liste temps, pour chaque appel de solve (tous les épisodes encore en cours d'un paquet, planifiés
    l'un après l'autre, 12 à 50 plans), le couple (durée en s, nombre de plans) ; temps moyen par plan de l'appel =
    durée / nombre. Une pause de la garde de batterie (SIGSTOP) tombe dans la durée de l'appel en cours : à résumer
    par la médiane des appels, pas par la moyenne."""
    solve = solver.solve

    def solve_chrono(*a, **k):
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        t0 = time.perf_counter()
        r = solve(*a, **k)
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        temps.append((time.perf_counter() - t0, int(r["actions"].shape[0])))
        return r
    solver.solve = solve_chrono
    return solver
