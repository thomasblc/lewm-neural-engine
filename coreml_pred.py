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
import os
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

MOTEURS = ("torch", "torch-fp16", "coreml-gpu", "coreml-ne", "ne-complet", "coreml-cpu")
APPELS = {"n": 0, "coreml_s": 0.0, "total_s": 0.0}   # EXP-013 : temps dans l'appel Core ML et autour (conversions, transferts)


def remise_a_zero_appels():
    APPELS.update(n=0, coreml_s=0.0, total_s=0.0)
CODAGES = ("fp16", "int8", "int4b32", "pal4", "int4c")   # EXP-013 : codage des poids dans Core ML (int4c : amendement 4)


def comprimer(ml, codage):
    """EXP-013 : codage des poids du programme Core ML (fonctions de coremltools, depuis les poids 16 bits) : int8 par
    canal, int4 par blocs de 32, int4 par canal, palettes de 16 valeurs (k-moyennes). Vitesse et mémoire seulement : la réussite de
    ces codages est l'objet d'une autre expérience."""
    import coremltools.optimize.coreml as cto
    if codage == "fp16":
        return ml
    if codage == "int8":
        cfg = cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8", granularity="per_channel")
        return cto.linear_quantize_weights(ml, cto.OptimizationConfig(global_config=cfg))
    if codage == "int4b32":
        cfg = cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int4", granularity="per_block", block_size=32)
        return cto.linear_quantize_weights(ml, cto.OptimizationConfig(global_config=cfg))
    if codage == "int4c":                              # amendement 4 : int4 par canal, seul int4 entièrement sur le Neural Engine
        cfg = cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int4", granularity="per_channel")
        return cto.linear_quantize_weights(ml, cto.OptimizationConfig(global_config=cfg))
    if codage == "pal4":
        return cto.palettize_weights(ml, cto.OptimizationConfig(global_config=cto.OpPalettizerConfig(mode="kmeans", nbits=4)))
    raise ValueError(codage)
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


def _empreinte_modele(m):
    """Empreinte du programme Core ML (spécification sérialisée de façon déterministe, sans la date de conversion) :
    le fichier model.mlmodel change d'une conversion à l'autre (ordre de sérialisation), pas son contenu."""
    spec = m.get_spec()
    for k in list(spec.description.metadata.userDefined.keys()):
        if "date" in k.lower():
            del spec.description.metadata.userDefined[k]
    return hashlib.sha256(spec.SerializeToString(deterministic=True)).hexdigest()


def convertir(model, moteur, chemin, codage="fp16"):
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
        ml = comprimer(ml, codage)
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
            "sha256_poids": _empreinte(chemin), "sha256_modele": _empreinte_modele(m), "placement": placement(m, cu),
            "codage": codage, "octets_poids": sum(x.stat().st_size for x in Path(chemin).rglob("weight.bin"))}
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


def etiquette_processus(commande):
    """Nom d'un processus pour repérer un calcul de fond : « python:script.py » pour un python, sinon l'exécutable."""
    mots = commande.split()
    exe = mots[0].split("/")[-1] if mots else ""
    if exe.lower().startswith("python"):
        script = next((m.split("/")[-1] for m in mots[1:] if m.endswith(".py")), "")
        return f"python:{script}"
    return exe


SCRIPTS_EXPERIENCE = ("python:vitesse.py", "python:run_grid.py", "python:repos_puce.py", "python:charge_vision.py",
                      "python:balayage_lot.py", "caffeinate", "ps")
# processus jamais comptés comme calcul de fond : ceux de l'expérience, l'affichage (écran gardé allumé par caffeinate ;
# l'essai le montre à 21-22 % d'un cœur au repos) et launchd (il lance les processus de surveillance de l'expérience ;
# 36 % juste après eux dans l'essai)
NON_COMPTES = SCRIPTS_EXPERIENCE + ("WindowServer", "launchd")


def processus_actifs(n=4):
    """EXP-013 : les n processus qui prennent le plus de CPU (pourcentage, étiquette), pour repérer un calcul de fond."""
    try:
        out = subprocess.run(["ps", "-Ao", "pcpu=,command=", "-r"], capture_output=True, text=True, timeout=5).stdout
        return [[float(l.split(None, 1)[0]), etiquette_processus(l.split(None, 1)[1])] for l in out.splitlines()[:n] if l.strip()]
    except Exception:  # noqa: BLE001
        return []


def placement_detail(m, cu):
    """EXP-013, essai de placement des codages : pour chaque type d'opération, le nombre d'opérations par processeur
    préféré, « aucun » compris (opérations sans processeur dans le plan de calcul, comme les constantes et les opérations
    de décompression des poids ; le moment de la décompression n'est pas observé)."""
    from coremltools.models.compute_plan import MLComputePlan
    plan = MLComputePlan.load_from_path(path=m.get_compiled_model_path(), compute_units=cu)
    d = {}
    for fn in plan.model_structure.program.functions.values():
        for op in fn.block.operations:
            u = plan.get_compute_device_usage_for_mlprogram_operation(op)
            k = "aucun" if u is None else type(u.preferred_compute_device).__name__.replace("ML", "").replace("ComputeDevice", "")
            d.setdefault(op.operator_name, {}).setdefault(k, 0)
            d[op.operator_name][k] += 1
    return d


def _module_coreml(module, entrees, chemin, cu, codage):
    """Convertit un module (formes fixes, entrées {nom: tenseur}) en Core ML 16 bits avec le codage demandé ; rend le
    MLModel chargé sur cu et les empreintes, la taille des poids et le placement."""
    import coremltools as ct
    with torch.no_grad():
        trace = torch.jit.trace(module, tuple(entrees.values()))
    ml = ct.convert(trace, inputs=[ct.TensorType(name=n, shape=tuple(t.shape)) for n, t in entrees.items()],
                    convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
                    minimum_deployment_target=ct.target.macOS15, skip_model_load=True)
    ml = comprimer(ml, codage)
    chemin = Path(chemin)
    shutil.rmtree(chemin, ignore_errors=True)
    ml.save(str(chemin))
    m = ct.models.MLModel(str(chemin), compute_units=cu)
    return m, {"sha256_poids": _empreinte(chemin), "sha256_modele": _empreinte_modele(m), "placement": placement(m, cu),
               "octets_poids": sum(x.stat().st_size for x in chemin.rglob("weight.bin"))}


def brancher_complet(model, chemin, codage="fp16", unite="ne"):
    """EXP-013, moteur ne-complet : tout le world model sur le Neural Engine (prédicteur réécrit « 1ctb », encodeur
    d'images réécrit en (B, C, 1, S) et encodeur d'actions réécrit, encodeur_ane.py), la CEM et le coût sur le CPU (le
    calcul run_grid doit tourner avec --device cpu). L'encodeur d'images n'est appelé que 2 fois par plan (but et
    observation de départ, gardés en cache par le modèle) ; l'encodeur d'actions une fois par itération."""
    import coremltools as ct
    import encodeur_ane as E
    assert next(model.parameters()).device.type == "cpu", "ne-complet : modèle sur le CPU (--device cpu)"
    cu = ct.ComputeUnit.CPU_AND_NE if unite == "ne" else ct.ComputeUnit.CPU_ONLY   # coreml-cpu : témoin hors GPU sans Neural Engine
    m_pred, info = convertir(model, "coreml-ne", Path(str(chemin) + "-predicteur.mlpackage"), codage)
    if unite != "ne":
        m_pred = ct.models.MLModel(str(Path(str(chemin) + "-predicteur.mlpackage")), compute_units=cu)
        info["placement"] = placement(m_pred, cu)
    g = torch.Generator().manual_seed(0)
    m_enc, i_enc = _module_coreml(E.CoeurEncodeurANE(model).eval(), {"pixels": torch.randn(1, 3, E.TAILLE, E.TAILLE, generator=g)},
                                  Path(str(chemin) + "-encodeur.mlpackage"), cu, codage)
    nb = model.action_encoder.patch_embed.in_channels
    m_act, i_act = _module_coreml(E.CoeurActionsANE(model).eval(), {"act": torch.randn(1, nb, E.T_ACTIONS, E.LOT_ACTIONS, generator=g)},
                                  Path(str(chemin) + "-actions.mlpackage"), cu, codage)
    disp = info["disposition"]

    def predict(emb, act_emb):
        t0 = time.perf_counter()
        B, T, _ = emb.shape
        assert B == LOT and T <= T_FIXE, (B, T)
        if T < T_FIXE:
            emb = F.pad(emb, (0, 0, 0, T_FIXE - T))
            act_emb = F.pad(act_emb, (0, 0, 0, T_FIXE - T))
        e, a = P.vers_disposition(emb.detach().float().numpy(), disp), P.vers_disposition(act_emb.detach().float().numpy(), disp)
        t1 = time.perf_counter()
        y = list(m_pred.predict({"emb": np.ascontiguousarray(e), "act": np.ascontiguousarray(a)}).values())[0]
        t2 = time.perf_counter()
        r = torch.from_numpy(np.ascontiguousarray(P.depuis_disposition(y, disp)[:, :T], dtype=np.float32))
        APPELS["n"] += 1
        APPELS["coreml_s"] += t2 - t1
        APPELS["total_s"] += time.perf_counter() - t0
        return r

    model.predict = predict
    model.action_encoder.forward = E.actions_coreml(m_act)
    model.encode = E.encode_coreml(model, m_enc)
    placements = {}
    for d in (info["placement"], i_enc["placement"], i_act["placement"]):
        for k, v in d.items():
            placements[k] = placements.get(k, 0) + v
    return {"moteur": "ne-complet" if unite == "ne" else "coreml-cpu", "disposition": "1ctb, (B, C, 1, S), 1ctb",
            "precision": "float16", "codage": codage,
            "coremltools": info["coremltools"], "macos": info["macos"], "fichier": Path(chemin).name,
            "sha256_poids": "|".join(x["sha256_poids"] for x in (info, i_enc, i_act)),
            "sha256_modele": "|".join(x["sha256_modele"] for x in (info, i_enc, i_act)),
            "placement": placements, "placement_par_modele": {"predicteur": info["placement"], "encodeur": i_enc["placement"],
                                                               "actions": i_act["placement"]},
            "octets_poids": info["octets_poids"] + i_enc["octets_poids"] + i_act["octets_poids"],
            "octets_par_modele": {"predicteur": info["octets_poids"], "encodeur": i_enc["octets_poids"], "actions": i_act["octets_poids"]}}


def brancher(model, moteur, chemin, codage="fp16"):
    """Remplace model.predict par l'appel Core ML. Rend la description (vide pour les moteurs PyTorch)."""
    if moteur in ("torch", "torch-fp16"):
        return {}
    if moteur in ("ne-complet", "coreml-cpu"):
        return brancher_complet(model, str(chemin).removesuffix(".mlpackage"), codage, "ne" if moteur == "ne-complet" else "cpu")
    m, info = convertir(model, moteur, chemin, codage)
    disp = None if info["disposition"] == "origine" else info["disposition"]

    def predict(emb, act_emb):
        t0 = time.perf_counter()
        B, T, _ = emb.shape
        assert B == LOT and T <= T_FIXE, (B, T)
        if T < T_FIXE:
            emb = F.pad(emb, (0, 0, 0, T_FIXE - T))
            act_emb = F.pad(act_emb, (0, 0, 0, T_FIXE - T))
        e, a = emb.detach().to("cpu", torch.float32).numpy(), act_emb.detach().to("cpu", torch.float32).numpy()
        if disp is not None:
            e, a = P.vers_disposition(e, disp), P.vers_disposition(a, disp)
        t1 = time.perf_counter()
        y = list(m.predict({"emb": np.ascontiguousarray(e), "act": np.ascontiguousarray(a)}).values())[0]
        t2 = time.perf_counter()
        y = y if disp is None else P.depuis_disposition(y, disp)
        r = torch.from_numpy(np.ascontiguousarray(y[:, :T], dtype=np.float32)).to(emb.device, emb.dtype)
        APPELS["n"] += 1
        APPELS["coreml_s"] += t2 - t1
        APPELS["total_s"] += time.perf_counter() - t0
        return r

    model.predict = predict
    return info


class Puissance:
    """Mesure la puissance consommée par le Mac pendant un calcul, avec les compteurs des capteurs de la batterie
    (PowerTelemetryData) : AccumulatedSystemLoad ajoute la puissance du système en mW une fois par seconde
    (SystemLoadAccumulatorCount), si bien que (accumulé fin - début) / (compte fin - début) est la puissance moyenne
    exacte entre deux mises à jour du capteur (toutes les 50 à 60 s environ). Mesure de tout le système : sert à
    comparer des moteurs dans les mêmes conditions, avec la puissance au repos mesurée à part. Une lecture toutes les
    5 s dans un fil, avec les conditions : charge moyenne du système, batterie, autres calculs run_grid."""

    def __init__(self, pas=5.0):
        self.pas, self.lectures, self._stop = pas, [], threading.Event()
        self._fil = threading.Thread(target=self._boucle, daemon=True)

    @staticmethod
    def lire():
        s = subprocess.run(["ioreg", "-rn", "AppleSmartBattery"], capture_output=True, text=True).stdout
        i = s.find('"PowerTelemetryData"')
        bloc = s[i:s.find("}", i)] if i >= 0 else ""
        d = {}
        for cle in ("AccumulatedSystemLoad", "SystemLoadAccumulatorCount", "SystemLoad", "SystemPowerIn"):
            k = bloc.find(f'"{cle}"=')
            if k >= 0:
                v = bloc[k + len(cle) + 3:].split(",")[0].strip()
                if v.isdigit():
                    d[cle] = int(v)
        for cle in ("UpdateTime", "CurrentCapacity", "InstantAmperage", "Voltage", "ExternalConnected"):
            k = s.find(f'"{cle}" = ')
            if k >= 0:
                v = s[k + len(cle) + 5:].split("\n")[0].strip()
                if v.isdigit():
                    d[cle] = int(v) - (2 ** 64 if int(v) >= 2 ** 63 else 0)   # InstantAmperage : entier signé sur 64 bits
                elif v in ("Yes", "No"):
                    d[cle] = v == "Yes"
        d["t"] = time.time()
        d["charge_moyenne_1min"] = os.getloadavg()[0]
        # vrais calculs seulement (exécutable python suivi de run_grid.py, comme le lanceur) ; pgrep exclut lui-même et ses
        # ancêtres, donc le calcul en cours : c'est le nombre des AUTRES calculs (0 attendu) ; une commande qui cite
        # seulement le nom (grep, surveillance) ne compte pas
        d["processus_run_grid"] = len(subprocess.run(["pgrep", "-f", "^[^ ]*[Pp]ython[^ ]* ([^ ]+ )*run_grid\\.py( |$)"],
                                                     capture_output=True, text=True).stdout.split())
        d["thermique"] = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True).stdout.strip()[:200]
        d["processus"] = processus_actifs()                # EXP-013 : un calcul de fond qui ne suivrait qu'un moteur
        return d

    def _boucle(self):
        while not self._stop.is_set():
            try:
                self.lectures.append(self.lire())
            except Exception:  # noqa: BLE001  une lecture manquée ne doit pas arrêter le calcul
                pass
            self._stop.wait(self.pas)

    def __enter__(self):
        self._fil.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._fil.join()

    def resume(self):
        """Puissance moyenne (W) entre la première et la dernière mise à jour du capteur vues pendant le calcul, et
        durée couverte (s)."""
        return resume_puissance(self.lectures, self.pas)


def resume_puissance(lectures, pas=5.0):
    """Puissance moyenne (W) sur les seules mises à jour du capteur faites pendant le calcul. Les lectures sont coupées
    en segments là où deux lectures successives sont espacées de plus de 2,5 pas (le fil de lecture est gelé avec le
    calcul pendant une pause de la garde de batterie : le compteur, lui, a continué au repos) ; dans chaque segment, la
    première valeur du compteur date d'avant le segment (calcul précédent, ou pause) et ne sert qu'à repérer la première
    vraie mise à jour. Énergie et secondes s'additionnent sur les segments."""
    v = [x for x in lectures if "AccumulatedSystemLoad" in x and "SystemLoadAccumulatorCount" in x and "t" in x]
    segments, cur = [], []
    for x in v:
        if cur and x["t"] - cur[-1]["t"] > 2.5 * pas:
            segments.append(cur)
            cur = []
        cur.append(x)
    if cur:
        segments.append(cur)
    energie_mj = secondes = 0
    for seg in segments:
        maj = [seg[0]]
        for x in seg[1:]:
            if x["SystemLoadAccumulatorCount"] != maj[-1]["SystemLoadAccumulatorCount"]:
                maj.append(x)
        maj = maj[1:]                                # la première valeur du segment est antérieure au segment
        if len(maj) >= 2:
            energie_mj += maj[-1]["AccumulatedSystemLoad"] - maj[0]["AccumulatedSystemLoad"]
            secondes += maj[-1]["SystemLoadAccumulatorCount"] - maj[0]["SystemLoadAccumulatorCount"]
    return {"puissance_moyenne_w": energie_mj / secondes / 1000 if secondes > 0 else None, "secondes_couvertes": secondes,
            "segments": len(segments)}


def chronometrer(solver, temps, compteur=None, energie=None, fenetres=None, mps=True):
    """Ajoute à la liste temps, pour chaque appel de solve (tous les épisodes encore en cours d'un paquet, planifiés
    l'un après l'autre, 12 à 50 plans), le couple (durée en s, nombre de plans) ; temps moyen par plan de l'appel =
    durée / nombre. Une pause de la garde de batterie (SIGSTOP) tombe dans la durée de l'appel en cours : à résumer
    par la médiane des appels, pas par la moyenne."""
    solve = solver.solve

    sync = mps and torch.backends.mps.is_available()   # EXP-013 : un calcul sur le CPU ne réveille pas le GPU

    def solve_chrono(*a, **k):
        if sync:
            torch.mps.synchronize()
        e0 = compteur.lire() if compteur else None
        h0, t0 = time.time(), time.perf_counter()
        r = solve(*a, **k)
        if sync:
            torch.mps.synchronize()
        temps.append((time.perf_counter() - t0, int(r["actions"].shape[0])))
        if fenetres is not None:                       # EXP-013 : fenêtre d'horloge (time.time, comme la vision)
            fenetres.append((h0, time.time()))
        if compteur:                                   # EXP-013 : énergie pendant la planification seule
            j, _ = compteur.ecart(e0, compteur.lire())
            for k_, v in j.items():
                energie[k_] = energie.get(k_, 0.0) + v
        return r
    solver.solve = solve_chrono
    return solver


def autocast_fp16(solver):
    """EXP-013, moteur torch-fp16 : poids du modèle convertis une fois en float16 (run_grid appelle model.half()), tirages
    et coût de la CEM gardés en float32 (solver._dtype), et torch.autocast en float16 sur le GPU pour les entrées en
    float32 (pixels, actions) : témoin du 16 bits dans PyTorch, sans reconversion des poids à chaque appel."""
    solver._dtype = torch.float32
    solve = solver.solve

    def solve_fp16(*a, **k):
        with torch.autocast(device_type="mps", dtype=torch.float16):
            return solve(*a, **k)
    solver.solve = solve_fp16
    return solver


def echauffer(model, moteur):
    """EXP-013 : mise en route avant la mesure (compilation des noyaux du GPU, premier appel Core ML) : deux appels du
    prédicteur à chacune des trois longueurs du déroulé, un de l'encodeur d'actions et un de l'encodeur, sur des entrées
    de la bonne forme, hors des mesures."""
    dev = next(model.parameters()).device
    dt = next(model.parameters()).dtype
    D = model.predictor.pos_embedding.shape[-1]
    nb = model.action_encoder.patch_embed.in_channels
    with torch.no_grad(), torch.autocast(device_type="mps", dtype=torch.float16, enabled=moteur == "torch-fp16"):
        emb = torch.zeros(LOT, T_FIXE, D, device=dev, dtype=torch.float32 if moteur != "torch-fp16" else dt)
        act = model.action_encoder(torch.zeros(LOT, 5, nb, device=dev, dtype=emb.dtype))
        for t in range(1, T_FIXE + 1):                 # le déroulé appelle le prédicteur avec T = 1, 2 puis 3
            for _ in range(2):
                model.predict(emb[:, :t], act[:, :t])
        model.encode({"pixels": torch.zeros(1, 1, 3, 224, 224, device=dev, dtype=emb.dtype)})
    if dev.type == "mps":
        torch.mps.synchronize()
