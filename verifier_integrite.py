"""Vérification d'intégrité d'EXP-009 (GPTQ contre RTN), refaite à partir des fichiers bruts. Répond, preuve par
preuve, à la question « y a-t-il pu avoir triche ? » : du modèle (une version « compressée » qui serait en fait la
pleine précision, ou qui recevrait la réponse), de la calibration (réglée sur les épisodes de test), ou de
l'expérimentateur (chiffres retouchés, analyse changée après coup). N'importe qui peut relancer ce script et doit
obtenir les mêmes chiffres. Écrit resultats/integrite-exp009/rapport.md, resultats.json et manifeste.sha256.

Usage (depuis code-exp009, environnement de le-wm) :
    python verifier_integrite.py
"""
import hashlib
import json
import os
import subprocess
from pathlib import Path

os.environ.setdefault("STABLEWM_HOME", "/Users/thomasblanc/Documents/PRO/world-models-demos/stable-wm")
import numpy as np
import torch

import analyse_exp003 as A
import analyse_exp006 as A6
import quant

ICI = Path(__file__).resolve().parent
R = ICI.parent / "resultats"
OUT = R / "integrite-exp009"
GRAINES = (42, 43, 44, 45, 46)
TACHES = {"pusht": ("exp009-pusht", "exp003", ("exp003", "exp005", "sanite-n50")),
          "tworoom": ("exp009-tworoom", "exp006", ("exp006",))}


def git(*args):
    return subprocess.run(["git", "-C", str(ICI), *args], capture_output=True, text=True).stdout.strip()


def couches(m):
    return {n: x for n, x in m.named_modules()
            if n.split(".")[0] in quant.GROUPES["all"] and isinstance(x, (torch.nn.Linear, torch.nn.Conv2d))}


def preuve_poids(tache):
    """1. Les versions GPTQ sont vraiment compressées : au plus 2^b - 1 valeurs par ligne, poids différents."""
    import stable_worldmodel as swm
    os.environ["LEWM_GPTQ_CALIB"] = str(R / f"exp009-calib-{tache}" / "calib.pt")
    f = couches(swm.wm.utils.load_pretrained(f"{tache}/lewm"))
    out = {}
    for bits in (4, 3):
        g = couches(quant.quantifier(swm.wm.utils.load_pretrained(f"{tache}/lewm"), f"all:{bits}:gptq")[0])
        r = couches(quant.quantifier(swm.wm.utils.load_pretrained(f"{tache}/lewm"), f"all:{bits}")[0])
        nmax, df, dr, tot = 0, 0, 0, 0
        for n in f:
            wg, wf, wr = (c[n].weight.detach().reshape(c[n].weight.shape[0], -1) for c in (g, f, r))
            nmax = max(nmax, max(len(torch.unique(l)) for l in wg))
            df, dr, tot = df + int((wg != wf).sum()), dr + int((wg != wr).sum()), tot + wg.numel()
        out[f"{bits} bits"] = {"couches": len(f), "valeurs_max_par_ligne": nmax, "autorise": 2 ** bits - 1,
                               "part_differente_de_fp32": df / tot, "part_differente_de_rtn": dr / tot}
        A.exiger(nmax <= 2 ** bits - 1 and df / tot > 0.9, f"{tache} {bits} bits : poids pas sur la grille ou trop proches de fp32")
    return out


def preuve_trajectoires(tache):
    """2. Pas de pleine précision déguisée : le hasard est apparié, donc une copie de la pleine précision aurait des
    trajectoires identiques, octet pour octet. 3. Le succès est recalculé depuis l'état physique enregistré."""
    dg, dr, _ = TACHES[tache]
    controler = A6.controler if tache == "tworoom" else (lambda d, nom: A.controler_course(d, nom))
    out = {}
    for b in (4, 3):
        ident, tot, accord = 0, 0, []
        for s in GRAINES:
            q, f = dict(np.load(R / dg / f"all-{b}-gptq_s{s}.npz")), dict(np.load(R / dr / f"fp32_s{s}.npz"))
            controler(q, f"{dg} all:{b}:gptq graine {s}")      # lève une erreur si succès et état physique divergent
            ident += sum(np.array_equal(q["state"][:, i], f["state"][:, i], equal_nan=True) for i in range(q["state"].shape[1]))
            tot += q["state"].shape[1]
            accord.append(float((q["success"] == f["success"]).mean()))
        out[f"{b} bits"] = {"trajectoires_identiques_a_fp32": int(ident), "sur": int(tot), "meme_issue_que_fp32": float(np.mean(accord))}
    f42, f43 = dict(np.load(R / dr / "fp32_s42.npz")), dict(np.load(R / dr / "fp32_s43.npz"))
    out["repere_fp32_entre_deux_graines"] = float((f42["success"] == f43["success"]).mean())
    return out


def preuve_calibration(tache):
    """4. La calibration n'a vu aucune trajectoire des évaluations."""
    _, _, evals = TACHES[tache]
    c = torch.load(R / f"exp009-calib-{tache}" / "calib.pt", map_location="cpu")
    vus = set()
    for e in evals:
        m = json.loads((R / e / "meta.json").read_text())
        vus |= set(m.get("protocole", m)["episodes_idx"])
    commun = set(c["episodes_idx"]) & vus
    A.exiger(not commun, f"{tache} : calibration et évaluations partagent {len(commun)} trajectoires")
    return {"trajectoires_calibration": len(set(c["episodes_idx"])), "trajectoires_evaluation": len(vus), "en_commun": 0}


def preuve_provenance():
    """5. Pré-enregistrement daté avant les données ; un seul commit pour tous les fichiers ; analyse figée."""
    commit_prereg = git("log", "-1", "--format=%H %cI", "--", "PREREG-EXP009.md")
    shas, premier = set(), None
    for dg, _, _ in TACHES.values():
        for f in sorted((R / dg).glob("*.npz")):
            shas.add(str(np.load(f)["git"]))
            t = f.stat().st_mtime
            premier = t if premier is None else min(premier, t)
    import datetime
    return {"dernier_commit_du_preregistrement": commit_prereg, "commits_des_fichiers": sorted(shas),
            "premier_fichier_ecrit": datetime.datetime.fromtimestamp(premier).astimezone().isoformat(timespec="seconds")}


def manifeste():
    """6. Empreintes SHA-256 de tous les fichiers de résultats d'EXP-009 et de ses références : tout changement
    ultérieur d'un fichier se verra."""
    lignes = []
    for d in ("exp009-pusht", "exp009-tworoom", "exp009-calib-pusht", "exp009-calib-tworoom", "exp009-repro-pusht",
              "exp009-repro-tworoom", "exp003", "exp004", "exp006"):
        for f in sorted((R / d).glob("*")):
            if f.is_file() and not f.name.endswith(".log"):
                lignes.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {d}/{f.name}")
    return lignes


def main():
    OUT.mkdir(exist_ok=True)
    r = {"code": {"commit": git("rev-parse", "HEAD"), "script": "verifier_integrite.py"}}
    for t in TACHES:
        r[t] = {"poids": preuve_poids(t), "trajectoires": preuve_trajectoires(t), "calibration": preuve_calibration(t)}
    r["provenance"] = preuve_provenance()
    m = manifeste()
    (OUT / "manifeste.sha256").write_text("\n".join(m) + "\n")
    r["manifeste"] = {"fichiers": len(m), "empreinte_du_manifeste": hashlib.sha256(("\n".join(m) + "\n").encode()).hexdigest()}
    (OUT / "resultats.json").write_text(json.dumps(r, indent=1, ensure_ascii=False))
    L = ["# Intégrité d'EXP-009 : vérifications refaites depuis les fichiers bruts", ""]
    for t in TACHES:
        for b in ("4 bits", "3 bits"):
            p, tr = r[t]["poids"][b], r[t]["trajectoires"][b]
            L.append(f"- {t}, GPTQ {b} : au plus {p['valeurs_max_par_ligne']} valeurs par ligne (autorisé {p['autorise']}) sur "
                     f"{p['couches']} couches ; {100 * p['part_differente_de_fp32']:.1f} % des poids différents de fp32, "
                     f"{100 * p['part_differente_de_rtn']:.1f} % de RTN ; trajectoires identiques à fp32 "
                     f"{tr['trajectoires_identiques_a_fp32']} sur {tr['sur']} ; même issue que fp32 {100 * tr['meme_issue_que_fp32']:.1f} %.")
        L.append(f"- {t} : même issue entre deux graines de fp32 {100 * r[t]['trajectoires']['repere_fp32_entre_deux_graines']:.1f} % ; "
                 f"calibration {r[t]['calibration']['trajectoires_calibration']} trajectoires, aucune parmi les "
                 f"{r[t]['calibration']['trajectoires_evaluation']} des évaluations ; succès recalculé depuis l'état physique : conforme.")
    pv = r["provenance"]
    L += [f"- Pré-enregistrement : {pv['dernier_commit_du_preregistrement']} ; premier fichier écrit {pv['premier_fichier_ecrit']} ; "
          f"commits des fichiers {', '.join(c[:7] for c in pv['commits_des_fichiers'])}.",
          f"- Manifeste : {r['manifeste']['fichiers']} fichiers, empreinte {r['manifeste']['empreinte_du_manifeste'][:16]}."]
    (OUT / "rapport.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
