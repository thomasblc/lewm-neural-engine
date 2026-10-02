"""Énergie par plan rapportée à la batterie (exploratoire, non pré-enregistrée comme résultat ; demandée par Thomas le
2 octobre après l'échec de la validation des compteurs IOReport). Deux estimations, toutes deux approchées, à partir de
deux sessions différentes (EXP-013, le 1er octobre ; validation, le 2 octobre) :
- A : énergie de la puce par plan d'EXP-013 (au-dessus du repos, fenêtres de planification) multipliée par le rapport
  batterie sur puce des puissances au-dessus du repos, mesuré moteur par moteur pendant la validation ;
- B : puissance tirée de la batterie au-dessus du repos pendant la validation (moyenne du segment, chargement exclu,
  pas seulement les fenêtres de planification) multipliée par le temps par plan d'EXP-013.
PyTorch float16 et Core ML sur le Neural Engine (encodeur sur le GPU) n'ont pas de segment de validation : non estimés.

Usage (depuis code-exp015) :
    python3 energie_batterie.py ../resultats/energie-batterie-2026-10-02
"""
import json
import subprocess
import sys
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "resultats"
MOTEURS = ("torch", "coreml-gpu", "ne-complet")


def main(sortie):
    c13 = json.loads((R / "analyse-exp013" / "resultats.json").read_text())["configurations"]
    v = json.loads((R / "validation-energie-2026-10-02" / "resultats.json").read_text())
    assert not v["essai"] and v["segments_valides"] and v["ecarts_positifs"], "validation non conforme"
    D = v["au_dessus_du_repos"]
    out = {"sources": ["analyse-exp013/resultats.json", "validation-energie-2026-10-02/resultats.json"],
           "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(), "moteurs": {}}
    for m in MOTEURS:
        e13, t13 = c13[m]["energie_solve_puce_j"]["mediane"], c13[m]["temps_par_plan_ms"]["mediane"] / 1000
        rapport = D[m]["P_b"] / D[m]["P_p"]
        out["moteurs"][m] = {"energie_puce_par_plan_exp013_j": e13, "temps_par_plan_exp013_s": t13,
                             "rapport_batterie_sur_puce": rapport, "puissance_batterie_au_dessus_du_repos_w": D[m]["P_b"],
                             "A_j_par_plan": e13 * rapport, "B_j_par_plan": D[m]["P_b"] * t13}
    for k in ("A_j_par_plan", "B_j_par_plan"):
        ne = out["moteurs"]["ne-complet"][k]
        out[f"rapports_{k[0]}"] = {f"{m} / ne-complet": out["moteurs"][m][k] / ne for m in ("torch", "coreml-gpu")}
    Path(sortie).mkdir(parents=True, exist_ok=True)
    (Path(sortie) / "resultats.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    for m, x in out["moteurs"].items():
        print(f"{m:11s} A {x['A_j_par_plan']:5.1f} J  B {x['B_j_par_plan']:5.1f} J  (puce EXP-013 {x['energie_puce_par_plan_exp013_j']:.1f} J, "
              f"rapport batterie/puce {x['rapport_batterie_sur_puce']:.3f}, {1000 * x['temps_par_plan_exp013_s']:.0f} ms)")
    print("rapports A", {k: round(x, 2) for k, x in out["rapports_A"].items()}, "B", {k: round(x, 2) for k, x in out["rapports_B"].items()})


if __name__ == "__main__":
    main(sys.argv[1])
