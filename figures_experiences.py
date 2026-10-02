"""Figures de la page des expériences (papier-corl-2026/figures/expNNN.html), relues depuis les sorties des analyses :
aucun chiffre recopié à la main. Une fonction par expérience ; la page se reconstruit ensuite avec
outils/construire_experiences.py.

Usage (depuis code-exp015) :
    python3 figures_experiences.py exp014 [exp015]
"""
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "resultats"
F = Path(__file__).resolve().parent.parent / "figures"
G, B, V_, N = "#7f8c8d", "#3c6ec8", "#2e9e6b", "#c0392b"
fr = lambda v, n=1: f"{v:.{n}f}".replace(".", ",")


def figure(titre, legende, svg, h):
    return (f'<div class="figure">\n    <p class="titre-fig">{titre}</p>\n    <svg viewBox="0 0 760 {h}" role="img" aria-label="{titre}" '
            f'style="width:100%;height:auto"><g font-size="13" font-family="inherit">{svg}</g></svg>\n    <p class="legende">{legende}</p>\n  </div>\n')


def foret(titre, legende, lignes, xmin, xmax, ticks, marge=None, x0=300, x1=640, h=30):
    """Écarts en points avec leur intervalle à 95 % ; ligne verticale à 0 et, si donnée, à la marge."""
    X = lambda v: x0 + (v - xmin) * (x1 - x0) / (xmax - xmin)
    bas = 40 + h * len(lignes)
    g = []
    for v in ticks:
        g.append(f'<line x1="{X(v):.1f}" y1="30" x2="{X(v):.1f}" y2="{bas}" stroke="#e3e3e3"/>'
                 f'<text x="{X(v):.1f}" y="{bas + 18}" text-anchor="middle" fill="#555" font-size="11">{fr(v, 0)}</text>')
    g.append(f'<line x1="{X(0):.1f}" y1="30" x2="{X(0):.1f}" y2="{bas}" stroke="#555"/>')
    if marge is not None:
        g.append(f'<line x1="{X(marge):.1f}" y1="30" x2="{X(marge):.1f}" y2="{bas}" stroke="{N}" stroke-dasharray="4 3"/>'
                 f'<text x="{X(marge) - 4:.1f}" y="26" text-anchor="end" fill="{N}" font-size="11">marge {fr(marge, 0)}</text>')
    y = 40
    for lab, e, col in lignes:
        g.append(f'<text x="10" y="{y + 13}">{lab}</text>'
                 f'<line x1="{X(e["bas"]):.1f}" y1="{y + 9}" x2="{X(e["haut"]):.1f}" y2="{y + 9}" stroke="{col}" stroke-width="3"/>'
                 f'<circle cx="{X(e["moyenne"]):.1f}" cy="{y + 9}" r="5" fill="{col}"/>'
                 f'<text x="{x1 + 10}" y="{y + 13}" font-size="11" fill="#333">{fr(e["moyenne"])} [{fr(e["bas"])} ; {fr(e["haut"])}]</text>')
        y += h
    H = bas + 40
    g.append(f'<text x="{(x0 + x1) / 2:.0f}" y="{H - 4}" text-anchor="middle" fill="#555" font-size="11">écart de réussite (points)</text>')
    return figure(titre, legende, "".join(g), H)


def exp014():
    r = json.loads((R / "analyse-exp014" / "resultats.json").read_text())
    T, D = r["tests"], r["descriptifs"]
    noms = {"int8": "int8 par canal", "int4c": "int4 par canal", "pal4": "palettes 16 valeurs"}
    tache = {"pusht": "PushT", "tworoom": "TwoRoom"}
    col = lambda x: V_ if x["verdict"] == "soutenue" else (N if x["perte_etablie"] else B)
    lignes = [(f"{tache[x['tache']]}, {noms[x['codage']]}", {k: x["ecart_points"][k] * 100 for k in ("moyenne", "bas", "haut")}, col(x))
              for x in T.values()]
    v = [l[1][k] for l in lignes for k in ("bas", "haut")]
    xmin, xmax = min(-4, 2 * (min(v) // 2)), max(2, 2 * (max(v) // 2 + 1))
    o16 = D["pusht"]["octets_poids_fp16"] / 1e6
    oct_ = ", ".join(f"{noms[c]} {fr(D['pusht']['octets_poids'][c] / 1e6)} Mo" for c in ("int8", "int4c", "pal4"))
    f1 = foret("Codages Core ML sur le Neural Engine : écart de réussite à PyTorch float32 (500 épisodes, 5 graines)",
               f"Vert : non-infériorité à 2 points soutenue (Holm sur 6) ; rouge : perte établie (intervalle entièrement sous 0) ; "
               f"bleu : ni l'un ni l'autre. Fichier des poids : 16 bits {fr(o16)} Mo, {oct_}.",
               lignes, xmin, xmax, list(range(int(xmin), int(xmax) + 1, 2)), marge=-2)
    (F / "exp014.html").write_text(f1)
    return [(l[0], fr(l[1]["moyenne"])) for l in lignes]


def exp015():
    r = json.loads((R / "analyse-exp015" / "resultats.json").read_text())
    D1, D3 = r["D1_ecarts"], r["D3_puce_contre_simulation"]
    noms = {"bl - ne16": ("Neural Engine, blocs seuls (int4c-bl)", V_), "pp - ne16": ("Neural Engine, pred_proj seule (int4c-pp)", V_),
            "sp - ne16": ("Neural Engine, sans pred_proj (int4c-sp)", V_), "int4c - ne16": ("Neural Engine, tout (int4c, EXP-014)", V_),
            "predictor4 - fp32": ("Simulé, blocs seuls (predictor:4)", G), "s_pp - fp32": ("Simulé, pred_proj seule (predproj:4)", G),
            "s_pr - fp32": ("Simulé, tout (predict:4)", G)}
    lignes = [(noms[k][0], D1[k], noms[k][1]) for k in noms if k in D1]
    lignes.append(("pred_proj seule contre int4c (L4)", r["tests"]["L4"]["ecart_points"], B))
    v = [l[1][k] for l in lignes for k in ("bas", "haut")]
    xmin, xmax = min(-4, 2 * (min(v) // 2)), max(2, 2 * (max(v) // 2 + 1))
    f1 = foret("Int4 par canal, portée par portée : écart de réussite à la référence (PushT, 500 épisodes, 5 graines)",
               "Vert : Core ML sur le Neural Engine, contre le 16 bits sur le Neural Engine ; gris : quantification simulée dans "
               "PyTorch, contre float32. Bleu : pred_proj seule contre int4c (test L4 : pas mieux de plus de 2 points). Marge "
               "des tests L1 à L5 : 2 points.", lignes, xmin, xmax, list(range(int(xmin), int(xmax) + 1, 2)), marge=-2)
    l3 = [("pred_proj", D3["pred_proj"], B), ("blocs", D3["blocs"], B), ("tout", D3["tout"], B)]
    v = [l[1][k] for l in l3 for k in ("bas", "haut")]
    f2 = foret("Puce contre simulation : différence entre la perte sur le Neural Engine et la perte simulée, par portée",
               "Différence de différences par épisode ; un intervalle qui contient 0 se lit « compatible », sans équivalence affirmée "
               "(pré-enregistré).", l3, min(-4, 2 * (min(v) // 2)), max(4, 2 * (max(v) // 2 + 1)),
               list(range(int(min(-4, 2 * (min(v) // 2))), int(max(4, 2 * (max(v) // 2 + 1))) + 1, 2)))
    (F / "exp015.html").write_text(f1 + f2)
    return [(l[0], fr(l[1]["moyenne"])) for l in lignes + l3]


def exp017():
    r = json.loads((R / "analyse-exp017" / "resultats.json").read_text())
    T, E = r["tests"], r["descriptifs"]["ecarts"]
    lignes = [("GPTQ sur la puce contre le 16 bits (G1)", T["G1"]["ecart_points"], V_ if T["G1"]["verdict"] == "soutenue" else B),
              ("GPTQ sur la puce contre int4c RTN (G2)", T["G2"]["ecart_points"], V_ if T["G2"]["verdict"] == "soutenue" else B),
              ("GPTQ sur la puce contre PyTorch float32", E["g - fp32"], G), ("GPTQ simulé contre PyTorch float32", E["gs - fp32"], G),
              ("GPTQ sur la puce contre tout sauf pred_proj (RTN)", E["g - sp"], G), ("Puce contre simulation", E["puce contre simulation"], G)]
    v = [l[1][k] for l in lignes for k in ("bas", "haut")]
    xmin, xmax = min(-4, 2 * (min(v) // 2)), max(2, 2 * (max(v) // 2 + 1))
    o = r["descriptifs"]["octets_poids"]
    f1 = foret("Codes de GPTQ écrits dans Core ML, prédicteur sur le Neural Engine (PushT, 500 épisodes, 5 graines)",
               f"Vert : test soutenu ; bleu : test non soutenu ; gris : descriptif. Fichier des poids : GPTQ {fr(o['g'] / 1e6)} Mo, int4c "
               f"{fr(o['int4c'] / 1e6)} Mo, sans pred_proj {fr(o['sp'] / 1e6)} Mo, 16 bits {fr(o['ne16'] / 1e6)} Mo. Lecture : {r['lecture']}.",
               lignes, xmin, xmax, list(range(int(xmin), int(xmax) + 1, 2)), marge=-2)
    (F / "exp017.html").write_text(f1)
    return [(l[0], fr(l[1]["moyenne"])) for l in lignes]


if __name__ == "__main__":
    for e in sys.argv[1:]:
        print(e, globals()[e]())
