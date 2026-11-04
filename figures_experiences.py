"""Figures de la page des expériences (papier-corl-2026/figures/expNNN.html), relues depuis les sorties des analyses :
aucun chiffre recopié à la main. Une fonction par expérience ; la page se reconstruit ensuite avec
outils/construire_experiences.py. Mise en forme reprise de code-exp018/figures_experiences.py.

Usage (depuis code-exp021-dev) :
    python3 figures_experiences.py exp021
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


def foret(titre, legende, lignes, xmin, xmax, ticks, marges=(), x0=300, x1=640, h=30):
    """Écarts en points avec leur intervalle à 95 % ; ligne verticale à 0 et à chaque marge donnée."""
    X = lambda v: x0 + (v - xmin) * (x1 - x0) / (xmax - xmin)
    bas = 40 + h * len(lignes)
    g = []
    for v in ticks:
        g.append(f'<line x1="{X(v):.1f}" y1="30" x2="{X(v):.1f}" y2="{bas}" stroke="#e3e3e3"/>'
                 f'<text x="{X(v):.1f}" y="{bas + 18}" text-anchor="middle" fill="#555" font-size="11">{fr(v, 0)}</text>')
    g.append(f'<line x1="{X(0):.1f}" y1="30" x2="{X(0):.1f}" y2="{bas}" stroke="#555"/>')
    for m in marges:
        ancre, dx = ("end", -4) if m < 0 else ("start", 4)
        g.append(f'<line x1="{X(m):.1f}" y1="30" x2="{X(m):.1f}" y2="{bas}" stroke="{N}" stroke-dasharray="4 3"/>'
                 f'<text x="{X(m) + dx:.1f}" y="26" text-anchor="{ancre}" fill="{N}" font-size="11">marge {"+" if m > 0 else ""}{fr(m, 0)}</text>')
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


def nuage(titre, legende, points, xmin, xmax, xticks, ymin, ymax, yticks, x0=90, x1=700, y0=30, y1=250):
    """Écart à la pleine précision (points, intervalle à 95 %) selon les bits par poids ; un point par version."""
    X = lambda v: x0 + (v - xmin) * (x1 - x0) / (xmax - xmin)
    Y = lambda v: y1 - (v - ymin) * (y1 - y0) / (ymax - ymin)
    g = []
    for v in xticks:
        g.append(f'<line x1="{X(v):.1f}" y1="{y0}" x2="{X(v):.1f}" y2="{y1}" stroke="#e3e3e3"/>'
                 f'<text x="{X(v):.1f}" y="{y1 + 18}" text-anchor="middle" fill="#555" font-size="11">{fr(v)}</text>')
    for v in yticks:
        g.append(f'<line x1="{x0}" y1="{Y(v):.1f}" x2="{x1}" y2="{Y(v):.1f}" stroke="{"#555" if v == 0 else "#e3e3e3"}"/>'
                 f'<text x="{x0 - 8}" y="{Y(v) + 4:.1f}" text-anchor="end" fill="#555" font-size="11">{fr(v, 0)}</text>')
    for lab, x, e, col, dx in points:
        g.append(f'<line x1="{X(x):.1f}" y1="{Y(e["bas"]):.1f}" x2="{X(x):.1f}" y2="{Y(e["haut"]):.1f}" stroke="{col}" stroke-width="3"/>'
                 f'<circle cx="{X(x):.1f}" cy="{Y(e["moyenne"]):.1f}" r="5" fill="{col}"/>'
                 f'<text x="{X(x) + dx:.1f}" y="{Y(e["moyenne"]) + 4:.1f}" text-anchor="{"start" if dx > 0 else "end"}" font-size="12">{lab}</text>')
    H = y1 + 40
    g.append(f'<text x="{(x0 + x1) / 2:.0f}" y="{H - 4}" text-anchor="middle" fill="#555" font-size="11">bits par poids des couches linéaires du prédicteur</text>'
             f'<text x="14" y="{(y0 + y1) / 2:.0f}" text-anchor="middle" fill="#555" font-size="11" transform="rotate(-90 14 {(y0 + y1) / 2:.0f})">écart à la pleine précision (points)</text>')
    return figure(titre, legende, "".join(g), H)


def exp021():
    r = json.loads((R / "exp021-analyse" / "resultats.json").read_text())
    T, D = r["tests"], r["descriptifs"]
    E_, bits, reu = D["ecarts"], D["bits_par_poids"], D["reussite"]
    noms = {"A": "ternaire, 1 % gardé, réglé", "B": "3 bits, réglé", "B0": "3 bits, non réglé",
            "C": "ternaire, 1 % gardé, non réglé", "D": "ternaire sans poids gardés, réglé", "E": "ternaire, 3,1 % gardé, réglé"}
    col = {"A": V_, "B": V_, "B0": G, "C": G, "D": V_, "E": V_}
    dx = {"A": 9, "B": -9, "B0": 9, "C": 9, "D": 9, "E": 9}
    points = [(f"{k} ({fr(100 * reu[k])} %)", bits[k], E_[f"{k} - fp32"], col[k], dx[k]) for k in noms]
    v = [p[2][c] for p in points for c in ("bas", "haut")]
    ymin, ymax = 2 * (min(v) // 2), max(2, 2 * (max(v) // 2 + 1))
    f1 = nuage(f"Réussite selon la mémoire : écart à la pleine précision ({fr(100 * reu['fp32'])} %), PushT, 1 000 épisodes neufs, 4 graines",
               "Point : écart moyen sur les 4 graines ; trait : intervalle percentile à 95 % (bootstrap des épisodes, descriptif). "
               "Entre parenthèses : réussite. Vert : réglé ; gris : non réglé. "
               + "B et B0 ont la même mémoire (3,06 bits) ; B0 est décalé à droite pour la lisibilité. " + " ; ".join(f"{k} : {noms[k]}" for k in noms) + ".",
               [p if p[0][:2] != "B0" else (p[0], p[1] + 0.03, *p[2:]) for p in points],
               2.0, 3.2, [2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2], ymin, ymax, list(range(int(ymin), int(ymax) + 1, 2)))
    lab = {"H1": "H1 : A contre C (effet du réglage)", "H2": "H2 : A contre D (effet des poids gardés)", "H3": "H3 : A contre B (barre 3 bits)"}
    lignes = [(lab[t], T[t]["ecart_points_descriptif"], V_ if T[t]["verdict_ecrit"] == "soutenue" else B) for t in ("H1", "H2", "H3")]
    v = [l[1][k] for l in lignes for k in ("bas", "haut")]
    xmin, xmax = min(-4, 2 * (min(v) // 2)), max(4, 2 * (max(v) // 2 + 1))
    bornes = " ; ".join(f"{t} {fr(T[t]['borne_points'], 2)}" for t in ("H1", "H2", "H3"))
    f2 = foret("Les trois tests pré-enregistrés : écart de A à sa référence (points, 1 000 épisodes, 4 graines)",
               "Vert : soutenue ; bleu : non soutenue (Holm sur 3, alpha 5 %). H1 et H2 demandent un écart au-dessus de +2, H3 "
               f"un écart au-dessus de -2. Trait : intervalle percentile à 95 % (descriptif) ; bornes inférieures des tests au niveau de Holm : {bornes}.",
               lignes, xmin, xmax, list(range(int(xmin), int(xmax) + 1, 2)), marges=(-2, 2))
    (F / "exp021.html").write_text(f1 + f2)
    return [(p[0], fr(p[2]["moyenne"])) for p in points] + [(l[0], fr(l[1]["moyenne"])) for l in lignes]


if __name__ == "__main__":
    for e in sys.argv[1:]:
        print(e, globals()[e]())
