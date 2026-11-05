"""Figures de la page des expériences (papier-corl-2026/figures/expNNN.html), relues depuis les sorties des analyses :
aucun chiffre recopié à la main. Une fonction par expérience ; la page se reconstruit ensuite avec
outils/construire_experiences.py. Mise en forme reprise de code-exp018/figures_experiences.py.

Usage (depuis code-exp021-dev) :
    python3 figures_experiences.py exp018 exp019 exp020 exp021
"""
import json
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "resultats"
F = Path(__file__).resolve().parent.parent / "figures"
G, B, V_, N = "#7f8c8d", "#3c6ec8", "#2e9e6b", "#c0392b"
# arrondi à la moitié loin de zéro après avoir effacé le bruit du flottant (-7,25 donne -7,3, comme outils/chiffres_papier.py)
fr = lambda v, n=1: str(Decimal(repr(round(v, 9))).quantize(Decimal(1).scaleb(-n), ROUND_HALF_UP)).replace(".", ",")


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


def barres(titre, legende, lignes, ref=None, x0=330, x1=640, h=26):
    """Réussites en % ; trait fin entre le plus bas et le plus haut si donnés ; ligne verticale à la référence.
    Une seule couleur (gris pour la référence) : la longueur de la barre porte le message, pas une couleur choisie à la main."""
    X = lambda v: x0 + v * (x1 - x0) / 100
    bas = 40 + h * len(lignes)
    g = []
    for v in range(0, 101, 20):
        g.append(f'<line x1="{X(v):.1f}" y1="30" x2="{X(v):.1f}" y2="{bas}" stroke="#e3e3e3"/>'
                 f'<text x="{X(v):.1f}" y="{bas + 18}" text-anchor="middle" fill="#555" font-size="11">{v}</text>')
    if ref is not None:
        g.append(f'<line x1="{X(ref[1]):.1f}" y1="30" x2="{X(ref[1]):.1f}" y2="{bas}" stroke="{N}" stroke-dasharray="4 3"/>'
                 f'<text x="{X(ref[1]) - 4:.1f}" y="26" text-anchor="end" fill="{N}" font-size="11">{ref[0]}</text>')
    y = 40
    for lab, v, mini, maxi, col in lignes:
        g.append(f'<text x="10" y="{y + 13}">{lab}</text>'
                 f'<rect x="{x0}" y="{y + 3}" width="{X(v) - x0:.1f}" height="{h - 10}" fill="{col if col == G else B}" opacity="0.85"/>'
                 f'<text x="{x1 + 10}" y="{y + 13}" font-size="11" fill="#333">{fr(v)} %'
                 + (f' ({fr(mini)} à {fr(maxi)})' if mini is not None else '') + '</text>')
        if mini is not None:
            g.append(f'<line x1="{X(mini):.1f}" y1="{y + 11}" x2="{X(maxi):.1f}" y2="{y + 11}" stroke="#222" stroke-width="1.5"/>')
        y += h
    H = bas + 40
    g.append(f'<text x="{(x0 + x1) / 2:.0f}" y="{H - 4}" text-anchor="middle" fill="#555" font-size="11">réussite (%)</text>')
    return figure(titre, legende, "".join(g), H)


def exp018():
    r = json.loads((R / "analyse-exp018" / "resultats.json").read_text())
    T, reu = r["tests"], r["descriptifs"]["reussite"]
    lab = {"T1": "T1 : int4c contre float32", "T2": "T2 : tout sauf pred_proj contre float32",
           "T3": "T3 : codes de GPTQ contre float32", "T4": "T4 : codes de GPTQ contre int4c"}
    lignes = [(lab[t], T[t]["ecart_points"], V_ if T[t]["verdict_ecrit"] == "soutenue" else B) for t in ("T1", "T2", "T3", "T4")]
    v = [l[1][k] for l in lignes for k in ("bas", "haut")]
    xmin, xmax = min(-4, 2 * (min(v) // 2)), max(4, 2 * (max(v) // 2 + 1))
    f1 = foret("Int4 sur le Neural Engine et ses deux réparations, sur 2 000 épisodes neufs (PushT, 5 graines)",
               "Vert : soutenue ; bleu : non soutenue (Holm sur 4). T1 demande une perte de plus de 2 points, T2 et T3 une perte de "
               "moins de 2 points, T4 un gain de plus de 2 points. Réussite : float32 " + fr(100 * reu["fp32"]) + " %, int4c "
               + fr(100 * reu["int4c"]) + " %, tout sauf pred_proj " + fr(100 * reu["sp"]) + " %, codes de GPTQ " + fr(100 * reu["g"])
               + " %. Trait : intervalle à 95 %.", lignes, xmin, xmax, list(range(int(xmin), int(xmax) + 1, 2)), marges=(-2, 2))
    (F / "exp018.html").write_text(f1)
    return [(l[0], fr(l[1]["moyenne"])) for l in lignes]


def exp019():
    r = json.loads((R / "exp019-pilote" / "analyse.json").read_text())
    reu, comp = r["reussite"], r["comparaisons"]
    noms = [("fp32", "pleine précision", G), ("predictor:3", "3 bits (RTN)", B), ("predictor:2:clip:gptq", "ternaire GPTQ", N),
            ("predictor:2:clip:gptq:k1:obs", "ternaire GPTQ, 1 % gardé", B), ("predictor:2:clip:gptq:k1:obs:aff", "la même, réglée", V_)]
    lignes = [(lab, 100 * reu[k], None, None, col) for k, lab, col in noms]
    c = lambda k: comp[f"{k} contre fp32"]
    detail = " ; ".join(f"{lab} {c(k)['seul_reussi']} contre {c(k)['ref_seule']}" for k, lab, _ in noms[1:])
    f1 = barres(f"Pilote : prédicteur ternaire avec 1 % de poids gardés, puis réglé (PushT, {r['positions']} épisodes, graine {r['graine']})",
                "Exploratoire, non pré-enregistré : un écart de moins de 10 points n'est pas établi sur 100 épisodes. Épisodes réussis "
                f"par la version seule contre par la pleine précision seule : {detail}.", lignes,
                ref=("pleine précision", 100 * reu["fp32"]))
    (F / "exp019.html").write_text(f1)
    return [(l[0], fr(l[1])) for l in lignes]


def exp020():
    c96 = R / "exp020-banc" / "c96"
    t = [json.loads((c96 / f"test-final-plan{p}.json").read_text())["resultats"] for p in (11, 12)]
    noms = [("fp32.pt", "float32", G), ("rtn2.pt", "ternaire RTN", N), ("gptq2.pt", "ternaire GPTQ", N),
            ("gptq2_k1.pt", "ternaire GPTQ, 1 % gardé", N), ("gptq3.pt", "GPTQ 3 bits", B),
            ("gptq2_k1_affine.pt", "1 % gardé, affiné par gradient", B), ("essai-opus-1/final.pt", "1 % gardé, méthode d'Opus", V_)]
    cle = lambda f: f"../resultats/exp020-banc/c96/{f}"
    l1 = []
    for f, lab, col in noms:
        v = [100 * x[cle(f)]["reussite"] for x in t]
        l1.append((f"{lab}, {fr(t[0][cle(f)]['bits_par_poids'], 2)} bits", sum(v) / 2, min(v), max(v), col))
    f1 = barres("Banc jouet, premier test : un LLM (Opus 5.5) répare le ternaire (300 épisodes neufs, 2 graines de planification)",
                "Barre : moyenne des deux graines de planification ; trait et parenthèses : plus bas et plus haut. Ici l'affinage par gradient était nourri de transitions tirées au hasard.", l1,
                ref=("float32", l1[0][1]))
    lignes = [json.loads(x) for x in (c96 / "references-2" / "resultats.jsonl").read_text().splitlines() if x.strip()]
    lignes += [json.loads(x) for x in (c96 / "references-2" / "affinage_regle.jsonl").read_text().splitlines() if x.strip()]
    m2 = [("gptq_k1", "GPTQ ternaire, 1 % gardé", N), ("dfq", "DFQ puis GPTQ", N), ("adaquant", "AdaQuant", N),
          ("qat", "QAT, réseau entier", B), ("affine", "affinage, 6 000 pas", B),
          ("affine_p20k_lr1e-03_h8", "le même, perte sur 8 pas", N),
          ("affine_p20k_lr1e-03_h0", "affinage réglé (20 000 pas, 1e-3)", V_), ("opus", "recette d'Opus, sans LLM", V_)]
    l2 = []
    for m, lab, col in m2:
        v = [100 * d[f"test_plan{p}"]["reussite"] for d in lignes if d["methode"] == m for p in (11, 12) if f"test_plan{p}" in d]
        assert len(v) == 6, (m, len(v))
        l2.append((lab, sum(v) / len(v), min(v), max(v), col))
    f2 = barres("Banc jouet, amendements 1 et 2 : les mêmes données pour toutes les méthodes (2,8 bits par poids)",
                "Barre : moyenne sur 3 graines de données et 2 graines de planification (300 épisodes neufs chacune) ; trait et "
                "parenthèses : plus bas et plus haut des 6. Bien réglé et nourri des mêmes données, l'affinage par gradient égale la "
                "recette d'Opus ; une perte sur 8 pas fait s'effondrer la planification.", l2, ref=("float32", l1[0][1]))
    (F / "exp020.html").write_text(f1 + f2)
    return [(l[0], fr(l[1])) for l in l1 + l2]


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
