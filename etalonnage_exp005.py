"""Étalonnage d'H6 avant le lancement d'EXP-005 : pseudo-expériences de 2 000 épisodes tirées des vraies données
épisode par épisode d'EXP-003 (fp32 et all:6, graines 42 à 46), passées dans la vraie fonction stat_h6.

Mesure les fausses alertes (sans effet sur la pose ; réussite exactement à -2 et à +2 points, pour chaque borne et
pour H6 entière) et la puissance (écart de pose égal à celui d'EXP-003, puis à la moitié). Les chiffres sont ceux
cités dans PREREG-EXP005.md. La puissance de la progression selon les bits (intervalle à 95 % au-dessus de 0) est
calculée par approximation normale à la fin.

Usage :
    python etalonnage_exp005.py        # environ 20 s sur CPU
"""
import numpy as np
from scipy import stats

import analyse_exp003 as A
import analyse_exp005 as A5

R = A.ICI.parent / "resultats" / "exp003"


def donnees_exp003():
    def tab(v):
        S, M3 = [], []
        for g in A.GA:
            d = dict(np.load(R / f"{v.replace(':', '-')}_s{g}.npz"))
            _, mT, *_ = A.controler_course(d, v)
            S.append(d["success"].astype(int))
            M3.append(mT)
        return np.stack(S), np.stack(M3)
    Sf, Mf = tab("fp32")
    S6, M6 = tab("all:6")
    bouge = A.depart({("fp32", A.GA[0]): dict(np.load(R / "fp32_s42.npz"))})["bouge"]
    return S6.sum(0) - Sf.sum(0), (A.lg(M6).mean(0) - A.lg(Mf).mean(0))[bouge]


def ecart_type_d68():
    lg = {v: A.lg(np.stack([A.controler_course(dict(np.load(R / f"{v.replace(':', '-')}_s{g}.npz")), v)[1]
                            for g in A.GA])).mean(0) for v in ("all:6", "all:8")}
    bouge = A.depart({("fp32", A.GA[0]): dict(np.load(R / "fp32_s42.npz"))})["bouge"]
    return float((lg["all:6"] - lg["all:8"])[bouge].std(ddof=1))


def decaler(x, cible, rng):
    """Population d'écarts x (entiers de -5 à 5) déplacée d'unités pour que sa moyenne vaille `cible`."""
    x = x.copy()
    k = int(round(abs(x.mean() - cible) * len(x)))
    sens = -1 if cible < x.mean() else +1
    ok = np.where((x > -5) if sens < 0 else (x < 5))[0]
    x[rng.choice(ok, size=k, replace=False)] += sens
    return x


def main():
    x_obs, w_obs = donnees_exp003()
    print(f"EXP-003 : écart de réussite {100 * x_obs.mean() / 5:+.2f} points, w moyen {w_obs.mean():+.4f}, "
          f"écart-type {w_obs.std(ddof=1):.3f}")
    inv, nb = A.Inversion(n=A5.N5), 1333
    # 1. pose : fausses alertes sans effet, puissance à l'écart d'EXP-003 et à la moitié (1 000 tirages chacun)
    rng = np.random.default_rng(2026)
    wc = w_obs - w_obs.mean()
    x_bord = decaler(x_obs, -0.10, rng)

    def tirages(xpop, wpop, n=1000):
        return [A5.stat_h6(xpop[rng.integers(0, len(xpop), A5.N5)], wpop[rng.integers(0, len(wpop), nb)], inv)
                for _ in range(n)]

    with np.errstate(all="ignore"):
        t = tirages(x_obs, wc)
        print(f"(b) sans effet : fausses alertes {100 * np.mean([s['p_b'] < 0.05 for s in t]):.1f} %")
        t = tirages(x_bord, wc + 0.03)          # H6 entière : réussite exactement à -2 points, pose dégradée de +0,030
        fa = np.mean([s["p"] < 0.05 for s in t])
        print(f"H6 à la borne, réussite à -2 points et pose +0,030 : établie à tort {100 * fa:.1f} % "
              f"(erreur Monte-Carlo {100 * np.sqrt(0.05 * 0.95 / len(t)):.1f} point)")
        for eff in (0.0305, 0.0153):
            t = tirages(x_obs, wc + eff)
            print(f"puissance, écart de pose +{eff:.3f} : H6 établie {100 * np.mean([s['p'] < 0.05 for s in t]):.1f} %")
    # 2. réussite : fausses alertes des deux bornes du TOST, exactement à -2 et +2 points (4 000 tirages chacun)
    rng = np.random.default_rng(7)
    for nom, cible in (("borne basse, -2 points", -0.10), ("borne haute, +2 points", +0.10)):
        xp = decaler(x_obs, cible, rng)
        X = xp[rng.integers(0, len(xp), (4000, A5.N5))]
        with np.errstate(all="ignore"):
            p = inv.p_lot(10 * X + 1, "plus") if cible < 0 else inv.p_lot(10 * X - 1, "moins")
        d = X / 5
        m, se = d.mean(1), d.std(1, ddof=1) / np.sqrt(A5.N5)
        pt = stats.t.sf((m + 0.02) / se, A5.N5 - 1) if cible < 0 else stats.t.cdf((m - 0.02) / se, A5.N5 - 1)
        print(f"(a) {nom} : fausses alertes {100 * np.mean(p < 0.05):.2f} % (TOST t : {100 * np.mean(pt < 0.05):.2f} %)")
    # 3. progression 8 puis 6 bits : écart-type de d = log M3 à 6 bits moins à 8 bits, strate d'EXP-003 ; décision par
    #    l'intervalle bilatéral à 95 % entièrement au-dessus de 0, soit un test unilatéral à 2,5 %
    sd = ecart_type_d68()
    se = sd / np.sqrt(nb)
    print(f"progression : écart-type de d {sd:.3f} ; puissance "
          + ", ".join(f"{100 * stats.norm.cdf(eff / se - stats.norm.ppf(0.975)):.0f} % pour {eff}" for eff in (0.025, 0.010)))


if __name__ == "__main__":
    main()
