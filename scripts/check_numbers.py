"""Check of published numbers against the public data, standard library only.

Recomputes, from data/outcomes, the success rate of every version in every run folder, and the paired mean differences
behind the tests of E18 (chip runs on 2,000 held-out episodes) and E21 (ternary predictor, simulation), then compares
them with the outputs of the frozen analyses in data/analyses. This script was written after the analyses, for readers;
it does not replace them (they also run the tests, bounds and controls, from the full result files).

Usage, from the repository root:
    python3 scripts/check_numbers.py
"""
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"


def outcomes(folder):
    """{(weights, seed, position): success} for one run folder."""
    with gzip.open(DATA / "outcomes" / f"{folder}.csv.gz", "rt", newline="") as f:
        return {(r["weights"], int(r["seed"]), int(r["position"])): int(r["success"]) for r in csv.DictReader(f)}


def rates():
    out = {}
    for f in sorted((DATA / "outcomes").glob("*.csv.gz")):
        acc = defaultdict(lambda: [0, 0])
        with gzip.open(f, "rt", newline="") as h:
            for r in csv.DictReader(h):
                acc[(r["engine"], r["weights"])][0] += int(r["success"])
                acc[(r["engine"], r["weights"])][1] += 1
        out[f.name[:-7]] = {k: (100 * s / n, n) for k, (s, n) in acc.items()}
    return out


def paired_mean(arm, ref):
    """Mean paired difference in points over the (seed, position) pairs present in both."""
    keys = arm.keys() & ref.keys()
    assert keys, "no common runs"
    return 100 * sum(arm[k] - ref[k] for k in keys) / len(keys), len(keys)


def select(d, weights, n_positions=None):
    return {(s, p): v for (w, s, p), v in d.items() if w == weights and (n_positions is None or p < n_positions)}


def check(name, mine, published, tol=1e-9):
    ok = abs(mine - published) < tol
    print(f"  {'OK  ' if ok else 'DIFF'} {name}: {mine:+.4f} (published {published:+.4f})")
    return ok


def main():
    allok = True
    print("Success rate by run folder and version (%, runs):")
    for folder, r in rates().items():
        print(f"  {folder}: " + "; ".join(f"{e}/{w} {p:.2f} ({n})" for (e, w), (p, n) in sorted(r.items())))

    print("E18, paired differences in points (chip on 2,000 held-out episodes, seeds 52 to 56):")
    e18 = json.loads((DATA / "analyses" / "analyse-exp018" / "resultats.json").read_text())["tests"]
    fp32 = select(outcomes("exp005"), "fp32")
    int4c = select(outcomes("exp018-int4c-pusht"), "int4c")
    sp = select(outcomes("exp018-int4c-sp-pusht"), "int4c-sp")
    g = select(outcomes("exp018-gptq-pusht"), "int4c-gptq")
    arms = {"int4c": int4c, "sp": sp, "g": g, "fp32": fp32}
    for t, x in e18.items():
        m, n = paired_mean(arms[x["bras"]], arms[x["reference"]])
        allok &= check(f"{t} {x['bras']} vs {x['reference']} ({n} pairs)", m, x["ecart_points"]["moyenne"])

    print("E21, paired differences in points (first 1,000 held-out episodes, seeds 52 to 55, simulation):")
    e21 = json.loads((DATA / "analyses" / "exp021-analyse" / "resultats.json").read_text())["tests"]
    d21 = outcomes("exp021-pusht")
    spec = {"A": "predictor:2:clip:gptq:k1:obs:aff", "B": "predictor:3:clip:gptq:aff", "C": "predictor:2:clip:gptq:k1:obs",
            "D": "predictor:2:clip:gptq:aff"}
    for t, x in e21.items():
        m, n = paired_mean(select(d21, spec[x["bras"]]), select(d21, spec[x["reference"]]))
        allok &= check(f"{t} {x['bras']} vs {x['reference']} ({n} pairs)", m, x["ecart_points_descriptif"]["moyenne"])
    fp21 = {k: v for k, v in fp32.items() if k[0] <= 55 and k[1] < 1000}
    m, n = paired_mean(select(d21, spec["A"]), fp21)
    a = json.loads((DATA / "analyses" / "exp021-analyse" / "resultats.json").read_text())["descriptifs"]["ecarts"]["A - fp32"]
    allok &= check(f"A vs float32 of E5 ({n} pairs)", m, a["moyenne"])

    print("All checks match." if allok else "Some checks differ.")
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
