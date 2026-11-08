"""Round 2: five combinations declared after round 1 (so they are data-informed and counted), a nested
strategy selection over the pre-planned round-1 menu (the honest estimate of 'pick a strategy by CV'),
and paired comparisons of the leading variants against v1.

    uv run --project /Users/aryan/Projects/hacktheburgh/keyboard-fatigue-nobaseline/analysis python run_round2.py

Writes results_round2.csv, nested_choices.json and paired_vs_v1.csv in this folder.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import common as C
from common import OrdinalLogit, lin, mean_of, pairwise, prep, tuned, within_ridge
from run_variants import ORD_A, variants

HERE = Path(__file__).parent
COLS = ["lopo:MAE", "lopo:skill", "lopo:within_concordance", "rolling:MAE", "rolling:skill",
        "rolling:within_concordance"]


def ordinal_aug(alpha, a=1.0):
    """Ordinal model on kss_delta with the baseline check-ins as extra training rows."""
    def fp(train, test):
        rows = pd.concat([train, C.aug_rows(train, test)])
        f = prep(rows, C.TYPING)
        y = rows.kss.to_numpy(float) - a * rows.kss_base.to_numpy(float)
        return OrdinalLogit(alpha).fit(f(rows), y).predict(f(test)) + a * test.kss_base.to_numpy(float)
    return fp


def within_aug(a):
    """Within-person ridge with the baseline check-ins as extra rows (they add within-person contrast)."""
    def fp(train, test):
        rows = pd.concat([train, C.aug_rows(train, test)])
        return within_ridge(a=a)(rows, test)
    return fp


def round2():
    return {
        "R2a ordinal kss_delta + baseline rows, alpha tuned": tuned(lambda al: ordinal_aug(al), ORD_A),
        "R2b mean(H1 v1+baseline rows, v3)": mean_of(lin(a=1.0, aug=True), C.H.v3),
        "R2c within-person ridge + baseline rows, anch": within_aug(1.0),
        "R2d within-person ridge + baseline rows, unanch": within_aug(0.0),
        "R2e pairwise + baseline rows, unanch": pairwise(a=0.0, aug=True),
    }


def row(res):
    return {f"{s}:{k}": v for s, r in res.items() for k, v in r.items() if k != "n"}


def nested(d, menu):
    """Inside each outer training set, score every menu strategy by inner leave-one-person-out
    (MSE and within-person concordance), pick the best, and predict the outer test rows with it."""
    out, chosen = {}, {}
    for crit in ("mse", "conc"):
        out[crit] = {}
        chosen[crit] = {}
    for scheme in ("lopo", "rolling"):
        preds = {c: np.full(len(d), np.nan) for c in ("mse", "conc")}
        base = np.full(len(d), np.nan)
        for c in ("mse", "conc"):
            chosen[c][scheme] = []
        for tr_mask, te_mask in C.H.splits(d, scheme):
            train, test = d[tr_mask], d[te_mask]
            y = train.kss.to_numpy(float)
            splits = C.inner_splits(train)
            score = {}
            for name, fn in menu.items():
                p = np.full(len(train), np.nan)
                for itr, ite in splits:
                    p[ite] = fn(train[itr], train[ite])
                p = np.clip(p, 1, 9)
                score[name] = (np.mean((p - y) ** 2), -C.concordance(train, p))
            for k, c in enumerate(("mse", "conc")):
                best = min(score, key=lambda n: (score[n][k], list(menu).index(n)))
                chosen[c][scheme].append(best)
                preds[c][te_mask] = np.clip(menu[best](train, test), 1, 9)
            base[te_mask] = y.mean()
        for c in ("mse", "conc"):
            out[c][scheme] = C.H.metrics(d, preds[c], base)
    return out, chosen


def paired(d, fn_a, fn_b, scheme, n=20000, seed=1):
    """Row-level sign-flip test of the squared-error difference (b - a); rows are not independent,
    so treat p as a rough guide. Also counts people for whom a has the lower MAE."""
    pa, _ = C.H.predict(d, fn_a, scheme)
    pb, _ = C.H.predict(d, fn_b, scheme)
    m = ~np.isnan(pa)
    y = d.kss.to_numpy(float)[m]
    diff = (pb[m] - y) ** 2 - (pa[m] - y) ** 2  # positive: a better
    rng = np.random.default_rng(seed)
    flips = rng.choice([-1, 1], size=(n, len(diff)))
    null = (flips * diff).mean(1)
    p = (np.sum(null >= diff.mean()) + 1) / (n + 1)
    pids = d.pid.to_numpy()[m]
    better = sum(np.mean(np.abs(pa[m][pids == q] - y[pids == q])) < np.mean(np.abs(pb[m][pids == q] - y[pids == q]))
                 for q in np.unique(pids))
    return {"mean_sq_gain": float(diff.mean()), "p_one_sided": float(p), "people_better": int(better),
            "people": int(len(np.unique(pids)))}


def main():
    t, d, B = C.load()
    rows = {}
    for name, fn in round2().items():
        t0 = time.time()
        rows[name] = row(C.H.evaluate(d, fn))
        print(f"{name:55s} {time.time() - t0:5.1f}s", flush=True)

    v1 = variants()
    menu = {"v1": C.H.BASELINES["v1"], "v3": C.H.v3} | {k: fn for k, (fn, g, ok) in v1.items() if ok}
    t0 = time.time()
    res, chosen = nested(d, menu)
    print(f"nested selection over {len(menu)} strategies: {time.time() - t0:.0f}s", flush=True)
    rows["N1 nested selection by inner MSE"] = row(res["mse"])
    rows["N2 nested selection by inner concordance"] = row(res["conc"])
    (HERE / "nested_choices.json").write_text(json.dumps(chosen, indent=1))

    table = pd.DataFrame(rows).T
    table.to_csv(HERE / "results_round2.csv")
    pd.set_option("display.width", 250)
    print(table[COLS].astype(float).round(3).to_string())
    for c in ("mse", "conc"):
        for s in ("lopo", "rolling"):
            print(c, s, pd.Series(chosen[c][s]).value_counts().to_dict())

    # paired comparisons against v1 for the variants that beat v1 on MAE in both schemes (+ two LOPO-only ones)
    r1 = pd.read_csv(HERE / "results_round1.csv", index_col=0)
    allres = pd.concat([r1[COLS], table[COLS]]).astype(float)
    v1row = allres.loc["v1"]
    lead = allres[(allres["lopo:MAE"] < v1row["lopo:MAE"]) & (allres["rolling:MAE"] < v1row["rolling:MAE"])].index
    lead = [x for x in lead if x not in ("v1", "N1 nested selection by inner MSE", "N2 nested selection by inner concordance")]
    print("variants beating v1 on MAE in both schemes:", lead)
    fns = {k: fn for k, (fn, g, ok) in variants().items()} | round2()
    extra = ["B7 within-person ridge, unanchored", "B3 pairwise C=1, unanchored"]
    prow = {}
    for name in list(dict.fromkeys(lead + extra)):
        for s in ("lopo", "rolling"):
            r = paired(d, fns[name], C.H.BASELINES["v1"], s)
            prow[(name, s)] = r
    ptab = pd.DataFrame(prow).T
    ptab.to_csv(HERE / "paired_vs_v1.csv")
    print(ptab.to_string())
    print(f"round-2 variants: {len(round2())}, nested: 2")


if __name__ == "__main__":
    main()
