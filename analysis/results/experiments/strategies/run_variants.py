"""Round 1: every pre-planned training-strategy variant under both schemes, next to the baselines.

    uv run --project /Users/aryan/Projects/hacktheburgh/keyboard-fatigue-nobaseline/analysis python run_variants.py

Writes results_round1.csv and tuned_choices_round1.json in this folder.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import common as C
from common import (TYPING, BayesianRidge, ElasticNet, Lasso, Ridge, composite, lin, mean_of, ordered_on_score,
                    ordinal, pairwise, person_offset, pipe, pls, practice, topk, tuned, within_ridge)

HERE = Path(__file__).parent
LASSO = list(np.logspace(-2, 0.3, 10))
ENET = [(al, l1) for l1 in (0.2, 0.5, 0.8) for al in np.logspace(-2, 0.5, 8)]
RIDGE_A = list(C.ALPHAS)
ORD_A = [1.0, 0.3, 3.0, 10.0]


def variants():
    """name -> (fit_predict, group, deployable). The order is the order of the table."""
    v = {}
    # A. anchoring
    v["A1 unanchored v1 (a=0)"] = (lin(a=0.0), "anchoring", True)
    v["A2 partial anchor, a tuned"] = (tuned(lambda a: lin(a=a), [0.0, 0.25, 0.5, 0.75, 1.0]), "anchoring", True)
    v["A3 kss_base as ridge feature"] = (lin(TYPING + ["kss_base"], a=0.0), "anchoring", True)
    v["A4 half anchor (a=0.5)"] = (lin(a=0.5), "anchoring", True)
    # B. pairwise within-person ranking (and its linear cousin)
    v["B1 pairwise C=1, anchored"] = (pairwise(C=1.0, a=1.0), "pairwise", True)
    v["B2 pairwise C tuned (conc), anchored"] = (
        tuned(lambda c: pairwise(C=c, a=1.0), [1.0, 0.1, 0.01, 10.0], criterion="conc"), "pairwise", True)
    v["B3 pairwise C=1, unanchored"] = (pairwise(C=1.0, a=0.0), "pairwise", True)
    v["B4 pairwise |dKSS|-weighted, anchored"] = (pairwise(C=1.0, a=1.0, weight=True), "pairwise", True)
    v["B5 pairwise + baseline rows, anchored"] = (pairwise(C=1.0, a=1.0, aug=True), "pairwise", True)
    v["B6 within-person ridge, anchored"] = (within_ridge(a=1.0), "pairwise", True)
    v["B7 within-person ridge, unanchored"] = (within_ridge(a=0.0), "pairwise", True)
    # C. ordinal
    v["C1 ordinal on KSS, alpha tuned"] = (tuned(lambda al: ordinal(alpha=al, a=0.0), ORD_A), "ordinal", True)
    v["C2 ordinal on kss_delta, alpha tuned"] = (tuned(lambda al: ordinal(alpha=al, a=1.0), ORD_A), "ordinal", True)
    v["C3 OrderedModel on OOF ridge score"] = (ordered_on_score(), "ordinal", True)
    # D. stronger regularization / fewer features, anchored (a=1) and unanchored (a=0)
    for a, tag in ((1.0, "anch"), (0.0, "unanch")):
        v[f"D1 lasso tuned, {tag}"] = (
            tuned(lambda al, a=a: lin(a=a, factory=pipe(Lasso(alpha=al, max_iter=20000))), LASSO), "regularize", True)
        v[f"D2 elastic net tuned, {tag}"] = (
            tuned(lambda g, a=a: lin(a=a, factory=pipe(ElasticNet(alpha=g[0], l1_ratio=g[1], max_iter=20000))), ENET),
            "regularize", True)
        v[f"D3 PLS 1 comp, {tag}"] = (lin(a=a, factory=pls(1)), "regularize", True)
        v[f"D4 PLS 2 comp, {tag}"] = (lin(a=a, factory=pls(2)), "regularize", True)
        v[f"D5 Bayesian ridge, {tag}"] = (lin(a=a, factory=pipe(BayesianRidge())), "regularize", True)
        v[f"D6 ridge, alpha by inner LOPO, {tag}"] = (
            tuned(lambda al, a=a: lin(a=a, factory=pipe(Ridge(alpha=al))), RIDGE_A), "regularize", True)
        v[f"D7 top-k features tuned, {tag}"] = (tuned(lambda k, a=a: topk(TYPING, k, a), [1, 2, 3, 6]), "regularize", True)
    # E. practice removal (check-in number as a nuisance covariate in training only)
    v["E1 practice-residualized, anch"] = (practice(a=1.0), "practice", True)
    v["E2 practice-residualized, unanch"] = (practice(a=0.0), "practice", True)
    v["E3 practice-resid. test too, anch (NOT deployable)"] = (practice(a=1.0, oracle=True), "practice", False)
    v["E4 practice-resid. test too, unanch (NOT deployable)"] = (practice(a=0.0, oracle=True), "practice", False)
    # F. composite label
    v["F1 composite KSS+MF label, anch"] = (composite(a=1.0), "composite", True)
    v["F2 composite KSS+MF label, unanch"] = (composite(a=0.0), "composite", True)
    # G. ensembles
    v["G1 mean(v1, v3)"] = (mean_of(C.H.BASELINES["v1"], C.H.v3), "ensemble", True)
    v["G2 mean(v1, unanchored v1)"] = (mean_of(C.H.BASELINES["v1"], lin(a=0.0)), "ensemble", True)
    # H. baseline check-ins as extra training rows
    v["H1 v1 + baseline rows, anch"] = (lin(a=1.0, aug=True), "augment", True)
    v["H2 unanchored + baseline rows"] = (lin(a=0.0, aug=True), "augment", True)
    # R. reference only
    v["R1 person offset from own past KSS (NOT deployable)"] = (person_offset(), "reference", False)
    return v


def main():
    t, d, B = C.load()
    print(f"{len(d)} scored rows, {len(B)} baseline rows usable for augmentation")
    v = variants()
    rows, choices = {}, {}
    for name, fn in C.H.BASELINES.items():
        res = C.H.evaluate(d, fn)
        rows[name] = {f"{s}:{k}": val for s, r in res.items() for k, val in r.items() if k != "n"} | {
            "group": "baseline", "deployable": name != "clock"}
    for name, (fn, group, ok) in v.items():
        t0 = time.time()
        res = {}
        for s in ("lopo", "rolling"):
            if hasattr(fn, "chosen"):
                fn.chosen.clear()
            res[s] = C.H.metrics(d, *C.H.predict(d, fn, s))
            if hasattr(fn, "chosen"):
                choices.setdefault(name, {})[s] = [x if not isinstance(x, (tuple, np.ndarray)) else list(x)
                                                   for x in fn.chosen]
        rows[name] = {f"{s}:{k}": val for s, r in res.items() for k, val in r.items() if k != "n"} | {
            "group": group, "deployable": ok}
        print(f"{name:55s} {time.time() - t0:5.1f}s  lopo MAE {res['lopo']['MAE']:.3f} conc "
              f"{res['lopo']['within_concordance']:.3f} | rolling MAE {res['rolling']['MAE']:.3f} conc "
              f"{res['rolling']['within_concordance']:.3f}", flush=True)
    table = pd.DataFrame(rows).T
    table.to_csv(HERE / "results_round1.csv")
    (HERE / "tuned_choices_round1.json").write_text(json.dumps(choices, indent=1, default=float))
    cols = ["lopo:MAE", "lopo:skill", "lopo:within_concordance", "rolling:MAE", "rolling:skill",
            "rolling:within_concordance"]
    pd.set_option("display.width", 250)
    print(table[cols].astype(float).round(3).to_string())
    print(f"n_variants (round 1, excluding the 4 baselines) = {len(v)}")


if __name__ == "__main__":
    main()
