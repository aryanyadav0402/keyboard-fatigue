"""Diagnostics: why anchoring helps or hurts, per-person errors of the leading variants, and coefficients.

    uv run --project /Users/aryan/Projects/hacktheburgh/keyboard-fatigue-nobaseline/analysis python diagnose.py
"""

import numpy as np
import pandas as pd
from scipy import stats

import common as C
from common import TYPING, lin, ordinal, pairwise, tuned, within_ridge

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)


def per_person(d, fn, scheme):
    pred, _ = C.H.predict(d, fn, scheme)
    m = ~np.isnan(pred)
    e = pd.DataFrame({"pid": d.pid[m], "err": pred[m] - d.kss[m]})
    return e.groupby("pid").err.agg(MAE=lambda x: x.abs().mean(), bias="mean", n="size")


def main():
    t, d, B = C.load()

    print("== Person level: baseline rating vs the person's later ratings ==")
    g = d.groupby("pid").agg(kss_base=("kss_base", "first"), kss_mean=("kss", "mean"), kss_sd=("kss", "std"),
                             delta_mean=("kss_delta", "mean"), n=("kss", "size"))
    print(g.round(2).to_string())
    print("Pearson r(kss_base, person mean kss) =", round(np.corrcoef(g.kss_base, g.kss_mean)[0, 1], 3))
    for y in ("kss", "kss_delta"):
        pm = d.groupby("pid")[y].transform("mean")
        print(f"{y:9s} between-person var {pm.var(ddof=0):.3f}, within-person var {(d[y] - pm).var(ddof=0):.3f}")
    # per-person mean of the 6 typing features: person-level offsets carried by the features themselves
    print("\nPerson means of the typing features (dev features are relative to each person's baseline;"
          " backspace_per100_adj is not):")
    print(d.groupby("pid")[TYPING].mean().round(3).to_string())

    print("\n== Within-person ordering: is KSS monotone in check-in order? ==")
    conc_clock = C.concordance(d, d.hours_rel.to_numpy(float))
    print(f"concordance of check-in time with KSS within person: {conc_clock:.3f}")
    for c in TYPING + ["rt_speed_dev", "h_rate_dev", "h_errr_dev"]:
        x = d[c].to_numpy(float)
        rho_t = np.nanmedian([stats.spearmanr(gg[c], gg.hours_rel).statistic for _, gg in d.groupby("pid")])
        print(f"  {c:22s} within-person conc with KSS {C.concordance(d, x):.3f}  (median within rho with clock {rho_t:+.2f})")

    print("\n== First 22 scored rows (the 00:37 batch the earlier report used) vs all 24 ==")
    for label, dd in (("first 22", d[d.timestamp <= pd.Timestamp('2026-11-08 00:18:48')].reset_index(drop=True)), ("all 24", d)):
        for name, fn in (("anchored v1", lin(a=1.0)), ("unanchored v1", lin(a=0.0))):
            r = C.H.metrics(dd, *C.H.predict(dd, fn, "lopo"))
            print(f"  {label:8s} n={len(dd)} {name:14s} LOPO MAE {r['MAE']:.3f} skill {r['skill']:.3f} conc {r['within_concordance']:.3f}")
        print("  per person (LOPO), anchored | unanchored:")
        pa, pu = per_person(dd, lin(a=1.0), "lopo"), per_person(dd, lin(a=0.0), "lopo")
        print(pd.concat([pa.add_prefix("anch_"), pu.add_prefix("unanch_")], axis=1).round(2).to_string())

    print("\n== Per-person errors of leading variants ==")
    cands = {"v1": lin(a=1.0), "A1 unanch": lin(a=0.0), "H1 v1+base rows": lin(a=1.0, aug=True),
             "B7 within unanch": within_ridge(a=0.0), "B3 pairwise unanch": pairwise(a=0.0),
             "B5 pairwise+base": pairwise(a=1.0, aug=True),
             "C2 ordinal delta": tuned(lambda al: ordinal(alpha=al, a=1.0), [1.0, 0.3, 3.0, 10.0]),
             "G1 mean(v1,v3)": C.mean_of(C.H.BASELINES["v1"], C.H.v3)}
    for scheme in ("lopo", "rolling"):
        tabs = {k: per_person(d, fn, scheme) for k, fn in cands.items()}
        print(f"-- {scheme}: MAE per person")
        print(pd.DataFrame({k: v.MAE for k, v in tabs.items()}).round(2).to_string())
        print(f"-- {scheme}: mean signed error (pred - true) per person")
        print(pd.DataFrame({k: v.bias for k, v in tabs.items()}).round(2).to_string())

    print("\n== Level-only references (no typing): how much is just the anchor and the intercept? ==")

    def anchor_only(train, test):
        return test.kss_base.to_numpy(float) + train.kss_delta.mean()

    def anchor_only_aug(train, test):
        rows = pd.concat([train, C.aug_rows(train, test)])
        return test.kss_base.to_numpy(float) + rows.kss_delta.mean()

    def guess_aug(train, test):
        rows = pd.concat([train, C.aug_rows(train, test)])
        return np.full(len(test), rows.kss.mean())

    refs = {"guess (harness)": C.H.guess, "Z1 anchor only: kss_base + mean change": anchor_only,
            "Z2 anchor only, mean change incl. baseline rows": anchor_only_aug,
            "Z3 guess incl. baseline rows": guess_aug,
            "v1": lin(a=1.0), "H1 v1 + baseline rows": lin(a=1.0, aug=True)}
    tab = {}
    for k, fn in refs.items():
        r = C.H.evaluate(d, fn)
        tab[k] = {f"{s}:{m}": r[s][m] for s in r for m in ("MAE", "skill", "within_concordance")}
    tab = pd.DataFrame(tab).T.astype(float).round(3)
    print(tab.to_string())
    tab.to_csv(C.H.EXP / "strategies" / "level_only_refs.csv")

    print("\n== Coefficients when fitted on all 24 rows (standardized features) ==")
    X = d[TYPING].to_numpy(float)
    co = {}
    for name, y, rows in (("v1 anchored", d.kss_delta, d), ("unanchored", d.kss, d)):
        m = C.ridge().fit(rows[TYPING].to_numpy(float), y.to_numpy(float))
        co[name] = list(m[-1].coef_) + [m[-1].intercept_, m[-1].alpha_]
    aug = pd.concat([d, B])
    for name, y in (("H1 anchored+base rows", aug.kss_delta), ("H2 unanch+base rows", aug.kss)):
        m = C.ridge().fit(aug[TYPING].to_numpy(float), y.to_numpy(float))
        co[name] = list(m[-1].coef_) + [m[-1].intercept_, m[-1].alpha_]
    f = C.prep(d, TYPING)
    Xs, pids = f(d), d.pid.to_numpy()
    for name, y in (("B7 within (FE)", d.kss.to_numpy(float)),):
        m = C.RidgeCV(alphas=C.ALPHAS, fit_intercept=False).fit(Xs - C.group_mean(Xs, pids), y - C.group_mean(y, pids))
        co[name] = list(m.coef_) + [y.mean(), m.alpha_]
    print(pd.DataFrame(co, index=TYPING + ["intercept", "alpha"]).round(3).to_string())


if __name__ == "__main__":
    main()
