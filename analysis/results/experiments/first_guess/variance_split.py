"""How much of the KSS variance is between people and how much within, and the same for the features.

    uv run --project <analysis> python variance_split.py

The clock appears here only as a nuisance covariate (to ask how much of the between-person spread
survives once the time of night is allowed for), never as a model input. Writes variance_split.json.
"""

import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from common import FG5, HERE, load, rule_adjust


def anova_icc(df: pd.DataFrame, col: str = "kss") -> dict:
    """One-way random-effects split (unequal group sizes): between-person variance, within-person
    variance, their shares, and the F test that the people's means differ."""
    d = df[["pid", col]].dropna()
    g = d.groupby("pid")[col]
    n_i, means = g.size(), g.mean()
    N, k, grand = len(d), len(n_i), d[col].mean()
    ssb = float((n_i * (means - grand) ** 2).sum())
    ssw = float(((d[col] - d.pid.map(means)) ** 2).sum())
    msb, msw = ssb / (k - 1), ssw / (N - k)
    n0 = (N - (n_i ** 2).sum() / N) / (k - 1)
    between = max(0.0, (msb - msw) / n0)
    return {"rows": N, "people": k, "between_var": between, "within_var": msw,
            "between_share": between / (between + msw) if between + msw > 0 else float("nan"),
            "F": msb / msw if msw > 0 else float("nan"),
            "F_p": float(stats.f.sf(msb / msw, k - 1, N - k)) if msw > 0 else float("nan"),
            "naive_between_share_of_SS": ssb / (ssb + ssw)}


def mixed(df: pd.DataFrame, formula: str) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = smf.mixedlm(formula, df, groups=df["pid"]).fit(reml=True)
    between, within = float(m.cov_re.iloc[0, 0]), float(m.scale)
    return {"formula": formula, "between_var": between, "within_var": within,
            "between_share": between / (between + within)}


def matched_spread(df: pd.DataFrame) -> dict:
    """At each check-in number with 3+ people: the spread of KSS across people (same time of night,
    same amount of practice), against the typical spread within a person."""
    rows = []
    for n, g in df.groupby("checkin_n"):
        if g.pid.nunique() >= 3:
            rows.append({"checkin_n": int(n), "people": int(g.pid.nunique()), "kss": g.kss.tolist(),
                         "var_across_people": float(g.kss.var(ddof=1))})
    within = df[df.groupby("pid").kss.transform("size") >= 2].groupby("pid").kss.var(ddof=1)
    return {"by_checkin": rows,
            "mean_var_across_people_same_checkin": float(np.mean([r["var_across_people"] for r in rows])),
            "mean_var_within_person": float(within.mean())}


def main():
    panel, walk = load()
    panel = panel.assign(hours=(panel.timestamp - panel.timestamp.min()).dt.total_seconds() / 3600,
                         log_n=np.log(panel.checkin_n))
    multi = panel[panel.groupby("pid").kss.transform("size") >= 2]
    out = {
        "kss_all_panel": anova_icc(panel),
        "kss_4_people_with_repeats": anova_icc(multi),
        "kss_mixed_no_time": mixed(panel, "kss ~ 1"),
        "kss_mixed_clock_adjusted": mixed(panel, "kss ~ hours"),
        "kss_mixed_checkin_adjusted": mixed(panel, "kss ~ log_n"),
        "kss_same_checkin_spread": matched_spread(panel),
        "kss_first_checkins": {"panel": panel[panel.checkin_n.eq(1)].kss.tolist(), "walkups": walk.kss.tolist()},
        "kss_by_person": {p: {"n": int(len(g)), "mean": round(float(g.kss.mean()), 2),
                              "sd": round(float(g.kss.std(ddof=1)), 2) if len(g) > 1 else None}
                          for p, g in panel.groupby("pid")},
    }
    # Features: how much of each one's spread is who you are (descriptive; shipped-style references
    # from every panel check-in).
    adj = rule_adjust(panel, panel)
    feats = FG5 + ["log_wpm", "sqrt_bsp", "log_rt", "vt_hold_med"]
    out["feature_between_share"] = {c: round(anova_icc(multi.assign(**{c: adj.loc[multi.index, c]}), c)["between_share"], 3)
                                    for c in feats}
    (HERE / "variance_split.json").write_text(json.dumps(out, indent=2, default=float) + "\n")
    for k, v in out.items():
        print(k, json.dumps(v, default=lambda x: round(float(x), 3)))


if __name__ == "__main__":
    main()
