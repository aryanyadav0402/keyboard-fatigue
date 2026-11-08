"""Post-hoc diagnostics, run AFTER the four candidates were scored. Nothing here is a model variant.

    uv run --project ../../.. python diagnose.py

1. anchor only: own baseline rating + average training change, no typing (a reference for what the
   typing features add on top of the anchor)
2. LOPO error per held-out person for v1 and the candidates
3. within-person associations (person-centred Spearman over P1-P4's check-ins, baselines included):
   each feature with KSS, with check-in number (practice / time), and with KSS after removing a
   within-person linear trend in check-in number from both
Writes associations.csv.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import harness  # noqa: E402
from harness import TYPING, anchored, ridge  # noqa: E402
from run_models import CANDIDATES, load  # noqa: E402
from run_features import rule_adjust  # noqa: E402


def anchor_only(train, test):
    return test.kss_base.to_numpy(float) + train.kss_delta.mean()


def centre(x: pd.Series, g: pd.Series) -> pd.Series:
    return x - x.groupby(g).transform("mean")


def detrend(x: pd.Series, n: pd.Series, g: pd.Series) -> pd.Series:
    """Residual of x on check-in number within each person (linear)."""
    out = pd.Series(np.nan, index=x.index)
    for _, idx in x.groupby(g).groups.items():
        xi, ni = x.loc[idx], n.loc[idx]
        ok = xi.notna()
        if ok.sum() >= 3:
            b = np.polyfit(ni[ok], xi[ok], 1)
            out.loc[idx[ok]] = xi[ok] - np.polyval(b, ni[ok])
    return out


def wrho(a: pd.Series, b: pd.Series) -> tuple[float, int]:
    ok = a.notna() & b.notna()
    if ok.sum() < 8:
        return np.nan, int(ok.sum())
    return float(stats.spearmanr(a[ok], b[ok]).statistic), int(ok.sum())


def main() -> None:
    pd.set_option("display.width", 250)
    t, d = load()
    print(harness.compare(d, {"anchor only (reference)": anchor_only}, baselines=False).to_string(), "\n")

    # per-person LOPO error
    rows = {}
    for name, cols in {"v1": TYPING, **CANDIDATES}.items():
        pred, _ = harness.predict(d, anchored(ridge, cols), "lopo")
        rows[name] = (pd.Series(np.abs(pred - d.kss.to_numpy(float))).groupby(d.pid.to_numpy()).mean())
    pred, _ = harness.predict(d, anchor_only, "lopo")
    rows["anchor only"] = pd.Series(np.abs(pred - d.kss.to_numpy(float))).groupby(d.pid.to_numpy()).mean()
    print("LOPO MAE by held-out person:")
    print(pd.DataFrame(rows).T.round(2).to_string(), "\n")

    # within-person associations, P1-P4, baselines included (features as rule-adjusted levels)
    panel = t[t.pid.isin(["P1", "P2", "P3", "P4"]) & t.kss.notna()].copy()
    ks_raw = pd.read_csv(HERE / "keystroke_features.csv")
    lvl = [c for c in ks_raw if c.startswith("ks_") and not c.endswith("_dev") and c not in ("ks_n_ins", "ks_n_del")]
    assert all(c in panel for c in lvl)  # load() already merged the keystroke features
    v1_levels = {"vt_rule_cost": "vt_rule_cost_adj", "backspace_per100": "backspace_per100_adj",
                 "vt_accuracy": "vt_accuracy_adj", "vt_gap_cv": "vt_gap_cv_adj",
                 "vt_pause1s_share": "vt_pause1s_share_adj", "vt_hold_med": "vt_hold_med_adj"}
    base_rel = ["ks_tail_base90", "ks_w1_base", "ks_ks_base"]
    adj_cols = [c for c in lvl if c not in base_rel]
    adj = rule_adjust(panel.assign(participant_type="panel"), adj_cols + [c for c in v1_levels if c in panel])
    feats = {**{c: adj[c] for c in adj_cols}, **{c: panel[c] for c in base_rel},
             **{v1_levels[c]: adj[c] for c in v1_levels if c in panel}}
    g, n, k = panel.pid, panel.checkin_n.astype(float), panel.kss.astype(float)
    k_c, n_c, k_dt = centre(k, g), centre(n, g), detrend(k, n, g)
    print(f"within-person KSS vs check-in number: rho = {wrho(k_c, n_c)[0]:.2f} (n={len(panel)})\n")
    out = []
    for name, x in feats.items():
        x = pd.Series(np.asarray(x, float), index=panel.index)
        r_k, nn = wrho(centre(x, g), k_c)
        r_n, _ = wrho(centre(x, g), n_c)
        r_p, _ = wrho(detrend(x, n, g), k_dt)
        out.append({"feature": name, "n": nn, "rho_kss": r_k, "rho_checkin_n": r_n, "rho_kss_detrended": r_p})
    a = pd.DataFrame(out)
    a.to_csv(HERE / "associations.csv", index=False)
    print("Within-person (person-centred) Spearman, P1-P4 incl. baselines; exploratory, many comparisons:")
    print(a.round(2).to_string(index=False))


if __name__ == "__main__":
    main()


def ablation_and_checks() -> None:
    """v1 minus one feature at a time (diagnostic; not candidates, not eligible for freezing), and two
    label-free checks of a speed-accuracy story for the error result."""
    t, d = load()
    fns = {f"v1 minus {c}": anchored(ridge, [x for x in TYPING if x != c]) for c in TYPING}
    tab = harness.compare(d, fns)
    tab.to_csv(HERE / "ablation_table.csv")
    print("\nv1 leave-one-feature-out (diagnostic):")
    print(tab[["lopo:MAE", "lopo:skill", "lopo:within_concordance", "rolling:MAE", "rolling:skill",
               "rolling:within_concordance"]].to_string())
    panel = t[t.pid.isin(["P1", "P2", "P3", "P4"])].copy()
    g = panel.pid
    for a, b in [("ks_err_per100", "vt_pause1s_share"), ("ks_err_per100", "ks_iki_q50"),
                 ("ks_err_typo_per100", "vt_pause1s_share"), ("ks_err_per100", "vt_gap_cv")]:
        r, n = wrho(centre(panel[a], g), centre(panel[b], g))
        print(f"label-free, within person: {a} vs {b}: rho = {r:.2f} (n={n})")


if __name__ == "__main__":
    ablation_and_checks()


def score_reliability() -> None:
    """Split-half reliability of the v1 score itself: half-sentence v1 features, rule-adjusted and
    person-centred, z-scored, combined with v1's standardised weights from all 24 (post hoc)."""
    from run_features import sb
    t, d = load()
    h = pd.read_csv(HERE / "halves.csv")
    h = h[h.pid.isin(["P1", "P2", "P3", "P4"])].copy()
    names = {"vt_rule_cost_dev": "rule_cost", "backspace_per100_adj": "backspace_per100",
             "vt_accuracy_dev": "accuracy", "vt_gap_cv_dev": "gap_cv", "vt_pause1s_share_dev": "pause1s_share",
             "vt_hold_med_dev": "hold_med"}
    m = ridge().fit(d[TYPING].to_numpy(float), d.kss_delta.to_numpy(float))
    w = dict(zip(TYPING, m[-1].coef_))
    combos = {"v1 score (6 features)": TYPING, "gap_cv + pause1s only": ["vt_gap_cv_dev", "vt_pause1s_share_dev"]}
    for tag, label in (("h", "first/second half"), ("o", "odd/even words")):
        for complete in (False, True):
            hh = h.dropna(subset=[f"v1_{s}_{tag}{p}" for s in names.values() for p in (1, 2)]) if complete else h
            z = {}
            for col, short in names.items():
                for part in (1, 2):
                    c = f"v1_{short}_{tag}{part}"
                    x = hh[c] - hh.groupby("question_type")[c].transform("median")
                    x = x - x.groupby(hh.pid).transform("mean")
                    z[(col, part)] = (x / x.std()).fillna(0.0)
            for cname, cols in combos.items():
                s1 = sum(w[c] * z[(c, 1)] for c in cols)
                s2 = sum(w[c] * z[(c, 2)] for c in cols)
                r = stats.spearmanr(s1, s2).statistic
                print(f"  {label:18s} {cname:24s} {'complete halves' if complete else 'all, gaps->mean':16s} "
                      f"within-person split-half r = {r:.2f}, Spearman-Brown {sb(r):.2f} (n={len(hh)})")


if __name__ == "__main__":
    print("\nReliability of model scores (post hoc):")
    score_reliability()
