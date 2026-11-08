"""Build keystroke features for every check-in and measure split-half reliability.

    uv run --project ../../.. python run_features.py

Writes (no names, pid only):
  keystroke_features.csv   one row per check-in: raw features, rule-adjusted change from baseline (_dev)
  halves.csv               the same features (and the v1 ones) on each half of the sentence
  reliability.csv          split-half reliability of every feature
No labels are used here: everything in this script is label-free.
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
import ks  # noqa: E402

KS_FEATURES = ["err_per100", "err_rule_per100", "err_typo_per100", "fix_rate", "left_per100",
               "spurious_del_per100", "detect_keys", "detect_ms", "immediate_share", "post_err_ratio",
               "resume_ratio", "pes_ratio", "iki_q10", "iki_q50", "iki_q90", "iki_logsd", "tail_share"]
BASE_RELATIVE = ["tail_base90", "w1_base", "ks_base"]  # already relative to the person's baseline
V1_HALF = {"rule_cost": "vt_rule_cost_dev", "backspace_per100": "backspace_per100_adj",
           "accuracy": "vt_accuracy_dev", "gap_cv": "vt_gap_cv_dev", "pause1s_share": "vt_pause1s_share_dev",
           "hold_med": "vt_hold_med_dev"}
V1_EXTRA = ["gap_med", "wpm", "pause_rel_share"]  # not in v1, for context
MIN_PANEL_PER_RULE = 3
N_BOOT = 1000


def rule_adjust(f: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """x - typical value for that rule + overall typical value; typical = panel median per rule
    (all check-ins of that rule when the panel has fewer than 3), as features.rule_references does."""
    panel = f.participant_type.eq("panel")
    out = {}
    for c in cols:
        typical = f.groupby("question_type")[c].median()
        on_panel = f[panel].groupby("question_type")[c].agg(["median", "count"])
        enough = on_panel.index[on_panel["count"] >= MIN_PANEL_PER_RULE]
        typical.loc[enough] = on_panel.loc[enough, "median"]
        overall = f.loc[panel, c].median()
        out[c] = f[c] - f.question_type.map(typical) + overall
    return pd.DataFrame(out, index=f.index)


def sb(r: float) -> float:
    return 2 * r / (1 + r) if r > 0 else r


def centred_pairs(h: pd.DataFrame, a: str, b: str, people: list[str]) -> pd.DataFrame:
    """Each half rule-adjusted (minus that half's median for the rule), then person-centred."""
    x = h[h.pid.isin(people)].copy()
    for c in (a, b):
        x[c] = x[c] - x.groupby("question_type")[c].transform("median")
        x[c] = x[c] - x.groupby("pid")[c].transform("mean")
    return x.dropna(subset=[a, b])


def reliability(h: pd.DataFrame, feats: list[str], people: list[str], tag: str = "h", seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for f in feats:
        a, b = f"{f}_{tag}1", f"{f}_{tag}2"
        if a not in h or b not in h:
            continue
        both = h.dropna(subset=[a, b])
        r_all = stats.spearmanr(both[a], both[b]).statistic if len(both) >= 6 else np.nan
        w = centred_pairs(h, a, b, people)
        r_w = stats.spearmanr(w[a], w[b]).statistic if len(w) >= 6 and w[a].std() > 0 and w[b].std() > 0 else np.nan
        boots = []
        if not np.isnan(r_w):
            xa, xb = w[a].to_numpy(float), w[b].to_numpy(float)
            for _ in range(N_BOOT):
                i = rng.integers(0, len(w), len(w))
                if np.ptp(xa[i]) > 0 and np.ptp(xb[i]) > 0:
                    boots.append(stats.spearmanr(xa[i], xb[i]).statistic)
        lo, hi = (np.quantile(boots, [0.05, 0.95]) if boots else (np.nan, np.nan))
        rows.append({"feature": f, "n_all": len(both), "r_all": r_all, "r_all_sb": sb(r_all),
                     "n_within": len(w), "r_within": r_w, "r_within_sb": sb(r_w),
                     "r_within_lo90": lo, "r_within_hi90": hi})
    return pd.DataFrame(rows)


def main() -> None:
    t = harness.load_table()
    sessions = harness.raw_sessions()
    ids = list(t.checkin_id)
    assert set(ids) <= set(sessions)
    info = t.set_index("checkin_id")

    replays = {cid: ks.replay(sessions[cid]) for cid in ids}
    log_gaps = {cid: ks.log_ordinary_gaps(ev, meta) for cid, (ev, pr, meta) in replays.items()}

    # each panel person's baseline gaps; a baseline check-in is compared with the other baseline one
    base_ids = t[t.is_baseline].groupby("pid").checkin_id.apply(list).to_dict()

    def reference(cid):
        pid = info.loc[cid, "pid"]
        refs = [c for c in base_ids.get(pid, []) if c != cid]
        return np.concatenate([log_gaps[c] for c in refs]) if refs else None

    full, halves = [], []
    for cid in ids:
        ev, pr, meta = replays[cid]
        ref = reference(cid)
        row = {"checkin_id": cid, **ks.keystroke_features(ev, meta, None, ref)}
        full.append(row)
        h = {"checkin_id": cid}
        for split, tag in (("half", "h"), ("oddeven", "o")):
            for part in (1, 2):
                for k, v in ks.keystroke_features(ev, meta, part, ref, split).items():
                    h[f"{k}_{tag}{part}"] = v
                for k, v in ks.v1_native(pr, meta, split, part).items():
                    h[f"v1_{k}_{tag}{part}"] = v
        halves.append(h)

    keep = ["checkin_id", "pid", "participant_type", "timestamp", "is_baseline", "question_type"]
    f = t[keep].merge(pd.DataFrame(full), on="checkin_id")
    adj = rule_adjust(f, KS_FEATURES)
    panel = f.participant_type.eq("panel")
    for c in KS_FEATURES:
        base_mean = adj[c].where(f.is_baseline).groupby(f.pid).transform("mean")
        f[f"ks_{c}_dev"] = (adj[c] - base_mean).where(panel)
    for c in BASE_RELATIVE:
        f[f"ks_{c}_dev"] = f[c]  # already a change from the person's baseline
    f = f.rename(columns={c: f"ks_{c}" for c in KS_FEATURES + BASE_RELATIVE + ["n_ins", "n_del"]})
    f.to_csv(HERE / "keystroke_features.csv", index=False)

    h = t[keep].merge(pd.DataFrame(halves), on="checkin_id")
    h.to_csv(HERE / "halves.csv", index=False)

    people = [p for p, n in t[panel].pid.value_counts().items() if n >= 3]
    feats = KS_FEATURES + BASE_RELATIVE + [f"v1_{k}" for k in list(V1_HALF) + ["gap_med"]]
    halves_rel = reliability(h, feats, people, "h")
    odd_rel = reliability(h, feats, people, "o")
    rel = halves_rel.merge(odd_rel, on="feature", suffixes=("", "_oe"))
    # the two splits' within-person estimates, averaged on Fisher's z scale, then Spearman-Brown
    z = (np.arctanh(rel.r_within.clip(-0.99, 0.99)) + np.arctanh(rel.r_within_oe.clip(-0.99, 0.99))) / 2
    rel["r_within_sb_mean"] = np.tanh(z).map(sb)
    rel["in_v1"] = rel.feature.map(lambda x: V1_HALF.get(x.removeprefix("v1_"), "") if x.startswith("v1_") else "")
    rel.to_csv(HERE / "reliability.csv", index=False)
    # sensitivity: without P1's first baseline check-in (8 errors left: a misread rule)
    odd_one = t.loc[t.pid.eq("P1") & t.is_baseline, "checkin_id"].iloc[0]
    sens = reliability(h[h.checkin_id != odd_one], feats, people, "h")[["feature", "r_within_sb"]]
    rel = rel.merge(sens.rename(columns={"r_within_sb": "r_within_sb_noP1b1"}), on="feature")
    rel.to_csv(HERE / "reliability.csv", index=False)

    # sentence-level descriptives (label-free)
    pd.set_option("display.width", 220)
    print(f"{len(f)} check-ins; within-person reliability uses {', '.join(people)} "
          f"({int(t.pid.isin(people).sum())} check-ins)\n")
    show = ["feature", "n_all", "r_all_sb", "n_within", "r_within_sb", "r_within_lo90", "r_within_hi90",
            "n_within_oe", "r_within_sb_oe", "r_within_sb_mean", "r_within_sb_noP1b1", "in_v1"]
    print(rel[show].round(2).to_string(index=False))
    sc = f[f.pid.isin(people)]
    print("\nPer check-in (P1-P4, all 32), median [IQR]:")
    for c in ["ks_err_per100", "ks_err_rule_per100", "ks_err_typo_per100", "ks_fix_rate", "ks_left_per100",
              "ks_spurious_del_per100", "ks_detect_keys", "ks_detect_ms", "ks_immediate_share",
              "ks_post_err_ratio", "ks_resume_ratio", "ks_pes_ratio", "ks_iki_q50", "ks_tail_share",
              "ks_tail_base90", "ks_w1_base"]:
        x = sc[c].dropna()
        print(f"  {c:26s} n={len(x):2d}  {x.median():8.3f} [{x.quantile(.25):.3f}, {x.quantile(.75):.3f}]")
    errs = sum(int(ev.loc[ev.kind.eq("ins"), "err"].astype(bool).sum()) for ev, _, _ in replays.values())
    print(f"\nError keystrokes in all 36 check-ins: {errs}")
    # how much do rule types move the ordinary-gap features? (Kruskal-Wallis across rule types)
    for c in ["ks_iki_q50", "ks_iki_q90", "ks_tail_share", "ks_err_per100", "ks_err_rule_per100"]:
        groups = [g[c].dropna().to_numpy() for _, g in f.groupby("question_type")]
        print(f"  rule-type effect on {c}: Kruskal-Wallis p = {stats.kruskal(*groups).pvalue:.3f}")


if __name__ == "__main__":
    main()
