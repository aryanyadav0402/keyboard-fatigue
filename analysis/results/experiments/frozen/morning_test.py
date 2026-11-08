"""Morning test, frozen at about 02:40 on 8 Nov 2026, before anyone looked at a check-in after 00:53:51.

The overnight experiments (analysis/results/experiments/, see FINDINGS.md) found that much of what our
models pick up on the night's data looks like practice: typing gets smoother and cleaner with every
check-in while the night gets later, so smooth typing ends up meaning "tired". This test judges the
models on check-ins they have never seen and adds the check that separates tiredness from practice:
what happens after people sleep, when they're still practised but rested.

Every choice below was fixed before the new data arrived. Don't change it after seeing results.

    cd analysis
    uv run python export_features.py --data <new data folder> --out results/<name>
    uv run python results/experiments/frozen/morning_test.py results/<name> <new data folder>

Checks, and what counts as passing:
1. Personal models (teammates compared with their own baseline), fitted on scored teammate check-ins
   up to CUTOFF and scored on later ones: guess, clock (a benchmark, never deployable), v1, v3 (SWELL
   stress score) and H1 (v1 with each teammate's two baseline check-ins as extra training rows).
   v1 passes if its error is below guessing's. H1 counts as better than v1 only if it has a lower error
   without a lower within-person concordance.
2. Order beyond the clock: new same-person pairs where the later check-in was rated less tired. A model
   that only tracks the night's trend gets these wrong; overnight, v1 got 6 of 6 and the clock 0.
3. After sleep: each teammate's first check-in after a gap of GAP_HOURS or more since their previous
   one (or a reported nap of an hour or more). Tiredness says a model's prediction should fall when the
   rating falls; practice says it won't. A model passes if it moves the same way as the rating for most
   such gaps.
4. First try for strangers, scoring each new teammate check-in and new walk-up as if it had no baseline:
   - the shipped index (the live API's model): its rank correlation with KSS. Below 0 means it tracks
     practice, as on the night's data (-0.50); the percentile shown to strangers is then backwards.
   - ridge_fg5 (frozen numbers in ridge_fg5.json): passes only if its error beats always saying 6.12,
     its rank correlation is above 0, and it predicts lower KSS after sleep (check 3).
5. Nap rule: predictions of NAP_AT or more against ratings of NAP_AT or more: caught, missed, false alarms.
6. Mistakes: the within-person correlation between errors made per 100 characters and KSS is below 0,
   and the median share of errors fixed stays at 0.9 or more.
Teammates only: anyone whose sessions are all aborted (the test session) is dropped from the panel.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parents[2]
sys.path.insert(0, str(ANALYSIS))
sys.path.insert(1, str(HERE))

from fair_test import SWELL_STRESS, add_her_numbers  # noqa: E402
from features import apply_rule_references  # noqa: E402
from first_guess import FIRST_GUESS  # noqa: E402
from frozen_model import check_model, first_guess_index  # noqa: E402
from model import CLOCK, TYPING, ridge  # noqa: E402

import ks  # noqa: E402  (frozen copy of the keystroke track's replay code)

CUTOFF = pd.Timestamp("2026-11-08 00:53:51")
GAP_HOURS = 2.0
NAP_AT = 7
LIVE_MODEL = ANALYSIS.parent / "service" / "models" / "team-2026-11-08-0110.json"
FG5 = json.loads((HERE / "ridge_fg5.json").read_text())
FG5_RAW = {"vt_rule_cost_adj": "vt_rule_cost", "vt_gap_cv_adj": "vt_gap_cv",
           "vt_pause_rel_share_adj": "vt_pause_rel_share", "backspace_per100_adj": "backspace_per100",
           "vt_accuracy_adj": "vt_accuracy"}


# ---- data ------------------------------------------------------------------------------------

def sessions_in(folder: Path) -> dict:
    out = {}
    for path in sorted(Path(folder).rglob("*.json")):
        if path.name.startswith("._"):
            continue
        try:
            s = json.loads(path.read_text())
        except (ValueError, UnicodeDecodeError):
            continue
        if isinstance(s, dict) and s.get("session_id"):
            out[s["session_id"]] = s
    return out


def load(out: Path, sessions: dict) -> pd.DataFrame:
    f = pd.read_csv(out / "features_processed.csv")
    raw = pd.read_csv(out / "features_raw.csv")
    t = f.merge(raw[["checkin_id"] + [c for c in raw.columns if c not in f.columns]], on="checkin_id")
    t["is_baseline"] = t.is_baseline.astype(bool)
    t = add_her_numbers(t, raw)
    t["timestamp"] = pd.to_datetime(t.timestamp)
    t["completed"] = t.checkin_id.map(lambda c: bool(sessions.get(c, {}).get("completed", True)))
    t["teammate"] = t.participant_type.eq("panel") & t.groupby("person").completed.transform("any")
    people = sorted(t.loc[t.teammate, "person"].unique())
    walkups = sorted(t.loc[~t.participant_type.eq("panel"), "person"].unique())
    codes = {p: f"P{i + 1}" for i, p in enumerate(people)} | {p: f"W{i + 1}" for i, p in enumerate(walkups)}
    t["pid"] = t.person.map(codes).fillna("dropped")
    if "kss_delta" not in t or t.kss_delta.isna().all():
        t["kss_delta"] = t.kss - t.kss_base
    return t.sort_values("timestamp").reset_index(drop=True)


def scored(t: pd.DataFrame) -> pd.DataFrame:
    keep = t.teammate & ~t.is_baseline & t.kss.notna() & t[TYPING].notna().any(axis=1)
    return t[keep]


# ---- frozen models ---------------------------------------------------------------------------

def guess(train, rows):
    return np.full(len(rows), train.kss.mean())


def anchored(cols):
    def fit_predict(train, rows):
        m = ridge().fit(train[cols].to_numpy(float), train.kss_delta.to_numpy(float))
        return m.predict(rows[cols].to_numpy(float)) + rows.kss_base.to_numpy(float)
    return fit_predict


def v3(train, rows):
    spread = {c: train[f"h_{c}_dev"].std() for c in SWELL_STRESS}
    score = lambda frame: sum(w * frame[f"h_{c}_dev"] / spread[c] for c, w in SWELL_STRESS.items()).to_numpy(float)
    m = ridge().fit(score(train).reshape(-1, 1), train.kss_delta.to_numpy(float))
    return m.predict(score(rows).reshape(-1, 1)) + rows.kss_base.to_numpy(float)


def make_h1(baseline_rows: pd.DataFrame):
    """v1 trained with each teammate's two baseline check-ins added as rows (strategies track, H1)."""
    def fit_predict(train, rows):
        both = baseline_rows[baseline_rows.groupby("person").person.transform("size") == 2]
        aug = pd.concat([train, both[both.person.isin(train.person.unique())]])
        m = ridge().fit(aug[TYPING].to_numpy(float), aug.kss_delta.to_numpy(float))
        return m.predict(rows[TYPING].to_numpy(float)) + rows.kss_base.to_numpy(float)
    return fit_predict


def ridge_fg5(rows: pd.DataFrame) -> np.ndarray:
    """The frozen no-baseline model from the first-guess track, applied with its own numbers."""
    cols = []
    for adj in FG5["features"]:
        raw = FG5_RAW[adj]
        ref = FG5["rule_references"][raw]
        typical = rows.question_type.map(ref["typical"]).fillna(ref["overall"]).astype(float)
        x = rows[raw].astype(float)
        x = x / typical.where(typical > 0) if ref["rule"] == "divide" else x - typical + ref["overall"]
        x = x.fillna(FG5["impute_median"][adj])
        cols.append((x - FG5["mean"][adj]) / FG5["sd"][adj] * FG5["weight_per_sd"][adj])
    return np.clip(FG5["intercept"] + np.sum(cols, axis=0), *FG5["clip"])


def shipped_index(rows: pd.DataFrame) -> np.ndarray:
    """The live API's first-guess index (higher = typing looks more tired, by the fixed directions)."""
    model = check_model(json.loads(LIVE_MODEL.read_text()))
    adj = apply_rule_references(rows.drop(columns=[c for c in rows if c.endswith("_adj")]), model["rule_references"])
    return first_guess_index(model).index(adj[FIRST_GUESS].to_numpy(float))


# ---- measures --------------------------------------------------------------------------------

def concordance(frame: pd.DataFrame, col: str) -> tuple[float, int, int]:
    good, pairs = 0.0, 0
    for _, g in frame.groupby("pid"):
        k, q = g.kss.to_numpy(float), g[col].to_numpy(float)
        for i in range(len(k)):
            for j in range(i + 1, len(k)):
                if k[i] != k[j]:
                    pairs += 1
                    s = np.sign(q[i] - q[j]) * np.sign(k[i] - k[j])
                    good += 1.0 if s > 0 else 0.5 if s == 0 else 0.0
    return (good / pairs if pairs else float("nan")), good, pairs


def spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    return float(stats.spearmanr(a[ok], b[ok]).statistic) if ok.sum() > 2 and np.ptp(a[ok]) > 0 else float("nan")


def run(out: Path, data: Path, cutoff: pd.Timestamp = CUTOFF) -> dict:
    sessions = sessions_in(data)
    t = load(out, sessions)
    rows = scored(t)
    train, test = rows[rows.timestamp <= cutoff], rows[rows.timestamp > cutoff].copy()
    baselines = t[t.teammate & t.is_baseline & t.kss.notna() & (t.timestamp <= cutoff)]
    teammates_all = t[t.teammate].copy()
    report = {"cutoff": str(cutoff), "train_checkins": len(train), "test_checkins": len(test),
              "test_people": int(test.pid.nunique())}
    print(f"Fitted on {len(train)} scored teammate check-ins up to {cutoff}; "
          f"{len(test)} new scored check-ins from {test.pid.nunique()} teammates.\n")
    if test.empty:
        print("No new scored teammate check-ins yet.")
        return report

    personal = {"guess": guess, "clock (benchmark)": anchored(CLOCK), "v1": anchored(TYPING), "v3": v3,
                "H1": make_h1(baselines)}
    for name, fn in personal.items():
        test[name] = np.clip(fn(train, test), 1, 9)
        teammates_all[name] = np.clip(fn(train, teammates_all), 1, 9)
    test["ridge_fg5"] = ridge_fg5(test)
    teammates_all["ridge_fg5"] = ridge_fg5(teammates_all)
    teammates_all["shipped_index"] = shipped_index(teammates_all)

    # 1. personal models
    kss = test.kss.to_numpy(float)
    guess_sq = ((test["guess"] - kss) ** 2).sum()
    print("1. Personal models on new check-ins (error in KSS points; skill 0 = guessing)")
    table = {}
    for name in personal:
        err = test[name].to_numpy(float) - kss
        conc, good, pairs = concordance(test, name)
        table[name] = {"MAE": float(np.abs(err).mean()),
                       "skill": float(1 - (err ** 2).sum() / guess_sq) if guess_sq > 0 else float("nan"),
                       "within_concordance": conc, "pairs": pairs}
        print(f"   {name:18} error {table[name]['MAE']:.2f}  skill {table[name]['skill']:+.2f}  "
              f"within-person order {conc:.2f} ({good:g} of {pairs} pairs)")
    report["personal"] = table
    v1_pass = table["v1"]["MAE"] < table["guess"]["MAE"]
    h1_better = table["H1"]["MAE"] < table["v1"]["MAE"] and not (
        table["H1"]["within_concordance"] < table["v1"]["within_concordance"])
    print(f"   v1 beats guessing: {'PASS' if v1_pass else 'FAIL'};  H1 better than v1: {'yes' if h1_better else 'no'}\n")
    report["v1_beats_guessing"], report["h1_better_than_v1"] = bool(v1_pass), bool(h1_better)

    # 2. order beyond the clock: new same-person pairs where the later check-in was less tired
    pairs = []
    allrows = pd.concat([train.assign(new=False), test.assign(new=True)])
    for name in personal:
        allrows.loc[allrows.new, name] = test[name].to_numpy(float)
    for pid, g in teammates_all[teammates_all.index.isin(rows.index)].groupby("pid"):
        g = g.sort_values("timestamp")
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                a, b = g.iloc[i], g.iloc[j]
                if b.timestamp > cutoff and b.kss < a.kss:
                    pairs.append({name: bool(b[name] < a[name]) for name in ("v1", "H1", "v3", "clock (benchmark)")})
    print(f"2. Order beyond the clock: {len(pairs)} new pairs where the later check-in was less tired")
    beyond = {name: int(sum(p[name] for p in pairs)) for name in ("v1", "H1", "v3", "clock (benchmark)")}
    for name, n in beyond.items():
        print(f"   {name:18} got {n} of {len(pairs)} right")
    report["beyond_clock"] = {"pairs": len(pairs), "right": beyond}
    print()

    # 3. after sleep
    sleep = []
    for pid, g in teammates_all.groupby("pid"):
        g = g.sort_values("timestamp").reset_index(drop=True)
        for i in range(1, len(g)):
            gap = (g.timestamp[i] - g.timestamp[i - 1]).total_seconds() / 3600
            nap = g.nap_since_last_min[i] if "nap_since_last_min" in g else np.nan
            if g.timestamp[i] > cutoff and (gap >= GAP_HOURS or (pd.notna(nap) and nap >= 60)):
                before, after = g.iloc[i - 1], g.iloc[i]
                sleep.append({"pid": pid, "gap_h": round(gap, 1), "kss": (before.kss, after.kss),
                              **{m: (round(float(before[m]), 2), round(float(after[m]), 2))
                                 for m in ("v1", "H1", "v3", "ridge_fg5", "shipped_index")}})
    print(f"3. After a gap of {GAP_HOURS:g}+ hours (likely sleep): {len(sleep)} cases")
    moved = {}
    for m in ("v1", "H1", "v3", "ridge_fg5", "shipped_index"):
        same = [np.sign(s[m][1] - s[m][0]) == np.sign(s["kss"][1] - s["kss"][0]) for s in sleep if s["kss"][1] != s["kss"][0]]
        moved[m] = (int(sum(same)), len(same))
    for s in sleep:
        print(f"   {s['pid']} after {s['gap_h']} h: KSS {s['kss'][0]:g} -> {s['kss'][1]:g} | v1 {s['v1'][0]} -> {s['v1'][1]} | "
              f"ridge_fg5 {s['ridge_fg5'][0]} -> {s['ridge_fg5'][1]} | shipped index {s['shipped_index'][0]} -> {s['shipped_index'][1]}")
    for m, (n, of) in moved.items():
        print(f"   {m:14} moved the same way as the rating in {n} of {of}" + ("  PASS" if of and n > of / 2 else ""))
    report["after_sleep"] = {"cases": sleep, "moved_with_rating": moved}
    print()

    # 4. first try for strangers
    new_walkups = t[~t.participant_type.eq("panel") & (t.timestamp > cutoff) & t.kss.notna()]
    strangers = pd.concat([teammates_all[(teammates_all.timestamp > cutoff) & teammates_all.kss.notna()],
                           new_walkups.assign(ridge_fg5=ridge_fg5(new_walkups),
                                              shipped_index=shipped_index(new_walkups) if len(new_walkups) else [])])
    flat = FG5["intercept"]
    fg_mae = float(np.mean(np.abs(strangers.ridge_fg5 - strangers.kss)))
    flat_mae = float(np.mean(np.abs(flat - strangers.kss)))
    fg_rho, idx_rho = spearman(strangers.ridge_fg5, strangers.kss), spearman(strangers.shipped_index, strangers.kss)
    fg_sleep = moved["ridge_fg5"]
    fg_pass = fg_mae < flat_mae and fg_rho > 0 and fg_sleep[1] > 0 and fg_sleep[0] > fg_sleep[1] / 2
    print(f"4. First try, scored as strangers ({len(strangers)} new check-ins, {len(new_walkups)} of them walk-ups)")
    print(f"   shipped index: rank correlation with KSS {idx_rho:+.2f}"
          + ("  (backwards: tracks practice)" if idx_rho < 0 else ""))
    print(f"   ridge_fg5: error {fg_mae:.2f} vs {flat_mae:.2f} for always {flat:.2f}; rank correlation {fg_rho:+.2f}; "
          f"after sleep {fg_sleep[0]} of {fg_sleep[1]}  {'PASS' if fg_pass else 'FAIL'}\n")
    report["first_try"] = {"n": len(strangers), "walkups": len(new_walkups), "shipped_index_rho": idx_rho,
                           "ridge_fg5_mae": fg_mae, "flat_mae": flat_mae, "ridge_fg5_rho": fg_rho,
                           "ridge_fg5_pass": bool(fg_pass)}

    # 5. nap rule
    print(f"5. Nap rule (prediction {NAP_AT}+ vs rating {NAP_AT}+) on new check-ins")
    naps = {}
    for name in ("v1", "H1", "clock (benchmark)"):
        said, was = test[name] >= NAP_AT, test.kss >= NAP_AT
        naps[name] = {"caught": int((said & was).sum()), "missed": int((~said & was).sum()),
                      "false_alarms": int((said & ~was).sum()), "quiet_and_right": int((~said & ~was).sum())}
        print(f"   {name:18} caught {naps[name]['caught']}, missed {naps[name]['missed']}, "
              f"false alarms {naps[name]['false_alarms']}, correctly quiet {naps[name]['quiet_and_right']}")
    report["nap_rule"] = naps
    print()

    # 6. mistakes
    ksf = []
    for r in test.itertuples():
        s = sessions.get(r.checkin_id)
        if s is None:
            continue
        try:
            ev, _, meta = ks.replay(s)
            feats = ks.keystroke_features(ev, meta)
        except Exception as err:  # a malformed session shouldn't stop the test
            print(f"   (skipped one check-in: {type(err).__name__})")
            continue
        ksf.append({"pid": r.pid, "kss": r.kss, "err_per100": feats["err_per100"], "fix_rate": feats["fix_rate"]})
    m = pd.DataFrame(ksf)
    if len(m):
        multi = m[m.groupby("pid").pid.transform("size") >= 2]
        cent = lambda c: multi[c] - multi.groupby("pid")[c].transform("mean")
        rho_err = spearman(cent("err_per100"), cent("kss")) if len(multi) > 2 else float("nan")
        fix_median = float(m.fix_rate.median()) if m.fix_rate.notna().any() else float("nan")
        print(f"6. Mistakes: within-person correlation of errors made with KSS {rho_err:+.2f} "
              f"({'PASS' if rho_err < 0 else 'FAIL'}); median share of errors fixed {fix_median:.2f} "
              f"({'PASS' if fix_median >= 0.9 else 'FAIL'})")
        report["mistakes"] = {"rho_errors_kss": rho_err, "median_fix_rate": fix_median, "n": len(m)}
    return report


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    out, data = ANALYSIS / sys.argv[1] if not Path(sys.argv[1]).is_absolute() else Path(sys.argv[1]), Path(sys.argv[2])
    report = run(out, data)
    (out / "morning_test.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(f"\nSaved {out / 'morning_test.json'}")


if __name__ == "__main__":
    main()
