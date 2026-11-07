"""Turn the loaded tables into one row of features per check-in.

Typing timings use character keys only, so Shift presses don't distort the rhythm, and gaps next
to a correction (Backspace or a wrong key) are left out. Task 1's rule rotates between visits
(capitalise vowels, dashes for spaces, and so on). "Rule" keys are the characters the rule changes,
such as capital vowels or dashes; "ordinary" keys are the ones it leaves alone. The rule cost is
the median gap next to a rule key divided by the median gap between ordinary keys.

Some rules are harder than others, so Task 1 features also get a rule-adjusted version (_adj),
measured against the panel's typical value for that rule. Rule cost and accuracy are divided by
it, so 1.0 means normal for that rule; the other features have it subtracted.

Processed versions (suffix _dev) subtract the person's average over their first two check-ins
(their baseline), so they read as "change from your own normal". kss_delta and hours_rel do the
same for the rating and the clock. The clock is hours awake when the app records a wake-up time,
and hours since the person's first check-in when it doesn't (the app's version 1.2 doesn't).
"""

import math
import re

import numpy as np
import pandas as pd

from load_data import MODIFIER_KEYS, Tables
from tasks import CAPITAL_VOWELS, score

PREFIX = {"rule": "vt_", "plain": "pt_", "free": "ft_"}
BASELINE_CHECKINS = 2
SLEEP_RESET_MIN = 60  # sleeping at least this long since the last check-in resets hours awake
MIN_REACTIONS = 3  # the app's 20-second reaction test gives only about 6-9 trials
LOWERCASE_AND_SPACE = frozenset("abcdefghijklmnopqrstuvwxyz ")

# Features that get a change-from-baseline (_dev) version, when present.
RELATIVE = [
    "vt_wpm", "vt_start_latency", "vt_gap_med", "vt_gap_cv", "vt_pause500_share", "vt_pause1s_share",
    "vt_hold_med", "vt_hold_iqr", "vt_rollover_share", "vt_accuracy", "vt_rule_cost", "vt_shift_lead_med",
    "pt_wpm", "pt_gap_med", "pt_gap_cv", "pt_pause1s_share", "pt_hold_med", "pt_accuracy",
    "ft_word_gap_med", "ft_pause1s_share", "ft_pause2s_share", "ft_longest_gap",
    "wpm_ratio", "gap_ratio", "cv_ratio", "acc_diff",
    "rt_median", "rt_speed", "rt_lapse_rate", "rt_slowest10", "rt_false_starts",
]
# Task 1 features that depend on which rule was shown, so they get a rule-adjusted (_adj) version.
TYPE_ADJUSTED = [c for c in RELATIVE if c.startswith("vt_")] + ["backspace_per100"]
RATIO_ADJUSTED = ("vt_rule_cost", "vt_accuracy")  # divided by the rule's typical value: 1.0 = normal
MIN_PANEL_PER_RULE = 3


def iqr(x) -> float:
    q75, q25 = np.percentile(x, [75, 25])
    return float(q75 - q25)


def rule_characters(source: str | None, expected: str | None) -> tuple[set, set]:
    """The characters the rule produces (such as capital vowels or dashes) and the ones it keeps."""
    if source and expected and len(source) == len(expected):
        changed = {e for s, e in zip(source, expected) if s != e}
        kept = {e for s, e in zip(source, expected) if s == e} - changed
        return changed, kept
    return set(CAPITAL_VOWELS), set(LOWERCASE_AND_SPACE)  # older files without a source: the vowel rule


def char_class(key: str, changed: set, kept: set) -> str:
    if key in changed:
        return "rule"
    if key in kept:
        return "ordinary"
    return "other"  # Backspace, wrong characters, and anything the sentence didn't ask for


def shift_lead(presses: pd.DataFrame, changed: set) -> float:
    """Median time from pressing Shift to pressing the capital the rule asked for."""
    keys, downs = presses.key.tolist(), presses.down_ms.to_numpy(float)
    leads = []
    for i, key in enumerate(keys):
        if key != "Shift":
            continue
        j = next((j for j in range(i + 1, len(keys)) if keys[j] != "Shift"), None)
        if j is not None and keys[j] in changed and keys[j].isupper():
            leads.append(downs[j] - downs[i])
    return float(np.median(leads)) if leads else math.nan


def typing_features(presses: pd.DataFrame, typed: str, expected: str | None, source: str | None = None,
                    rule: bool = False) -> dict:
    """Speed, rhythm, pauses, holds, errors and (Task 1 only) rule cost for one copy task."""
    p = presses.sort_values("down_ms")
    chars = p[~p.key.isin(MODIFIER_KEYS) & (p.key != "Enter")]
    if len(chars) < 5:
        return {}
    typed = typed.rstrip("\n")
    out = {"n_keys": len(chars), "n_backspace": int((chars.key == "Backspace").sum())}
    out["backspace_per100"] = 100 * out["n_backspace"] / len(chars)

    # Speed: typed characters / 5 per minute, from the first press to the last release.
    written = chars[chars.key != "Backspace"]
    last = written.up_ms.max() if written.up_ms.notna().any() else written.down_ms.max()
    minutes = (last - p.down_ms.min()) / 60000
    out["wpm"] = len(typed) / 5 / minutes if minutes > 0 else math.nan
    out["start_latency"] = float(p.down_ms.min())

    if rule:
        changed, kept = rule_characters(source, expected)
    else:
        changed, kept = set(), set(expected) if expected else set(LOWERCASE_AND_SPACE)
    keys = chars.key.tolist()
    downs, ups = chars.down_ms.to_numpy(float), chars.up_ms.to_numpy(float)
    classes = [char_class(k, changed, kept) for k in keys]
    ordinary, touching, every, rollover = [], [], [], []
    for i in range(1, len(keys)):
        a, b = classes[i - 1], classes[i]
        if "other" in (a, b):
            continue
        gap = downs[i] - downs[i - 1]
        every.append(gap)
        rollover.append(bool(downs[i] < ups[i - 1]))
        (touching if "rule" in (a, b) else ordinary).append(gap)

    if len(ordinary) >= 5:
        med = float(np.median(ordinary))
        out.update(gap_med=med, gap_iqr=iqr(ordinary), gap_cv=iqr(ordinary) / med)
    if len(every) >= 5:
        every = np.asarray(every)
        out.update(gap_all_med=float(np.median(every)), pause500_share=float(np.mean(every > 500)),
                   pause1s_share=float(np.mean(every > 1000)), longest_gap=float(every.max()),
                   rollover_share=float(np.mean(rollover)))

    letters = chars[chars.key.str.len().eq(1) & chars.key.str.isalpha()]
    holds = (letters.up_ms - letters.down_ms).dropna()
    holds = holds[(holds > 0) & (holds < 2000)]
    if len(holds) >= 5:
        out.update(hold_med=float(holds.median()), hold_iqr=iqr(holds))

    if expected:
        out.update({k: v for k, v in score(typed, expected).items() if k != "distance"})
    if rule:
        if len(touching) >= 3 and len(ordinary) >= 3:
            out["touch_gap_med"] = float(np.median(touching))
            out["ord_gap_med"] = float(np.median(ordinary))
            out["rule_cost"] = out["touch_gap_med"] / out["ord_gap_med"]
        out["shift_lead_med"] = shift_lead(p, changed)
        out["caps_lock_used"] = bool(p.caps_lock.any() or (p.key == "CapsLock").any())
    return out


def free_features(presses: pd.DataFrame) -> dict:
    """Typing speed inside words, pauses and corrections in a free-text task (app version 1.0 only)."""
    p = presses.sort_values("down_ms")
    chars = p[~p.key.isin(MODIFIER_KEYS)]
    if len(chars) < 10:
        return {}
    out = {"n_keys": len(chars), "n_backspace": int((chars.key == "Backspace").sum())}
    out["backspace_per100"] = 100 * out["n_backspace"] / len(chars)
    keys, downs = chars.key.tolist(), chars.down_ms.to_numpy(float)
    word_gaps, every = [], []
    for i in range(1, len(keys)):
        a, b = keys[i - 1], keys[i]
        if "Backspace" in (a, b):
            continue
        gap = downs[i] - downs[i - 1]
        every.append(gap)
        if len(a) == 1 and len(b) == 1 and a.isalpha() and b.isalpha():
            word_gaps.append(gap)
    if len(word_gaps) >= 10:
        out["word_gap_med"] = float(np.median(word_gaps))
    if len(every) >= 10:
        every = np.asarray(every)
        out.update(pause1s_share=float(np.mean(every > 1000)), pause2s_share=float(np.mean(every > 2000)),
                   longest_gap=float(every.max()))
    return out


def pvt_features(trials: pd.DataFrame, lapse_ms: float, false_start_ms: float, timeout_ms: float) -> dict:
    """Reaction-test summary from the main trials: median, speed (mean 1/RT), lapses, slowest 10%."""
    main = trials[trials.phase.astype(str) == "main"]
    if main.empty:
        return {}
    outcome = main.outcome.astype(str)
    rt = pd.to_numeric(main.rt_ms, errors="coerce")
    rt = rt.where(~outcome.str.contains("timeout"), rt.fillna(timeout_ms))
    responses = rt[outcome != "false_start"].dropna()
    early = responses < false_start_ms
    out = {"rt_false_starts": int((outcome == "false_start").sum() + early.sum())}
    rts = responses[~early]
    if len(rts) < MIN_REACTIONS:
        return out
    lapses = rts >= lapse_ms
    out.update(rt_trials=len(rts), rt_median=float(rts.median()), rt_speed=float((1000 / rts).mean()),
               rt_lapses=int(lapses.sum()), rt_lapse_rate=float(lapses.mean()),
               rt_slowest10=float(rts[rts >= rts.quantile(0.9)].mean()))
    return out


def reported_wake(timestamp: pd.Timestamp, wake_time) -> pd.Timestamp | None:
    """The most recent time at or before timestamp that matches the HH:MM someone reported."""
    if pd.isna(timestamp) or not isinstance(wake_time, str):
        return None
    match = re.match(r"\s*(\d{1,2}):(\d{2})", wake_time)
    if not match or int(match.group(1)) > 23 or int(match.group(2)) > 59:
        return None
    wake = timestamp.normalize() + pd.Timedelta(hours=int(match.group(1)), minutes=int(match.group(2)))
    return wake - pd.Timedelta(days=1) if wake > timestamp else wake


def hours_awake(timestamp: pd.Timestamp, wake_time) -> float:
    """Hours since the reported wake time, allowing for midnight in between."""
    wake = reported_wake(timestamp, wake_time)
    return math.nan if wake is None else (timestamp - wake).total_seconds() / 3600


def add_hours_awake(f: pd.DataFrame) -> pd.Series:
    """Hours awake per check-in, reset by sleep even if someone forgot to update their wake time.

    After sleeping for N minutes since the last check-in, we assume they fell asleep right after
    that check-in (the protocol has one just before sleeping), so they woke N minutes later.
    The later of that estimate and their reported wake time wins.
    """
    hours = pd.Series(np.nan, index=f.index)
    for _, g in f.groupby("person", sort=False):
        reset, prev = None, None
        for idx, ts, wake_time, slept in zip(g.index, g.timestamp, g.wake_time, g.nap_since_last_min):
            if prev is not None and slept >= SLEEP_RESET_MIN:
                reset = prev + pd.Timedelta(minutes=float(slept))
            options = [w for w in (reported_wake(ts, wake_time), reset) if w is not None and w <= ts]
            if options:
                hours[idx] = (ts - max(options)).total_seconds() / 3600
            prev = ts
    return hours


def add_clock(f: pd.DataFrame) -> pd.DataFrame:
    """The clock for the benchmark: hours awake for people with wake-up times, otherwise hours
    since their first check-in."""
    f["hours_since_first"] = (f.timestamp - f.groupby("person").timestamp.transform("min")).dt.total_seconds() / 3600
    has_wake = f.groupby("person").hours_awake.transform(lambda s: s.notna().any()).astype(bool)
    f["hours_clock"] = f.hours_awake.where(has_wake, f.hours_since_first)
    f["clock_basis"] = np.where(has_wake, "hours awake", "hours since first check-in")
    return f


def adjust_for_rule_type(f: pd.DataFrame) -> pd.DataFrame:
    """Add _adj versions of Task 1 features that don't depend on which rule was shown.

    The reference is the panel's typical (median) value for that rule type, so walk-ups and judges
    never shape it; a rule type with fewer than 3 panel check-ins falls back to every check-in of
    that type. Rule cost and accuracy are divided by it, so 1.0 means normal for that rule. The
    other features have it subtracted and the panel's overall median added back.
    """
    cols = [c for c in TYPE_ADJUSTED if c in f]
    if "question_type" not in f or not cols:
        return f
    panel = f.participant_type.eq("panel")
    new = {}
    for col in cols:
        typical = f.groupby("question_type")[col].median()
        on_panel = f[panel].groupby("question_type")[col].agg(["median", "count"])
        enough = on_panel.index[on_panel["count"] >= MIN_PANEL_PER_RULE]
        typical.loc[enough] = on_panel.loc[enough, "median"]
        reference = f.question_type.map(typical)
        if col in RATIO_ADJUSTED:
            new[col + "_adj"] = f[col] / reference.where(reference > 0)
        else:
            overall = f.loc[panel, col].median() if f.loc[panel, col].notna().any() else f[col].median()
            new[col + "_adj"] = f[col] - reference + overall
    return pd.concat([f, pd.DataFrame(new, index=f.index)], axis=1)


def add_relative(f: pd.DataFrame) -> pd.DataFrame:
    """Mark each panel person's first two usable check-ins as their baseline, and add _dev
    features: each (rule-adjusted) feature minus that person's average over those two check-ins."""
    panel = f.participant_type.eq("panel")
    typing_cols = [c for c in ("vt_wpm", "pt_wpm", "ft_n_keys") if c in f]
    has_typing = f[typing_cols].notna().any(axis=1) if typing_cols else pd.Series(False, index=f.index)
    usable = panel & f.kss.notna() & has_typing
    base_idx = f[usable].groupby("person").head(BASELINE_CHECKINS).index
    f["is_baseline"] = f.index.isin(base_idx)
    baseline = f.loc[base_idx].groupby("person")
    new = {}
    for col in [c for c in RELATIVE if c in f] + ["kss", "hours_clock"]:
        source = col + "_adj" if col + "_adj" in f else col
        normal = f.person.map(baseline[source].mean()).where(panel)
        if col == "kss":
            new["kss_base"], new["kss_delta"] = normal, f[col] - normal
        elif col == "hours_clock":
            new["hours_rel"] = f[col] - normal
        else:
            new[col + "_dev"] = f[source] - normal
    return pd.concat([f, pd.DataFrame(new, index=f.index)], axis=1)


ID_COLUMNS = ["checkin_id", "person", "participant_type", "timestamp", "checkin_n", "is_baseline",
              "question_type", "hours_awake", "hours_since_first", "hours_clock", "clock_basis",
              "nap_since_last_min", "kss"]
COMBINED = ["backspace_per100", "wpm_ratio", "gap_ratio", "cv_ratio", "acc_diff"]


def raw_table(f: pd.DataFrame) -> pd.DataFrame:
    """One row per check-in: who, when, the rating, and every feature as measured."""
    measured = [c for c in f if c.startswith(tuple(PREFIX.values()) + ("rt_",)) and not c.endswith(("_dev", "_adj"))]
    return f[[c for c in ID_COLUMNS if c in f] + measured + [c for c in COMBINED if c in f]]


def processed_table(f: pd.DataFrame) -> pd.DataFrame:
    """What the models use: each feature's change from the person's first two check-ins (_dev),
    backspaces per 100 keys adjusted for the rule type, and the rating and clock relative to baseline."""
    relative = [c for c in ("kss_base", "kss_delta", "hours_rel") if c in f]
    changes = [c for c in f if c.endswith("_dev")]
    backspaces = ["backspace_per100_adj"] if "backspace_per100_adj" in f else [c for c in ("backspace_per100",) if c in f]
    return f[[c for c in ID_COLUMNS if c in f] + relative + changes + backspaces]


def build_features(t: Tables) -> pd.DataFrame:
    """One row per check-in: session info, self-report, typing, reaction test and context."""
    if t.sessions.empty:
        return pd.DataFrame()
    sessions = t.sessions.set_index("checkin_id")
    rows = {cid: {} for cid in sessions.index}
    presses_by_unit = dict(tuple(t.keys.groupby(["checkin_id", "task_idx", "unit"]))) if len(t.keys) else {}

    for u in t.units.itertuples(index=False):
        if u.practice or u.task not in PREFIX or u.checkin_id not in rows:
            continue
        prefix, row = PREFIX[u.task], rows[u.checkin_id]
        row[prefix + "focus_losses"] = u.focus_losses
        if u.task == "rule":
            row["question_type"] = u.question_type
        if u.focus_losses > 0:
            continue  # they looked away mid-task, so its timings can't be trusted
        presses = presses_by_unit.get((u.checkin_id, u.task_idx, u.unit))
        if presses is None:
            continue
        if u.task == "free":
            feats = free_features(presses)
        else:
            feats = typing_features(presses, u.typed, u.expected, u.source, rule=u.task == "rule")
        row.update({prefix + k: v for k, v in feats.items()})

    for cid, trials in t.pvt.groupby("checkin_id") if len(t.pvt) else []:
        if cid in rows:
            s = sessions.loc[cid]
            rows[cid].update(pvt_features(trials, s.pvt_lapse_ms, s.pvt_false_start_ms, s.pvt_timeout_ms))

    f = sessions.join(pd.DataFrame.from_dict(rows, orient="index")).rename_axis("checkin_id").reset_index()

    back_cols = [c for c in f if c.endswith("_n_backspace")]
    key_cols = [c for c in f if c.endswith("_n_keys")]
    if key_cols:  # corrections across all typing tasks together, which is steadier than one task
        f["backspace_per100"] = 100 * f[back_cols].sum(axis=1, min_count=1) / f[key_cols].sum(axis=1, min_count=1)
    if {"vt_wpm", "pt_wpm"} <= set(f):  # load cost against a plain copy task, if the app ever has one
        f["wpm_ratio"] = f.vt_wpm / f.pt_wpm
        f["gap_ratio"] = f.get("vt_gap_all_med") / f.get("pt_gap_all_med")
        f["cv_ratio"] = f.get("vt_gap_cv") / f.get("pt_gap_cv")
        f["acc_diff"] = f.get("vt_accuracy") - f.get("pt_accuracy")

    f = f.sort_values(["person", "timestamp"]).reset_index(drop=True)
    f["checkin_n"] = f.groupby("person").cumcount() + 1
    f["log_checkin"] = np.log(f.checkin_n)
    f["hours_awake"] = add_hours_awake(f)
    f = add_clock(f)
    return add_relative(adjust_for_rule_type(f))
