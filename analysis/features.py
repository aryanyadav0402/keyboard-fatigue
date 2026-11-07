"""Turn the loaded tables into one row of features per check-in.

Typing timings use character keys only (letters and space), so Shift presses don't distort the
rhythm, and gaps next to a correction (Backspace or a wrong key) are left out. In the vowel task,
"ordinary" gaps are between two lowercase letters or spaces, and the rule cost is the median gap
next to a capital divided by the median ordinary gap.

Person-relative versions (suffix _z) subtract the person's mean over their first two check-ins
(their baseline) and divide by the baseline spread pooled across the panel. kss_delta and
hours_rel are the same idea, without the division.
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

# Features that get a person-relative (_z) version, when present.
RELATIVE = [
    "vt_wpm", "vt_start_latency", "vt_gap_med", "vt_gap_cv", "vt_pause500_share", "vt_pause1s_share",
    "vt_hold_med", "vt_hold_iqr", "vt_rollover_share", "vt_accuracy", "vt_rule_cost", "vt_shift_lead_med",
    "pt_wpm", "pt_gap_med", "pt_gap_cv", "pt_pause1s_share", "pt_hold_med", "pt_accuracy",
    "ft_word_gap_med", "ft_pause1s_share", "ft_pause2s_share", "ft_longest_gap",
    "backspace_per100", "wpm_ratio", "gap_ratio", "cv_ratio", "acc_diff",
    "rt_median", "rt_speed", "rt_lapse_rate", "rt_slowest10", "rt_false_starts",
]


def iqr(x) -> float:
    q75, q25 = np.percentile(x, [75, 25])
    return float(q75 - q25)


def char_class(key: str, rule: bool) -> str:
    if rule and key in CAPITAL_VOWELS:
        return "capital"
    if key == " " or (len(key) == 1 and key.isalpha() and key.islower()):
        return "ordinary"
    return "other"  # Backspace, digits, punctuation and wrong capitals


def shift_lead(presses: pd.DataFrame) -> float:
    """Median time from pressing Shift to pressing the capital vowel that follows it."""
    keys, downs = presses.key.tolist(), presses.down_ms.to_numpy(float)
    leads = []
    for i, key in enumerate(keys):
        if key != "Shift":
            continue
        j = next((j for j in range(i + 1, len(keys)) if keys[j] != "Shift"), None)
        if j is not None and keys[j] in CAPITAL_VOWELS:
            leads.append(downs[j] - downs[i])
    return float(np.median(leads)) if leads else math.nan


def typing_features(presses: pd.DataFrame, typed: str, expected: str | None, rule: bool) -> dict:
    """Speed, rhythm, pauses, holds, errors and (vowel task only) rule cost for one copy task."""
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

    keys = chars.key.tolist()
    downs, ups = chars.down_ms.to_numpy(float), chars.up_ms.to_numpy(float)
    classes = [char_class(k, rule) for k in keys]
    ordinary, touching, every, rollover = [], [], [], []
    for i in range(1, len(keys)):
        a, b = classes[i - 1], classes[i]
        if "other" in (a, b):
            continue
        gap = downs[i] - downs[i - 1]
        every.append(gap)
        rollover.append(bool(downs[i] < ups[i - 1]))
        (touching if "capital" in (a, b) else ordinary).append(gap)

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
        out["shift_lead_med"] = shift_lead(p)
        out["caps_lock_used"] = bool(p.caps_lock.any() or (p.key == "CapsLock").any())
    return out


def free_features(presses: pd.DataFrame) -> dict:
    """Typing speed inside words, pauses and corrections in the free-text task."""
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
    if len(rts) < 10:
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


def add_relative(f: pd.DataFrame) -> pd.DataFrame:
    """Mark each panel person's first two usable check-ins as baseline and add _z features."""
    panel = f.participant_type.eq("panel")
    typing_cols = [c for c in ("vt_wpm", "pt_wpm", "ft_n_keys") if c in f]
    has_typing = f[typing_cols].notna().any(axis=1) if typing_cols else pd.Series(False, index=f.index)
    usable = panel & f.kss.notna() & has_typing
    base_idx = f[usable].groupby("person").head(BASELINE_CHECKINS).index
    f["is_baseline"] = f.index.isin(base_idx)
    base = f.loc[base_idx]
    new = {}
    for col in [c for c in RELATIVE if c in f] + ["kss", "hours_awake"]:
        rel = f[col] - f.person.map(base.groupby("person")[col].mean())
        if col == "kss":
            new["kss_base"] = f.person.map(base.groupby("person")[col].mean())
            new["kss_delta"] = rel
            continue
        if col == "hours_awake":
            new["hours_rel"] = rel
            continue
        sd = math.sqrt(base.groupby("person")[col].var().mean())  # pooled within-person spread
        if not np.isfinite(sd) or sd < 1e-9:
            sd = f.loc[panel, col].std()
        if not np.isfinite(sd) or sd < 1e-9:
            sd = 1.0
        new[col + "_z"] = (rel / sd).where(panel)
    return pd.concat([f, pd.DataFrame(new, index=f.index)], axis=1)


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
        if u.focus_losses > 0:
            continue  # they looked away mid-task, so its timings can't be trusted
        presses = presses_by_unit.get((u.checkin_id, u.task_idx, u.unit))
        if presses is None:
            continue
        if u.task == "free":
            feats = free_features(presses)
        else:
            feats = typing_features(presses, u.typed, u.expected, rule=u.task == "rule")
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
    if {"vt_wpm", "pt_wpm"} <= set(f):  # load cost against a plain copy task, once the app has one
        f["wpm_ratio"] = f.vt_wpm / f.pt_wpm
        f["gap_ratio"] = f.get("vt_gap_all_med") / f.get("pt_gap_all_med")
        f["cv_ratio"] = f.get("vt_gap_cv") / f.get("pt_gap_cv")
        f["acc_diff"] = f.get("vt_accuracy") - f.get("pt_accuracy")

    f = f.sort_values(["person", "timestamp"]).reset_index(drop=True)
    f["checkin_n"] = f.groupby("person").cumcount() + 1
    f["log_checkin"] = np.log(f.checkin_n)
    f["hours_awake"] = add_hours_awake(f)
    return add_relative(f)
