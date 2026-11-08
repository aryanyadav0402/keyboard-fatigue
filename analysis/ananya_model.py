"""Ananya's per-person tiredness model, hosted next to ours.

Her model compares a session's typing with the person's usual typing from the night's check-ins. It
takes 8 keystroke features over each half of the typing, subtracts the person's usual values, and
runs them through a standard scaler and a ridge regression. The average over the two halves is a shift,
which is added to the person's usual KSS for the time of day: pre-nap is 17:00 to 04:00 and post-nap
is after 04:00. The clock only picks which usual KSS to start from; it is never a model input. Only
people with a baseline can be scored. Their baselines are stored under scrambled keys
(frozen_model.teammate_key), so the model file holds no names.

The feature code is a port of her predict.py and gives the same numbers. Her current model was trained
on every key in a check-in, including the reaction test's presses, so `key_tasks` is null (every key).
A model trained on the typing task alone sets it to [0].
"""

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from attempt import SCALE, BadSession, compact_to_session, kss_wording, read_attempt
from frozen_model import FORMAT, teammate_key

KIND = "ananya"
LABEL = "ananya model"  # the internal label stored with each attempt
MODIFIERS = {"Shift", "Control", "Alt", "Meta", "CapsLock"}
REPEAT_FLAG = 32
MIN_KEYS_PER_WINDOW = 8
PHASE_TEXT = {"pre": "before the 04:00 nap (17:00 to 04:00)", "post": "after the 04:00 nap"}


def key_events(rows, tasks=None) -> pd.DataFrame:
    """Each key press with its time and hold time in seconds, without modifiers or auto-repeats."""
    down, events = {}, []
    for task, _unit, kind, t, code, key, flags in rows:
        if tasks is not None and task not in tasks:
            continue
        if kind == "d":
            if int(flags or 0) & REPEAT_FLAG:
                continue
            down[code] = (t, key)
        elif code in down:
            t0, k = down.pop(code)
            events.append((k, t0 / 1000, (t - t0) / 1000))
    e = pd.DataFrame(events, columns=["k", "pr", "ht"])
    return e[~e.k.isin(MODIFIERS) & (e.ht > 0) & (e.ht < 2)].sort_values("pr").reset_index(drop=True)


def window_features(e: pd.DataFrame, a: float, b: float) -> dict | None:
    """Her 8 features for the presses from a to b seconds, or None with fewer than 8 presses."""
    w = e[(e.pr >= a) & (e.pr < b)]
    if len(w) < MIN_KEYS_PER_WINDOW:
        return None
    iki = np.diff(w.pr.values)
    span = w.pr.max() - w.pr.min() + 1e-9
    with np.errstate(divide="ignore", invalid="ignore"):
        f = dict(log_iki_med=np.log(np.median(iki)), log_iki_sd=np.log(iki.std() + 1e-3), iki_cv=iki.std() / iki.mean(),
                 cps=len(w) / span, pauses_pm=(iki > 1).sum() / span * 60, pause_frac=iki[iki > 1].sum() / span,
                 log_dwell=np.log(w.ht.median()), bs_rate=(w.k == "Backspace").mean())
    f = {k: float(v) for k, v in f.items()}
    return f if all(math.isfinite(v) for v in f.values()) else None


def windows(rows, tasks=None, n: int = 2) -> list[dict]:
    """The features of each of n equal time windows of the typing that has enough presses."""
    e = key_events(rows, tasks)
    if e.empty:
        return []
    a0, b0 = e.pr.min(), e.pr.max()
    width = (b0 - a0) / n
    edges = [(a0 + i * width, a0 + (i + 1) * width if i < n - 1 else b0 + 1e-6) for i in range(n)]
    return [f for f in (window_features(e, a, b) for a, b in edges) if f]


def local_hour(session: dict) -> int:
    """The hour the session started, in the typist's own time zone."""
    clock = session.get("clock") if isinstance(session.get("clock"), dict) else {}
    offset = clock.get("tz_offset_min") if isinstance(clock.get("tz_offset_min"), (int, float)) else 0
    start = clock.get("start_epoch_ms")
    if isinstance(start, (int, float)):
        return (datetime.fromtimestamp(start / 1000, timezone.utc) + timedelta(minutes=offset)).hour
    iso = clock.get("start_local_iso")
    if isinstance(iso, str) and iso[11:13].isdigit():
        return int(iso[11:13])  # her predict.py's rule
    return (datetime.now(timezone.utc) + timedelta(minutes=offset)).hour


def phase_of(hour: int) -> str:
    return "pre" if hour >= 17 or hour < 4 else "post"


def export_ananya(package: Path, name: str, note: str = "", typing_only: bool = False) -> dict:
    """Her package (model_weights.json, person_baselines.json) as a model file with no names in it."""
    weights = json.loads((Path(package) / "model_weights.json").read_text())
    baselines = json.loads((Path(package) / "person_baselines.json").read_text())
    cols = list(weights["feature_order"])
    people = {}
    for pid, b in baselines.items():
        people[teammate_key(pid)] = {
            "usual_features": {c: float(b["usual_features"][c]) for c in cols},
            "usual_kss_by_phase": {p: float(v) for p, v in b["usual_kss_by_phase"].items()},
            "sessions": int(b.get("n_clean_sessions", 0)),
        }
    return {
        "format": FORMAT,
        "kind": KIND,
        "name": name,
        "note": note,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "feature_order": cols,
        "scaler_mean": [float(x) for x in weights["scaler_mean"]],
        "scaler_scale": [float(x) for x in weights["scaler_scale"]],
        "coef": [float(x) for x in weights["coef"]],
        "intercept": float(weights["intercept"]),
        "alpha": weights.get("alpha"),
        "windows": int(weights.get("windows", 2)),
        "key_tasks": [0] if typing_only else None,
        "people": people,
        "trained_on": {"people": len(people), "sessions": sum(p["sessions"] for p in people.values())},
    }


def describe_ananya(model: dict) -> dict:
    """A short public summary for GET /api/models, shaped like describe_model's."""
    return {"name": model["name"], "kind": KIND, "note": model.get("note", ""), "created_at": model.get("created_at"),
            "trained_on": model.get("trained_on"), "first_guess_ready": False, "personal_ready": True,
            "teammates": len(model.get("people") or {}), "first_guess_follows_tiredness": False,
            "typing_only": model.get("key_tasks") == [0], "features": model["feature_order"]}


def score_ananya(body: dict, model: dict, store=None) -> dict:
    """Score one posted attempt (the short form or a whole session) with her model.

    Replies use the labels the demo page knows: a teammate's score is a "best-attempt model" reply whose
    note names her model, and anyone without a baseline gets "not enough data yet"."""
    parsed, username = read_attempt(body)  # the same checks and username rules as our own models
    session = body if body.get("schema") == "typing-station" else compact_to_session(body, "DEMO")
    try:
        wf = windows(session["keys"], model.get("key_tasks"), model.get("windows", 2))
    except (KeyError, TypeError, ValueError):
        raise BadSession("each key should be [task, unit, 'd' or 'u', t, code, key, flags]") from None
    if not wf:
        raise BadSession("not enough typing to score: send at least 8 key presses in each half of the typing")
    cols = model["feature_order"]
    means = {c: float(np.mean([f[c] for f in wf])) for c in cols}
    features = {
        "gap_median_ms": round(math.exp(means["log_iki_med"]) * 1000, 1),
        "key_hold_ms": round(math.exp(means["log_dwell"]) * 1000, 1),
        "backspaces_per_100_keys": round(means["bs_rate"] * 100, 2),
        "keys_per_second": round(means["cps"], 2),
        "pauses_per_minute": round(means["pauses_pm"], 2),
        "pause_share": round(means["pause_frac"], 3),
        "rhythm_unevenness": round(means["iki_cv"], 3),
    }
    rule = str(parsed["units"][0].get("question_type") or "")
    key = teammate_key(username) if username else None
    person = (model.get("people") or {}).get(key) if key else None
    phase = phase_of(local_hour(session))
    reply = {"scale": SCALE, "fatigue_index": None, "username": username, "rule": rule, "features": features,
             "scaled_features": None, "typical_for_this_rule": {}, "best": None, "best_reference": None,
             "phase": phase, "model_version": model["name"], "saved_as_training_data": False}

    if person is None or phase not in person["usual_kss_by_phase"]:
        note = (f"Ananya's model has no check-ins from you {PHASE_TEXT[phase]}, so it can't tell what's usual for you now."
                if person is not None else
                f"Ananya's model compares your typing with your own usual typing from our night, so it can only "
                f"score the {len(model.get('people') or {})} people who did check-ins. Try our main model for a first guess.")
        return dict(reply, score=None, label=None, level=None, model="not enough data yet", note=note,
                    model_inputs={}, usual_kss=None, shift=None, history=None)

    X = np.array([[f[c] - person["usual_features"][c] for c in cols] for f in wf])
    z = (X - np.array(model["scaler_mean"])) / np.array(model["scaler_scale"])
    shift = float((z @ np.array(model["coef"]) + model["intercept"]).mean())
    usual = float(person["usual_kss_by_phase"][phase])
    score = float(np.clip(usual + shift, 1, 9))
    rounded, label, level = kss_wording(score)
    saved = None
    if store is not None:
        record = {"user_key": key, "model_version": model["name"], "model": LABEL, "score": score,
                  "first_guess": None, "rule": rule, "adj": {}, "shown": features}
        try:
            store.add(record)
            saved = True
        except Exception:  # a database hiccup never stops scoring
            saved = False
    return dict(reply, score=rounded, label=label, level=level, model="best-attempt model",
                note=(f"Ananya's model: compared with your usual typing from last night's check-ins. You usually rated "
                      f"{usual:g} {PHASE_TEXT[phase]}, and your typing now moves that by {shift:+.1f}."),
                model_inputs={c: round(float(X[:, i].mean()), 6) for i, c in enumerate(cols)},
                usual_kss=usual, shift=round(shift, 3),
                history=None if store is None else {"saved": bool(saved), "attempts": None})
