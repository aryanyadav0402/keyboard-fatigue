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


def tester_average(model: dict) -> dict:
    """The average tester: everyone's usual features, and their usual KSS for each phase."""
    people = list((model.get("people") or {}).values())
    kss = {}
    for phase in ("pre", "post"):
        values = [p["usual_kss_by_phase"][phase] for p in people if phase in p["usual_kss_by_phase"]]
        if values:
            kss[phase] = float(np.mean(values))
    features = {c: float(np.mean([p["usual_features"][c] for p in people])) for c in model["feature_order"]}
    return {"usual_features": features if people else None, "usual_kss_by_phase": kss, "people": len(people)}


def own_history(rows: list[dict], phase: str, cols: list[str]) -> tuple[dict | None, int]:
    """The average features of a user's earlier attempts with her model in this phase, and how many there were."""
    found = []
    for r in rows:
        shown = r.get("shown") if isinstance(r.get("shown"), dict) else {}
        f = shown.get("ananya_features")
        if (r.get("model") == LABEL and shown.get("phase") == phase and isinstance(f, dict)
                and all(isinstance(f.get(c), (int, float)) for c in cols)):
            found.append(f)
    if not found:
        return None, 0
    return {c: float(np.mean([f[c] for f in found])) for c in cols}, len(found)


def describe_ananya(model: dict) -> dict:
    """A short public summary for GET /api/models, shaped like describe_model's."""
    return {"name": model["name"], "kind": KIND, "note": model.get("note", ""), "created_at": model.get("created_at"),
            "trained_on": model.get("trained_on"), "first_guess_ready": False, "personal_ready": True,
            "teammates": len(model.get("people") or {}), "first_guess_follows_tiredness": False,
            "typing_only": model.get("key_tasks") == [0], "features": model["feature_order"]}


def score_ananya(body: dict, model: dict, store=None) -> dict:
    """Score one posted attempt (the short form or a whole session) with her model.

    Replies use the labels the demo page knows, with a note naming her model. A teammate is compared with
    their own baseline ("best-attempt model"). Anyone else is compared with the testers' average on their
    first try ("no-baseline model"), then with their own earlier tries in the same phase from `store`
    ("best-attempt model"). Either way they start from the testers' usual KSS for the time of day."""
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
    testers = tester_average(model)
    reply = {"scale": SCALE, "fatigue_index": None, "username": username, "rule": rule, "features": features,
             "scaled_features": None, "typical_for_this_rule": {}, "best": None, "best_reference": None,
             "phase": phase, "model_version": model["name"], "saved_as_training_data": False}

    # Whose usual typing to compare with, and which usual KSS to start from. The demo page asks no
    # sleepiness question, so anyone but a teammate starts from the testers' usual KSS for this phase.
    earlier, history_ok = 0, None
    if person is not None:
        basis, usual_features = "teammate", person["usual_features"]
        usual = person["usual_kss_by_phase"].get(phase, testers["usual_kss_by_phase"].get(phase))
    else:
        own = None
        if store is not None and key:
            try:
                own, earlier = own_history(store.history(key), phase, cols)
                history_ok = True
            except Exception:  # a database hiccup never stops scoring: use the testers' average
                history_ok = False
        basis, usual_features = ("own history", own) if own is not None else ("tester average", testers["usual_features"])
        usual = testers["usual_kss_by_phase"].get(phase)
    if usual_features is None or usual is None:
        return dict(reply, score=None, label=None, level=None, model="not enough data yet",
                    note=f"Ananya's model has no check-ins {PHASE_TEXT[phase]} to compare with yet.",
                    model_inputs={}, usual_kss=None, shift=None, baseline=None, history=None)

    X = np.array([[f[c] - usual_features[c] for c in cols] for f in wf])
    z = (X - np.array(model["scaler_mean"])) / np.array(model["scaler_scale"])
    shift = float((z @ np.array(model["coef"]) + model["intercept"]).mean())
    usual = float(usual)
    score = float(np.clip(usual + shift, 1, 9))
    rounded, label, level = kss_wording(score)
    saved = None
    if store is not None and key and history_ok is not False:
        # her features go in "shown" with no adj, so our own models never use these records as a baseline
        record = {"user_key": key, "model_version": model["name"], "model": LABEL, "score": score, "first_guess": None,
                  "rule": rule, "adj": {}, "shown": dict(features, ananya_features=means, phase=phase)}
        try:
            store.add(record)
            saved = True
        except Exception:  # a database hiccup never stops scoring
            saved = False
    moves = f"and your typing now moves that by {shift:+.1f}."
    if basis == "teammate":
        anchor = (f"You usually rated {usual:g}" if phase in person["usual_kss_by_phase"]
                  else f"Our testers usually rated {usual:.1f}")
        note = f"Ananya's model: compared with your usual typing from last night's check-ins. {anchor} {PHASE_TEXT[phase]}, {moves}"
    elif basis == "own history":
        tries = "try" if earlier == 1 else "tries"
        note = (f"Ananya's model: compared with your own {earlier} earlier {tries} here. Our testers usually rated "
                f"{usual:.1f} {PHASE_TEXT[phase]}, {moves}")
    else:
        note = (f"Ananya's model, first try: we don't know your usual typing yet, so this compares it with our "
                f"{testers['people']} testers' average. They usually rated {usual:.1f} {PHASE_TEXT[phase]}, {moves}"
                + (" Your next tries will be compared with your own." if key else ""))
    return dict(reply, score=rounded, label=label, level=level,
                model="no-baseline model" if basis == "tester average" else "best-attempt model", note=note,
                model_inputs={c: round(float(X[:, i].mean()), 6) for i, c in enumerate(cols)},
                usual_kss=usual, shift=round(shift, 3), baseline=basis,
                history=None if store is None else {"saved": bool(saved), "attempts": earlier + 1 if history_ok else None})
