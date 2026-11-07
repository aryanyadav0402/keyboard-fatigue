"""Score one demo attempt: a tiredness estimate and the typing features behind it.

The demo page posts the typing from a single sentence to serve.py's POST /api/score, with a
username. The first attempt gets a first guess from the no-baseline model. Each user's best attempt
so far (their most alert score) is kept in a small SQLite database, and later attempts are compared
with it: the model scores how much more tired the typing looks than at their best, and adds that
to their best score. Typing just like the best attempt gives the same score, and a more alert
attempt becomes the new best.

Only the best attempt's score and features are stored, never keystrokes, and none of it is used as
training data. The models are trained on the team's check-ins that serve.py has loaded.

The body can be a whole typing-station session, or this shorter form:

    {
      "username": "zoe",                     # optional; without it nothing is stored or compared
      "sentence": "a curious raccoon found a shiny golden key",
      "rule": "capitalize_vowels",           # optional; this is the default
      "typed": "A cUrIOUs rAccOOn fOUnd A shIny gOldEn kEy",
      "shown_at": 1532.4,                    # performance.now() when the sentence appeared
      "keys": [{"type": "down", "key": "A", "code": "KeyA", "t": 2011.7},
               {"type": "up", "key": "A", "code": "KeyA", "t": 2090.3}],
      "reaction_times_ms": [312, 290, 305]   # optional
    }

Key times ("t") are the browser's event.timeStamp, on the same clock as performance.now().
"""

import json
import sqlite3
import threading
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from features import build_features
from first_guess import describe
from load_data import Tables, normalise_name, parse_session
from model import display_models
from tasks import apply_rule

KSS_LABELS = {1: "Extremely alert", 2: "Very alert", 3: "Alert", 4: "Rather alert",
              5: "Neither alert nor sleepy", 6: "Some signs of sleepiness",
              7: "Sleepy, but no effort to stay awake", 8: "Sleepy, some effort to stay awake",
              9: "Very sleepy, fighting sleep"}
NOTES = {
    "best-attempt model": "Compared with your best attempt so far, which scored {best:.1f}.",
    "no-baseline model": ("A first guess: we don't know your normal typing yet, so treat it as rough. "
                          "Your next attempts will be compared with your best one."),
    "not enough data yet": "The model needs more check-ins from the team before it can score anyone.",
}
# The features the demo shows, with friendlier names.
SHOWN = {
    "vt_wpm": "wpm", "vt_start_latency": "start_latency_ms", "vt_rule_cost": "rule_cost",
    "vt_gap_med": "gap_median_ms", "vt_gap_cv": "rhythm_unevenness", "vt_pause1s_share": "long_pause_share",
    "vt_hold_med": "key_hold_ms", "vt_accuracy": "accuracy", "vt_missed_capitals": "missed_capitals",
    "vt_wrong_capitals": "wrong_capitals", "vt_other_errors": "other_errors",
    "backspace_per100": "backspaces_per_100_keys", "vt_shift_lead_med": "shift_lead_ms",
    "rt_median": "reaction_median_ms", "rt_speed": "reaction_speed",
}
# Stored for the best attempt and compared on later ones. They're adjusted for the rule type, so an
# attempt with one rule can be compared with a best attempt that had another.
REFERENCE = ["vt_rule_cost_adj", "vt_accuracy_adj", "vt_gap_cv_adj", "vt_pause1s_share_adj",
             "vt_hold_med_adj", "backspace_per100_adj"]
DB_LOCK = threading.Lock()


class BadSession(ValueError):
    """The posted session can't be scored; the message says why."""


def clean(value):
    """Make one value JSON-friendly: NaN becomes null, numpy numbers become plain ones, times become text."""
    if isinstance(value, str):
        return value
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return round(float(value), 3)
    return value


def open_db(path: str | Path) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("""CREATE TABLE IF NOT EXISTS best_attempts (
        username TEXT PRIMARY KEY, score REAL NOT NULL, rule TEXT, features TEXT NOT NULL,
        attempts INTEGER NOT NULL, best_at TEXT NOT NULL, updated_at TEXT NOT NULL)""")
    return conn


def get_best(db_path: str | Path | None, username: str | None) -> dict | None:
    """The stored best attempt for a username, or None."""
    if not db_path or not username or not Path(db_path).exists():
        return None
    conn = open_db(db_path)
    try:
        row = conn.execute("SELECT score, rule, features, attempts, best_at, updated_at FROM best_attempts "
                           "WHERE username = ?", (username,)).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return {"username": username, "score": row[0], "rule": row[1], "features": json.loads(row[2]),
            "attempts": row[3], "best_at": row[4], "updated_at": row[5]}


def record_attempt(db_path: str | Path, username: str, score: float, rule: str, features: dict) -> dict:
    """Count one attempt, and keep it as the best if it's the most alert so far."""
    now = datetime.now().isoformat(timespec="seconds")
    with DB_LOCK:
        conn = open_db(db_path)
        try:
            with conn:  # one transaction
                row = conn.execute("SELECT score FROM best_attempts WHERE username = ?", (username,)).fetchone()
                new_best = row is None or score < row[0]
                if row is None:
                    conn.execute("INSERT INTO best_attempts VALUES (?, ?, ?, ?, 1, ?, ?)",
                                 (username, score, rule, json.dumps(features), now, now))
                elif new_best:
                    conn.execute("UPDATE best_attempts SET score = ?, rule = ?, features = ?, attempts = attempts + 1, "
                                 "best_at = ?, updated_at = ? WHERE username = ?",
                                 (score, rule, json.dumps(features), now, now, username))
                else:
                    conn.execute("UPDATE best_attempts SET attempts = attempts + 1, updated_at = ? WHERE username = ?",
                                 (now, username))
        finally:
            conn.close()
    best = get_best(db_path, username)
    return {"score": round(best["score"], 1), "attempts": best["attempts"], "new_best": new_best,
            "best_at": best["best_at"]}


def compact_to_session(body: dict, person: str) -> dict:
    """Turn the short demo form into a session shaped like the check-in app's."""
    sentence = body.get("sentence") or body.get("source")
    if not isinstance(sentence, str) or not sentence.strip():
        raise BadSession("'sentence' is missing")
    if not isinstance(body.get("keys"), list) or not body["keys"]:
        raise BadSession("'keys' is missing or empty")
    rule = body.get("rule") or "capitalize_vowels"
    expected = body.get("expected") or apply_rule(rule, sentence)
    if expected is None:
        raise BadSession(f"unknown rule '{rule}'; send 'expected' with the correct answer")
    keys = []
    for k in body["keys"]:
        try:
            kind = "d" if str(k["type"]).lower() in ("down", "keydown", "d") else "u"
            keys.append([0, 0, kind, float(k["t"]), str(k.get("code", "")), str(k["key"]), 32 if k.get("repeat") else 0])
        except (KeyError, TypeError, ValueError):
            raise BadSession("each key needs 'type' (down or up), 'key' and 't'") from None
    times = [k[3] for k in keys]
    shown = float(body["shown_at"]) if body.get("shown_at") is not None else min(times) - 1
    try:
        reactions = [float(x) for x in body.get("reaction_times_ms") or []]
    except (TypeError, ValueError):
        raise BadSession("'reaction_times_ms' should be a list of numbers") from None
    return {
        "schema": "typing-station", "session_id": f"demo-{uuid.uuid4()}", "status": "final",
        "completed": True, "test_mode": False, "participant_id": person, "cohort": "walkup",
        "clock": {"start_local_iso": datetime.now().isoformat(timespec="seconds")},
        "self_report": {},
        "tasks": [{"idx": 0, "type": "transform", "name": "transform_typing", "t_start": shown, "t_end": max(times)}],
        "units": [{"task": 0, "unit": 0, "kind": "transform", "question_type": rule, "source": sentence,
                   "expected": expected, "typed": str(body.get("typed") or ""), "t_shown": shown, "t_end": max(times)}],
        "keys": keys,
        "pvt": [["main", i + 1, None, None, None, rt, "lapse" if rt >= 355 else "valid", "Space"]
                for i, rt in enumerate(reactions)],
        "ui": [],
    }


def compared_with_best(row: pd.Series, best_features: dict, typing: list[str]) -> tuple[list, list]:
    """The panel model's inputs for this attempt against the best one, and for an exact repeat of the best."""
    attempt, repeat = [], []
    for col in typing:
        if col.endswith("_dev"):  # change from the best attempt, like the panel's change from baseline
            reference = best_features.get(col[: -len("_dev")] + "_adj")
            value = row.get(col[: -len("_dev")] + "_adj")
            missing = reference is None or pd.isna(value)
            attempt.append(np.nan if missing else float(value) - reference)
            repeat.append(np.nan if missing else 0.0)
        else:  # used as it is, such as backspaces per 100 keys
            value, reference = row.get(col), best_features.get(col)
            missing = reference is None or pd.isna(value)
            attempt.append(np.nan if missing else float(value))
            repeat.append(np.nan if missing else reference)
    # A feature missing from either attempt is missing from both, so it can't shift the score.
    return attempt, repeat


def score_session(body: dict, train: Tables, db_path: str | Path | None = None) -> dict:
    """Score one posted attempt against the team's check-ins, comparing it with the user's best attempt."""
    if not isinstance(body, dict):
        raise BadSession("send a JSON object")
    # The short form sends "username"; a whole session carries it as participant_id (or metadata.username).
    metadata = body.get("metadata") if isinstance(body.get("metadata"), dict) else {}
    given = str(body.get("username") or body.get("name") or body.get("participant_id")
                or metadata.get("username") or "").strip()
    username = normalise_name(given) or None
    person = f"DEMO:{username or uuid.uuid4()}"  # kept apart from the team's own IDs
    if body.get("schema") == "typing-station":
        session = dict(body, session_id=f"demo-{uuid.uuid4()}", status="final", test_mode=False,
                       cohort="walkup", participant_id=person, self_report={})
    else:
        session = compact_to_session(body, person)
    try:
        parsed = parse_session(session, "demo")
    except (KeyError, TypeError, ValueError) as err:
        raise BadSession(f"couldn't read the session ({err})") from None

    def add(table: pd.DataFrame, rows: list) -> pd.DataFrame:
        return pd.concat([table, pd.DataFrame(rows)], ignore_index=True) if rows else table

    combined = Tables(sessions=add(train.sessions, [parsed["session"]]), units=add(train.units, parsed["units"]),
                      keys=add(train.keys, parsed["keys"]), pvt=add(train.pvt, parsed["pvt"]))
    f = build_features(combined)
    row = f.loc[f.checkin_id == parsed["session"]["checkin_id"]].iloc[0]
    if pd.isna(row.get("vt_wpm")):
        raise BadSession("not enough typing to score: send at least 5 key presses from the sentence")
    typed_chars = len(str(parsed["units"][0].get("typed") or "").rstrip("\n"))
    if row.get("vt_n_keys", 0) < 0.6 * typed_chars:
        raise BadSession(f"the key events cover only {int(row.get('vt_n_keys', 0))} of the {typed_chars} typed "
                         "characters: send every key press and release from the sentence")

    m = display_models(f)
    best = get_best(db_path, username)
    score, model_name, inputs = None, "not enough data yet", {}
    if best is not None and m["main"] is not None:
        attempt, repeat = compared_with_best(row, best["features"].get("adj", {}), m["typing"])
        change = m["main"].predict(np.array([attempt]))[0] - m["main"].predict(np.array([repeat]))[0]
        score, model_name = float(np.clip(best["score"] + change, 1, 9)), "best-attempt model"
        inputs = dict(zip(m["typing"], attempt))
    elif m["fallback"] is not None:
        X = row[m["fallback_cols"]].to_numpy(float).reshape(1, -1)
        score, model_name = float(np.clip(m["fallback"].predict(X)[0], 1, 9)), "no-baseline model"
        inputs = dict(zip(m["fallback_cols"], X[0]))

    shown = {label: clean(row.get(col)) for col, label in SHOWN.items() if col in f}
    best_info = None
    if username and db_path and score is not None:
        # unrounded, so repeating the best attempt exactly gives exactly the best score
        reference = {c: float(row[c]) if pd.notna(row.get(c)) else None for c in REFERENCE if c in f}
        best_info = record_attempt(db_path, username, score, row.question_type, {"adj": reference, "raw": shown})
    scaled = None
    if best is not None:
        best_raw = best["features"].get("raw", {})
        scaled = {label: (round(value / best_raw[label], 3) if value is not None and best_raw.get(label) else None)
                  for label, value in shown.items()}
    index_info = None  # how this attempt compares with the panel's check-ins, shown to people with no baseline
    if m["fallback"] is not None:
        panel = f[f.participant_type.eq("panel") & f.kss.notna()]
        cols = m["fallback_cols"]
        index_info = describe(m["fallback"], panel[cols].to_numpy(float), row[cols].to_numpy(float))
    same_rule = f[f.participant_type.eq("panel") & f.question_type.eq(row.question_type)]
    rounded = None if score is None else round(score, 1)
    kss = None if score is None else int(np.clip(round(score), 1, 9))
    return {
        "score": rounded,
        "scale": "KSS 1-9: 1 = extremely alert, 9 = fighting sleep",
        "label": None if kss is None else KSS_LABELS[kss],
        "level": None if kss is None else "alert" if kss <= 5 else "getting tired" if kss == 6 else "tired",
        "model": model_name,
        "note": NOTES[model_name].format(best=best["score"] if best else 0),
        "fatigue_index": index_info,
        "username": username,
        "rule": row.question_type,
        "features": shown,
        "scaled_features": scaled,
        "typical_for_this_rule": {label: clean(same_rule[col].median()) for col, label in SHOWN.items()
                                  if col in same_rule and same_rule[col].notna().any()},
        "model_inputs": {k: clean(v) for k, v in inputs.items()},
        "best": best_info,
        "saved_as_training_data": False,
    }
