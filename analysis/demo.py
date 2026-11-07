"""Score one demo attempt: a tiredness estimate and the typing features behind it.

The demo page posts the typing from a single sentence to serve.py's POST /api/score, with a
username. The first attempt gets a first guess from the no-baseline model. Each user's best attempt
so far (their most alert score) is kept in a small SQLite database, and later attempts are compared
with it: the model scores how much more tired the typing looks than at their best, and adds that
to their best score. Typing just like the best attempt gives the same score, and a more alert
attempt becomes the new best.

Only the best attempt's score and features are stored, never keystrokes, and none of it is used as
training data. The models are trained on the team's check-ins that serve.py has loaded. The hosted
API scores the same way from a frozen model file instead (frozen_model.py).

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
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from attempt import (NOTES, REFERENCE, SCALE, SHOWN, BadSession, check_typing, clean,  # noqa: F401
                     compact_to_session, compared_with_best, kss_wording, read_attempt)
from features import build_features
from first_guess import describe
from load_data import Tables
from model import display_models

DB_LOCK = threading.Lock()


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


def score_session(body: dict, train: Tables, db_path: str | Path | None = None) -> dict:
    """Score one posted attempt against the team's check-ins, comparing it with the user's best attempt."""
    parsed, username = read_attempt(body)

    def add(table: pd.DataFrame, rows: list) -> pd.DataFrame:
        return pd.concat([table, pd.DataFrame(rows)], ignore_index=True) if rows else table

    combined = Tables(sessions=add(train.sessions, [parsed["session"]]), units=add(train.units, parsed["units"]),
                      keys=add(train.keys, parsed["keys"]), pvt=add(train.pvt, parsed["pvt"]))
    f = build_features(combined)
    row = f.loc[f.checkin_id == parsed["session"]["checkin_id"]].iloc[0]
    check_typing(row, parsed)

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

    shown = {label: clean(row.get(col)) for col, label in SHOWN.items()}  # every key, null when not measured
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
    # How this attempt compares with the team's check-ins: only for people with no baseline. Anyone we
    # have a baseline for is compared only with themselves.
    index_info = None
    if model_name == "no-baseline model":
        panel = f[f.participant_type.eq("panel") & f.kss.notna()]
        cols = m["fallback_cols"]
        index_info = describe(m["fallback"], panel[cols].to_numpy(float), row[cols].to_numpy(float))
    same_rule = f[f.participant_type.eq("panel") & f.question_type.eq(row.question_type)]
    rounded, label, level = kss_wording(score)
    return {
        "score": rounded,
        "scale": SCALE,
        "label": label,
        "level": level,
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
