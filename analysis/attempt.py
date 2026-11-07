"""One demo attempt: reading what the page posts, and the pieces of the reply both scorers share.

Used by demo.py (the local server, which refits on the team's check-ins) and by frozen_model.py
(the hosted API, which only has a frozen model file). Nothing here needs scikit-learn.
"""

import uuid
from datetime import datetime

import numpy as np
import pandas as pd

from load_data import normalise_name, parse_session
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
SCALE = "KSS 1-9: 1 = extremely alert, 9 = fighting sleep"
# The features the demo shows, with friendlier names.
SHOWN = {
    "vt_wpm": "wpm", "vt_start_latency": "start_latency_ms", "vt_rule_cost": "rule_cost",
    "vt_gap_med": "gap_median_ms", "vt_gap_cv": "rhythm_unevenness", "vt_pause1s_share": "long_pause_share",
    "vt_hold_med": "key_hold_ms", "vt_accuracy": "accuracy", "vt_missed_capitals": "missed_capitals",
    "vt_wrong_capitals": "wrong_capitals", "vt_other_errors": "other_errors",
    "backspace_per100": "backspaces_per_100_keys", "vt_shift_lead_med": "shift_lead_ms",
    "rt_median": "reaction_median_ms", "rt_speed": "reaction_speed",
}
# Kept for the best attempt and compared on later ones. They're adjusted for the rule type, so an
# attempt with one rule can be compared with a best attempt that had another.
REFERENCE = ["vt_rule_cost_adj", "vt_accuracy_adj", "vt_gap_cv_adj", "vt_pause1s_share_adj",
             "vt_hold_med_adj", "backspace_per100_adj"]


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


def read_attempt(body) -> tuple[dict, str | None]:
    """Parse a posted attempt (the short form or a whole typing-station session) and its username."""
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
        return parse_session(session, "demo"), username
    except (KeyError, TypeError, ValueError) as err:
        raise BadSession(f"couldn't read the session ({err})") from None


def check_typing(row: pd.Series, parsed: dict) -> None:
    """Refuse attempts with too little typing, or keys that don't cover what was typed."""
    if pd.isna(row.get("vt_wpm")):
        raise BadSession("not enough typing to score: send at least 5 key presses from the sentence")
    typed_chars = len(str(parsed["units"][0].get("typed") or "").rstrip("\n"))
    if row.get("vt_n_keys", 0) < 0.6 * typed_chars:
        raise BadSession(f"the key events cover only {int(row.get('vt_n_keys', 0))} of the {typed_chars} typed "
                         "characters: send every key press and release from the sentence")


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


def kss_wording(score: float | None) -> tuple[float | None, str | None, str | None]:
    """The rounded score, its KSS label, and a level: alert (5 or less), getting tired (6) or tired."""
    if score is None:
        return None, None, None
    kss = int(np.clip(round(score), 1, 9))
    return round(score, 1), KSS_LABELS[kss], "alert" if kss <= 5 else "getting tired" if kss == 6 else "tired"
