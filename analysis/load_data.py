"""Read check-in files from the typing-station app into tidy tables.

Each check-in is one JSON file. load_sessions() reads every *.json under a folder and returns:

  sessions  one row per check-in: who, when, the self-report and data-quality flags
  units     one row per typing task: the sentence shown, what was typed, and focus losses
  keys      one row per key press, with press and release times in ms since the task appeared
  pvt       one row per reaction-test trial
  problems  messages about files or answers that look wrong
"""

import json
import math
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from tasks import apply_rule

# Cohorts that make up the overnight panel. Every other cohort is a one-off (the held-out test set).
PANEL_COHORTS = {"teammate", "panel", "overnight"}
MODIFIER_KEYS = {"Shift", "Control", "Alt", "AltGraph", "Meta", "CapsLock", "Fn", "OS"}
SHIFT_FLAG, CAPS_LOCK_FLAG, REPEAT_FLAG = 1, 16, 32
KEY_FIELDS = ["task", "unit", "type", "t", "code", "key", "flags"]  # the app's order for each key event
# typing-station unit kinds, mapped onto our task names
TASK_NAMES = {"transform": "rule", "copy": "plain", "plain": "plain", "free": "free"}
SELF_REPORT = ["kss", "mental_fatigue", "motivation", "wake_time", "sleep_last_night_h",
               "nap_since_last_min", "caffeine_last3h", "caffeine_min_ago", "posture"]


@dataclass
class Tables:
    sessions: pd.DataFrame
    units: pd.DataFrame
    keys: pd.DataFrame
    pvt: pd.DataFrame
    problems: list[str] = field(default_factory=list)
    skipped_test_runs: int = 0
    skipped_partial: int = 0


def number(value) -> float:
    """Read numbers like 2, "2", "2.5" or "2+" as floats; anything else becomes NaN."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    match = re.match(r"\s*(\d+(?:\.\d+)?)", str(value)) if value is not None else None
    return float(match.group(1)) if match else math.nan


def pair_presses(events: list) -> list[dict]:
    """Turn key-down and key-up events into one row per press, skipping auto-repeats."""
    presses, held = [], {}
    for task, unit, kind, t, code, key, flags in events:
        flags = int(flags or 0)
        if kind == "d":
            if flags & REPEAT_FLAG:
                continue
            presses.append({"task_idx": task, "unit": unit, "key": key, "code": code, "down": t,
                            "up": math.nan, "caps_lock": bool(flags & CAPS_LOCK_FLAG)})
            held[(task, code)] = len(presses) - 1
        elif kind == "u":
            i = held.pop((task, code), None)
            if i is not None:
                presses[i]["up"] = t
    return presses


def focus_losses(ui: list, start: float, end: float | None) -> int:
    """Count blur and hidden-tab events between a task's start and end."""
    end = math.inf if end is None else end
    return sum(1 for t, event, _ in ui
               if start <= t <= end and ("blur" in str(event) or "hidden" in str(event)))


def normalise_name(name: str) -> str:
    """The check-in app's ID for a typed name: accents dropped, spaces to dashes, upper case
    (Zoë O'Neil -> ZOE-ONEIL)."""
    plain = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    plain = re.sub(r"[^A-Za-z0-9 \-]", "", plain).strip()
    return re.sub(r"\s+", "-", plain).upper()


def read_session(path: Path) -> dict:
    return parse_session(json.loads(path.read_text()), path.name)


def parse_session(d: dict, source_name: str = "session") -> dict:
    """One session (as the app saves it) as rows for the session, unit, key and reaction-test tables."""
    if d.get("schema") != "typing-station":
        raise ValueError("not a typing-station session file")
    cid = d["session_id"]
    clock, report, config = d.get("clock") or {}, d.get("self_report") or {}, d.get("config") or {}
    pvt_config = config.get("pvt") or {}
    timestamp = pd.Timestamp(clock["start_local_iso"]) if clock.get("start_local_iso") else pd.NaT
    if pd.notna(timestamp) and timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)  # keep local wall-clock time
    cohort = str(d.get("cohort") or "")
    session = {
        "checkin_id": cid,
        "person": str(d.get("participant_id") or "").strip(),
        "cohort": cohort,
        "participant_type": "panel" if cohort.lower() in PANEL_COHORTS else "oneoff",
        "file": source_name,
        "status": str(d.get("status") or "final"),
        "timestamp": timestamp,
        "epoch_ms": clock.get("start_epoch_ms"),
        "test_mode": bool(d.get("test_mode")),
        "aborted": bool(d.get("aborted")),
        "completed": bool(d.get("completed")),
        "pvt_lapse_ms": number(pvt_config.get("lapse_ms", 355)),
        "pvt_false_start_ms": number(pvt_config.get("false_start_ms", 100)),
        "pvt_timeout_ms": number(pvt_config.get("timeout_ms", 10000)),
    }
    for name in SELF_REPORT:
        value = report.get(name)
        session[name] = value if name in ("wake_time", "posture") else number(value)

    tasks = {t["idx"]: t for t in d.get("tasks") or []}
    ui = d.get("ui") or []
    units, problems = [], []
    for u in d.get("units") or []:
        task = tasks.get(u["task"], {})
        name = TASK_NAMES.get(u.get("kind"), str(u.get("kind")))
        source, expected = u.get("source"), u.get("expected")
        # Task 1's rule rotates (vowels, dashes, dots...); files from before that were all vowels.
        question_type = u.get("question_type") or ("capitalize_vowels" if name == "rule" else None)
        if name == "rule" and source:
            computed = apply_rule(question_type, source)
            if computed is not None:
                if expected is not None and expected != computed:
                    problems.append(f"{source_name}: the app's expected answer for '{source}' breaks the {question_type} rule")
                expected = computed  # recompute rather than trust a stored answer
        elif name == "plain" and source:
            expected = source
        units.append({
            "checkin_id": cid, "task_idx": u["task"], "unit": u.get("unit", 0), "task": name,
            "question_type": question_type,
            "practice": "practice" in str(task.get("name", "")),
            "source": source, "expected": expected, "typed": u.get("typed") or "",
            "t_shown": u.get("t_shown"), "t_first_key": u.get("t_first_key"), "t_end": u.get("t_end"),
            "focus_losses": focus_losses(ui, task.get("t_start", -math.inf), task.get("t_end")),
        })

    shown = {(u["task_idx"], u["unit"]): u for u in units}
    keys = []
    fields = list(d.get("key_fields") or KEY_FIELDS)
    events = d.get("keys") or []
    if fields != KEY_FIELDS:  # columns in a different order: put them back in the usual one
        events = [[dict(zip(fields, row)).get(f) for f in KEY_FIELDS] for row in events]
    for p in pair_presses(events):
        u = shown.get((p["task_idx"], p["unit"]))
        if u is None or u["t_shown"] is None:
            continue  # reaction-test presses and anything outside a typing task
        keys.append({"checkin_id": cid, "task_idx": p["task_idx"], "unit": p["unit"],
                     "key": p["key"], "code": p["code"], "caps_lock": p["caps_lock"],
                     "down_ms": p["down"] - u["t_shown"], "up_ms": p["up"] - u["t_shown"]})

    fields = d.get("pvt_fields") or ["phase", "trial", "isi_ms", "onset_t", "resp_t", "rt_ms", "outcome", "code"]
    pvt = [{"checkin_id": cid, **dict(zip(fields, row))} for row in d.get("pvt") or []]
    return {"session": session, "units": units, "keys": keys, "pvt": pvt, "problems": problems}


def load_sessions(folder: str | Path) -> Tables:
    """Read every session file under folder.

    Test runs and the app's in-progress checkpoints are skipped, and so is anyone whose only check-ins
    were aborted (a quick try of the app, not a participant). When the same session appears more than
    once (say, a copy in Downloads too), the most complete copy wins.
    """
    best, problems = {}, []
    skipped_tests = skipped_partial = 0
    for path in sorted(Path(folder).rglob("*.json")):
        if path.name.startswith("._") or "__MACOSX" in path.parts:
            continue  # leftovers from zips made on a Mac, not check-ins
        try:
            s = read_session(path)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as err:
            problems.append(f"{path.name}: skipped ({err})")
            continue
        if s["session"]["test_mode"]:
            skipped_tests += 1
            continue
        if s["session"]["status"] == "in_progress":
            skipped_partial += 1
            continue
        cid = s["session"]["checkin_id"]
        rank = (s["session"]["completed"], len(s["keys"]), len(s["pvt"]))
        if cid not in best or rank > best[cid][0]:
            best[cid] = (rank, s)
    chosen = [s for _, s in best.values()]
    finished = {s["session"]["person"] for s in chosen if s["session"]["completed"]}
    kept = [s for s in chosen if s["session"]["person"] in finished or not s["session"]["aborted"]]
    skipped_tests += len(chosen) - len(kept)
    sessions, units, keys, pvt = [], [], [], []
    for s in kept:
        sessions.append(s["session"])
        units += s["units"]
        keys += s["keys"]
        pvt += s["pvt"]
        problems += s["problems"]
    return Tables(
        sessions=pd.DataFrame(sessions, columns=None if sessions else ["checkin_id", "person"]),
        units=pd.DataFrame(units, columns=None if units else ["checkin_id", "task_idx", "unit", "task"]),
        keys=pd.DataFrame(keys, columns=None if keys else ["checkin_id", "task_idx", "unit", "key"]),
        pvt=pd.DataFrame(pvt, columns=None if pvt else ["checkin_id", "phase", "rt_ms", "outcome"]),
        problems=problems,
        skipped_test_runs=skipped_tests,
        skipped_partial=skipped_partial,
    )


def empty_tables() -> Tables:
    """Tables with no check-ins, for before any data has loaded."""
    return Tables(sessions=pd.DataFrame(columns=["checkin_id", "person", "participant_type"]),
                  units=pd.DataFrame(columns=["checkin_id", "task_idx", "unit", "task"]),
                  keys=pd.DataFrame(columns=["checkin_id", "task_idx", "unit", "key"]),
                  pvt=pd.DataFrame(columns=["checkin_id", "phase", "rt_ms", "outcome"]))


def folder_problem(folder: str | Path) -> str | None:
    """Why a data folder gives no check-ins, when it plainly has none."""
    path = Path(folder).expanduser().resolve()
    if not path.is_dir():
        return f"there's no folder at {path}"
    if not any(path.rglob("*.json")):
        return f"there are no check-in files (*.json) in {path} yet"
    return None
