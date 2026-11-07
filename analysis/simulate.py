"""Write fake check-in files in the check-in app's version 1.2 format, to test the pipeline before
real data arrives.

Two worlds share the same people, schedules and KSS ratings:

  effect/  typing and reaction times get worse with true fatigue
  null/    typing and reaction times ignore fatigue; only practice and noise move them

Like the real app, Task 1's rule rotates through five types and some rules are harder than others,
the reaction test lasts 20 seconds, and nobody reports a wake-up time. In both worlds people get
faster with practice (at the rule more than at typing in general). KSS follows true fatigue: hours
awake, how fast that person tires, a night-time dip, caffeine and hour-to-hour noise, plus each
person's own way of using the scale. The effect sizes are in Typist; they are moderate, because with
smaller ones 7 people can't show that typing adds anything to the clock.

Some sessions also get an in-progress checkpoint copy under partial/, which the loader should skip.

    uv run python simulate.py --out ../data/sim
"""

import argparse
import json
import math
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from tasks import APP_QUESTIONS, RULES, apply_rule

START = datetime(2026, 11, 7, 14, 0)
PANEL, ONE_OFFS, CHECKINS = 7, 20, 14
LETTERS = list("abcdefghijklmnopqrstuvwxyz")
RULE_TYPES = list(RULES)
ITEMS_BY_TYPE = {t: [q for q in APP_QUESTIONS if q[1] == t] for t in RULE_TYPES}
# How much harder each rule is than capitalising vowels. Made up for the simulation, not measured.
DIFFICULTY = {"dashes_for_spaces": 0.5, "capitalize_vowels": 1.0, "capitalize_first_and_last": 1.1,
              "dashes_and_vowels": 1.2, "dots_and_vowels": 1.3}
INSTRUCTIONS = {
    "capitalize_vowels": "Type this sentence, but CAPITALIZE all vowels (a, e, i, o, u):",
    "dashes_for_spaces": "Replace every space with a dash (-):",
    "dots_and_vowels": "Replace every space with a dot (.) and CAPITALIZE all vowels:",
    "dashes_and_vowels": "Replace spaces with dashes (-) and CAPITALIZE all vowels:",
    "capitalize_first_and_last": "Capitalize the FIRST and LAST letter of each word:",
}
KEY_CODES = {" ": "Space", "-": "Minus", ".": "Period", "Backspace": "Backspace", "Enter": "Enter"}


def epoch_ms(t: datetime) -> float:
    return (t - datetime(1970, 1, 1)).total_seconds() * 1000  # London is on UTC in November


def make_people(rng) -> list[dict]:
    people = []
    for i in range(PANEL + ONE_OFFS):
        panel = i < PANEL
        people.append({
            "id": f"PANEL-{i + 1:02d}" if panel else f"WALKUP-{i - PANEL + 1:02d}",
            "cohort": "teammate" if panel else "walkup",
            "panel": panel,
            "sleeper": panel and i in (1, 4, 6),
            "wake": START.replace(hour=7) + timedelta(minutes=float(rng.uniform(0, 180))),
            "gap": float(np.exp(rng.normal(np.log(230), 0.3))),  # median ordinary key-to-key gap, ms
            "cv": float(rng.uniform(0.30, 0.45)),
            "rule": float(rng.uniform(1.30, 1.60)),
            "hold": float(rng.uniform(85, 125)),
            "rt": float(rng.uniform(240, 310)),
            "err": float(rng.uniform(0.005, 0.02)),
            "kss_offset": float(rng.normal(0, 0.8)),
            "tire": float(np.exp(rng.normal(0, 0.3))),  # how fast this person gets tired
            "first_type": int(rng.integers(len(RULE_TYPES))),
            "item_order": {t: rng.permutation(len(ITEMS_BY_TYPE[t])) for t in RULE_TYPES},
        })
    return people


def question_for(p: dict, n: int) -> dict:
    """Like the app: the rule type cycles through all five, and items within a type don't repeat."""
    rule_type = RULE_TYPES[(p["first_type"] + n - 1) % len(RULE_TYPES)]
    items = ITEMS_BY_TYPE[rule_type]
    qid, _, source = items[p["item_order"][rule_type][((n - 1) // len(RULE_TYPES)) % len(items)]]
    instruction = INSTRUCTIONS[rule_type]
    return {"id": qid, "type": rule_type, "instruction": instruction, "source": source,
            "prompt": f"{instruction} '{source}'", "expected": apply_rule(rule_type, source)}


def make_schedule(rng, p: dict) -> tuple[list[datetime], tuple[datetime, datetime] | None]:
    """Check-in times, and the night's sleep for people who sleep."""
    if not p["panel"]:
        first = START + timedelta(hours=float(rng.uniform(0.5, 19)))
        second = [first + timedelta(minutes=float(rng.uniform(40, 120)))] if rng.random() < 0.5 else []
        return [first] + second, None
    times = [START + timedelta(minutes=80 * k + float(rng.uniform(-10, 10))) for k in range(CHECKINS)]
    if not p["sleeper"]:
        return times, None
    sleep_start = datetime(2026, 11, 8, 2, 30) + timedelta(minutes=float(rng.uniform(0, 60)))
    sleep_end = sleep_start + timedelta(minutes=float(rng.uniform(90, 180)))
    times = [t for t in times if not sleep_start - timedelta(minutes=30) < t < sleep_end + timedelta(minutes=40)]
    times += [sleep_start - timedelta(minutes=10), sleep_end + timedelta(minutes=30)]
    return sorted(times), (sleep_start, sleep_end)


def make_states(rng, p: dict, times: list[datetime], sleep) -> list[dict]:
    """True fatigue, KSS, the quick questions and the Task 1 question for each check-in."""
    states, drift = [], 0.0
    for n, t in enumerate(times, 1):
        woke = sleep[1] if sleep and t > sleep[1] else p["wake"]
        hours = (t - woke).total_seconds() / 3600
        clock = t.hour + t.minute / 60
        night = 0.25 * math.exp(-(((clock - 4.5 + 12) % 24 - 12) ** 2) / 8)  # dip around 04:30
        caffeine = int(rng.random() < 0.25)
        drift = 0.6 * drift + rng.normal(0, 0.1)
        fatigue = float(np.clip(p["tire"] * max(0.0, hours - 2) / 18 + night - 0.15 * caffeine + drift, 0, 1.5))
        kss = int(np.clip(round(2 + 5 * fatigue + p["kss_offset"] + rng.normal(0, 0.7)), 1, 9))
        states.append({
            "n": n, "t": t, "fatigue": fatigue, "kss": kss, "caffeine": caffeine,
            "motivation": int(np.clip(round(4 - 2 * fatigue + rng.normal(0, 0.8)), 1, 5)),
            "mental_fatigue": int(np.clip(round(1 + 3 * fatigue + rng.normal(0, 0.7)), 1, 5)),
            "question": question_for(p, n),
            "aborted": bool(rng.random() < 0.03),
        })
    return states


class Typist:
    """Key events and reaction times for one check-in, given the person, practice and fatigue."""

    def __init__(self, rng, p: dict, n: int, fatigue: float, effect: bool):
        e = fatigue if effect else 0.0
        self.rng, self.e, self.events = rng, e, []
        # Practice: keys get 15% faster by the 14th check-in, and the rule's extra cost halves.
        # Fatigue effects at e = 1 are moderate: 10% slower keys, 40% more irregular, an 80%
        # bigger extra cost for the rule, more pauses and errors, and 20% slower reactions.
        self.gap = p["gap"] * n ** -0.06 * (1 + 0.10 * e)
        self.sigma = math.asinh(p["cv"] * (1 + 0.4 * e) / 2) / 0.6745  # lognormal with IQR/median = cv
        self.rule = 1 + (p["rule"] - 1) * n ** -0.25 * (1 + 0.8 * e)
        self.hold = p["hold"] * (1 + 0.08 * e)
        self.pause_p = 0.02 + 0.06 * e
        self.err = p["err"] * (1 + 1.5 * e)
        self.rt = p["rt"] * (1 + 0.2 * e)
        self.lapse_p = 0.02 + 0.15 * e ** 1.5
        self.false_start_p = 0.02 + 0.03 * e

    def draw_gap(self, scale: float = 1.0) -> float:
        gap = self.gap * scale * math.exp(self.rng.normal(0, self.sigma))
        if self.rng.random() < self.pause_p:
            gap += self.rng.uniform(600, 2000)
        return gap

    def press(self, task: int, t: float, key: str, flags: int = 0) -> None:
        code = KEY_CODES.get(key, f"Key{key.upper()}")
        hold = max(30.0, self.hold * math.exp(self.rng.normal(0, 0.2)))
        self.events.append([task, 0, "d", round(t, 1), code, key, flags])
        self.events.append([task, 0, "u", round(t + hold, 1), code, key, flags])

    def shift(self, task: int, t: float, down: bool) -> None:
        self.events.append([task, 0, "d" if down else "u", round(t, 1), "ShiftLeft", "Shift", int(down)])

    def copy_task(self, task: int, source: str, expected: str, shown: float, difficulty: float) -> tuple[str, float]:
        """Type the expected answer; gaps next to characters the rule changes are slower."""
        rng, typed = self.rng, []
        rule = 1 + (self.rule - 1) * difficulty
        prev_t = shown + rng.uniform(1500, 3500) * (1 + 0.3 * difficulty) * (1 + 0.3 * self.e)  # reading time
        prev_changed, shift_at = False, None
        for want, plain in zip(expected, source):
            changed = want != plain
            gap = self.draw_gap(rule if (changed or prev_changed) else 1.0)
            t = prev_t + gap
            ch = plain if changed and rng.random() < 2 * self.err * difficulty else want  # forgot the rule
            capital = ch.isupper()
            if shift_at is not None and not capital:
                self.shift(task, min(t - 20, prev_t + rng.uniform(80, 160)), down=False)
                shift_at = None
            if capital and shift_at is None:
                shift_at = t - gap * rng.uniform(0.4, 0.6)
                self.shift(task, shift_at, down=True)
            if not changed and ch.isalpha() and ch.islower() and rng.random() < self.err * difficulty:
                self.press(task, t, str(rng.choice(LETTERS)))  # hit a wrong letter
                if rng.random() < 0.75:  # noticed it: delete and retype
                    t += rng.uniform(250, 700)
                    self.press(task, t, "Backspace")
                    t += rng.uniform(150, 400)
                else:
                    typed.append(self.events[-1][5])
                    prev_t, prev_changed = t, False
                    continue
            self.press(task, t, ch, flags=int(capital))
            typed.append(ch)
            prev_t, prev_changed = t, changed
        if shift_at is not None:
            self.shift(task, prev_t + rng.uniform(80, 160), down=False)
        end = prev_t + rng.uniform(500, 2000)  # a last look before pressing Enter
        self.events.append([task, 0, "d", round(end, 1), "Enter", "Enter", 0])
        return "".join(typed), end

    def pvt(self, start: float, seconds: int = 20) -> tuple[list, float]:
        rng, rows, trial = self.rng, [], 0
        t = start + 3000
        end = t + seconds * 1000
        while True:
            trial += 1
            isi = float(rng.integers(1000, 4001))
            if rng.random() < self.false_start_p:
                resp = t + rng.uniform(200, isi - 50)
                rows.append(["main", trial, isi, None, round(resp, 1), None, "false_start", "Space"])
                t = resp + 1000
            onset = t + isi
            if onset > end:
                return rows, end
            rt = self.rt * math.exp(rng.normal(0, 0.12))
            if rng.random() < self.lapse_p:
                rt += rng.uniform(200, 1500)
            rows.append(["main", trial, isi, round(onset, 1), round(onset + rt, 1), round(rt, 1),
                         "lapse" if rt >= 355 else "valid", "Space"])
            t = onset + rt + 1000


def make_session(rng, p: dict, st: dict, effect: bool) -> dict:
    typist = Typist(rng, p, st["n"], st["fatigue"], effect)
    q = st["question"]
    t0 = 1000 + rng.uniform(0, 500)  # performance.now() when the session starts
    answered = t0 + rng.uniform(15000, 40000)  # name and the five quick questions
    ui = [[round(t0, 1), "session_start", None], [round(answered, 1), "self_report_done", None]]

    start = answered + rng.uniform(3000, 8000)
    typed, end = typist.copy_task(0, q["source"], q["expected"], start + 5, DIFFICULTY[q["type"]])
    first_key = min(e[3] for e in typist.events if e[0] == 0 and e[2] == "d" and e[5] != "Shift")
    tasks = [{"idx": 0, "type": "transform", "name": "transform_typing", "t_start": round(start, 1),
              "t_end": round(end, 1), "completed": True, "skipped": False}]
    units = [{"task": 0, "unit": 0, "kind": "transform", "question_id": q["id"], "question_type": q["type"],
              "instruction": q["instruction"], "source": q["source"], "prompt": q["prompt"],
              "expected": q["expected"], "typed": typed, "t_shown": round(start + 5, 1),
              "t_first_key": first_key, "t_end": round(end, 1)}]
    if rng.random() < 0.03:
        ui.append([round((start + end) / 2, 1), "window_blur", None])

    start = end + rng.uniform(5000, 12000)
    pvt_rows, end = typist.pvt(start)
    if st["aborted"]:
        pvt_rows, end = pvt_rows[: len(pvt_rows) // 2], start + 8000
    tasks.append({"idx": 1, "type": "pvt", "name": "pvt_main", "t_start": round(start, 1),
                  "t_end": None if st["aborted"] else round(end, 1), "completed": not st["aborted"], "skipped": False})
    for task in tasks:
        ui.append([task["t_start"], "task_start", task["name"]])
        if task["t_end"] is not None:
            ui.append([task["t_end"], "task_end", task["name"]])
    if st["aborted"]:
        ui.append([round(end, 1), "aborted", None])

    origin = epoch_ms(st["t"]) - t0
    return {
        "schema": "typing-station", "schema_version": 1, "app_version": "1.2.0-simulated",
        "session_id": str(uuid.UUID(bytes=rng.bytes(16), version=4)), "status": "final",
        "completed": not st["aborted"], "recovered": False, "aborted": st["aborted"], "test_mode": False,
        "participant_id": p["id"], "participant_name": p["id"], "cohort": p["cohort"], "collector": "",
        "station_id": "SIM", "local_session_number": st["n"],
        "clock": {"time_origin_epoch_ms": origin, "start_epoch_ms": round(epoch_ms(st["t"])),
                  "start_perf_ms": round(t0, 1), "start_local_iso": f"{st['t']:%Y-%m-%dT%H:%M:%S}+00:00",
                  "tz_offset_min": 0, "tz_name": "Europe/London"},
        "config": {"durations_s": {"pvt_main_s": 20},
                   "pvt": {"isi_min_ms": 1000, "isi_max_ms": 4000, "feedback_ms": 1000, "lapse_ms": 355,
                           "false_start_ms": 100, "timeout_ms": 10000, "lead_s": 3},
                   "question_pool_id": "simulated", "questions_per_session": 1, "questions": [q],
                   "practice_pvt": False, "free_task": False},
        "consent": {"waived": True}, "profile": None,
        "self_report": {"kss": st["kss"], "mental_fatigue": st["mental_fatigue"], "motivation": st["motivation"],
                        "sleep_last_night_h": 7, "caffeine_last3h": str(st["caffeine"]), "answered_t": round(answered, 1)},
        "post": {"interrupted": "none"},
        "tasks": tasks,
        "key_fields": ["task", "unit", "type", "t", "code", "key", "flags"],
        "keys": sorted(typist.events, key=lambda e: e[3]),
        "units": units,
        "pvt_fields": ["phase", "trial", "isi_ms", "onset_t", "resp_t", "rt_ms", "outcome", "code"],
        "pvt": pvt_rows,
        "ui_fields": ["t", "event", "detail"],
        "ui": sorted(ui, key=lambda e: e[0]),
        "error": None,
        "end": {"end_epoch_ms": round(origin + end), "end_perf_ms": round(end, 1)},
    }


def checkpoint_copy(session: dict) -> dict:
    """What the app's server keeps under data/partial while a session is still running."""
    partial = dict(session, status="in_progress", completed=False, pvt=[], end=None)
    partial["keys"] = session["keys"][: len(session["keys"]) // 2]
    return partial


def main() -> None:
    parser = argparse.ArgumentParser(description="Write simulated check-in files for both worlds.")
    parser.add_argument("--out", default="../data/sim", help="folder for the effect/ and null/ worlds")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    plan = []
    for p in make_people(rng):
        times, sleep = make_schedule(rng, p)
        plan.append((p, make_states(rng, p, times, sleep)))

    for world, effect in (("effect", True), ("null", False)):
        folder = Path(args.out) / world
        for old in folder.rglob("*.json"):
            old.unlink()
        world_rng = np.random.default_rng(args.seed + 1)
        count = partial = 0
        for p, states in plan:
            for st in states:
                session = make_session(world_rng, p, st, effect)
                for sub, data in [("sessions", session)] + [("partial", checkpoint_copy(session))] * (st["n"] % 9 == 5):
                    path = folder / sub / p["id"] / f"{session['session_id']}.json"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(data))
                    partial += sub == "partial"
                count += 1
        print(f"{world}: wrote {count} check-ins and {partial} in-progress checkpoints to {folder}")


if __name__ == "__main__":
    main()
