"""Write fake check-in files in the typing-station format, to test the pipeline before real data.

Two worlds share the same people, schedules and KSS ratings:

  effect/  typing, Shift timing and reaction times get worse with true fatigue
  null/    typing and reaction times ignore fatigue; only practice and noise move them

In both worlds people get faster with practice (at the vowel rule more than at typing in
general). KSS follows true fatigue: hours awake, how fast that person tires, a night-time dip,
caffeine and hour-to-hour noise, plus each person's own way of using the scale. The effect sizes
are in Typist; they are moderate, because with smaller ones 7 people can't show that typing adds
anything to hours awake.

    uv run python simulate.py --out ../data/sim
"""

import argparse
import json
import math
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from tasks import CAPITAL_VOWELS, POOL, capitalise_vowels

START = datetime(2026, 11, 7, 14, 0)
PANEL, ONE_OFFS, CHECKINS = 7, 20, 14
LETTERS = list("abcdefghijklmnopqrstuvwxyz")
FREE_WORDS = ("we lived in a small town near the sea and our house had a green door with a garden "
              "where my brother and i played football after school every day in summer").split()


def epoch_ms(t: datetime) -> float:
    return (t - datetime(1970, 1, 1)).total_seconds() * 1000  # London is on UTC in November


def make_people(rng) -> list[dict]:
    people = []
    for i in range(PANEL + ONE_OFFS):
        panel = i < PANEL
        people.append({
            "id": f"{'P' if panel else 'J'}{i + 1:02d}-" + "".join(rng.choice(list("ABCDEFGHJKLMNPQRSTUVWXYZ23456789"), 4)),
            "cohort": ("teammate" if i < 4 else "overnight") if panel else ("judge" if i < PANEL + 8 else "visitor"),
            "panel": panel,
            "sleeper": panel and i in (1, 4, 6),
            "updates_wake_time": bool(rng.random() < 0.7),
            "wake": START.replace(hour=7) + timedelta(minutes=float(rng.uniform(0, 180))),
            "gap": float(np.exp(rng.normal(np.log(230), 0.3))),  # median ordinary key-to-key gap, ms
            "cv": float(rng.uniform(0.30, 0.45)),
            "rule": float(rng.uniform(1.30, 1.60)),
            "hold": float(rng.uniform(85, 125)),
            "rt": float(rng.uniform(240, 310)),
            "err": float(rng.uniform(0.005, 0.02)),
            "kss_offset": float(rng.normal(0, 0.8)),
            "tire": float(np.exp(rng.normal(0, 0.3))),  # how fast this person gets tired
            "order": rng.permutation(len(POOL)),
        })
    return people


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
    """True fatigue, KSS and the self-report for each check-in."""
    states, drift, prev = [], 0.0, None
    for n, t in enumerate(times, 1):
        woke = sleep[1] if sleep and t > sleep[1] else p["wake"]
        hours = (t - woke).total_seconds() / 3600
        clock = t.hour + t.minute / 60
        night = 0.25 * math.exp(-(((clock - 4.5 + 12) % 24 - 12) ** 2) / 8)  # dip around 04:30
        caffeine = int(rng.random() < 0.25)
        drift = 0.6 * drift + rng.normal(0, 0.1)
        fatigue = float(np.clip(p["tire"] * max(0.0, hours - 2) / 18 + night - 0.15 * caffeine + drift, 0, 1.5))
        kss = int(np.clip(round(2 + 5 * fatigue + p["kss_offset"] + rng.normal(0, 0.7)), 1, 9))
        slept = sleep is not None and prev is not None and prev <= sleep[0] < t
        reported = sleep[1] if sleep and t > sleep[1] and p["updates_wake_time"] else p["wake"]
        states.append({
            "n": n, "t": t, "fatigue": fatigue, "kss": kss, "caffeine": caffeine,
            "slept_min": round((sleep[1] - sleep[0]).total_seconds() / 60) if slept else 0,
            "wake_time": f"{reported:%H:%M}",
            "motivation": int(np.clip(round(8 - 4 * fatigue + rng.normal(0, 1.2)), 0, 10)),
            "mental_fatigue": int(np.clip(round(kss + rng.normal(0, 1)), 1, 9)),
            "sentence": POOL[p["order"][(n - 1) % len(POOL)]],
            "aborted": bool(rng.random() < 0.03),
        })
        prev = t
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
        code = {" ": "Space", "Backspace": "Backspace", "Enter": "Enter"}.get(key, f"Key{key.upper()}")
        hold = max(30.0, self.hold * math.exp(self.rng.normal(0, 0.2)))
        self.events.append([task, 0, "d", round(t, 1), code, key, flags])
        self.events.append([task, 0, "u", round(t + hold, 1), code, key, flags])

    def shift(self, task: int, t: float, down: bool) -> None:
        self.events.append([task, 0, "d" if down else "u", round(t, 1), "ShiftLeft", "Shift", int(down)])

    def copy_task(self, task: int, target: str, shown: float) -> tuple[str, float]:
        """Type target (capitals included for the vowel task). Returns the typed text and end time."""
        rng, typed = self.rng, []
        prev_t = shown + rng.uniform(1500, 3500) * (1 + 0.3 * self.e)  # reading before the first key
        prev_cap, shift_at = False, None
        for ch in target:
            cap = ch in CAPITAL_VOWELS
            gap = self.draw_gap(self.rule if (cap or prev_cap) else 1.0)
            t = prev_t + gap
            if shift_at is not None and not cap:
                self.shift(task, min(t - 20, prev_t + rng.uniform(80, 160)), down=False)
                shift_at = None
            if cap and rng.random() < 2 * self.err:  # forgot the capital
                ch, cap = ch.lower(), False
            if cap and shift_at is None:
                shift_at = t - gap * rng.uniform(0.4, 0.6)
                self.shift(task, shift_at, down=True)
            if not cap and rng.random() < self.err:  # hit a wrong letter
                self.press(task, t, str(rng.choice(LETTERS)))
                if rng.random() < 0.75:  # noticed it: delete and retype
                    t += rng.uniform(250, 700)
                    self.press(task, t, "Backspace")
                    t += rng.uniform(150, 400)
                else:
                    typed.append(self.events[-1][5])
                    prev_t, prev_cap = t, False
                    continue
            self.press(task, t, ch, flags=int(cap))
            typed.append(ch)
            prev_t, prev_cap = t, cap
        if shift_at is not None:
            self.shift(task, prev_t + rng.uniform(80, 160), down=False)
        end = prev_t + rng.uniform(500, 2000)  # a last look before pressing Enter
        self.events.append([task, 0, "d", round(end, 1), "Enter", "Enter", 0])
        return "".join(typed), end

    def free_task(self, task: int, shown: float, seconds: int = 90) -> tuple[str, float]:
        rng, typed, end = self.rng, [], shown + seconds * 1000
        prev_t = shown + rng.uniform(3000, 8000)
        while True:
            for ch in str(rng.choice(FREE_WORDS)) + " ":
                t = prev_t + self.draw_gap(1.4 if ch == " " else 1.0)
                if t > end:
                    return "".join(typed), end
                if ch != " " and rng.random() < self.err:
                    self.press(task, t, str(rng.choice(LETTERS)))
                    t += rng.uniform(250, 700)
                    self.press(task, t, "Backspace")
                    t += rng.uniform(150, 400)
                self.press(task, t, ch)
                typed.append(ch)
                prev_t = t
            if rng.random() < 0.12 + 0.1 * self.e:  # stops to think
                prev_t += rng.uniform(1500, 5000)

    def pvt(self, start: float, seconds: int = 180) -> tuple[list, float]:
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
    t0 = 1000 + rng.uniform(0, 500)  # performance.now() when the session starts
    answered = t0 + rng.uniform(40000, 90000)
    ui = [[round(t0, 1), "session_start", None], [round(answered, 1), "self_report_done", None]]

    start = answered + rng.uniform(5000, 15000)
    expected = capitalise_vowels(st["sentence"])
    typed, end = typist.copy_task(0, expected, start + 5)
    first_key = min(e[3] for e in typist.events if e[0] == 0 and e[2] == "d")
    tasks = [{"idx": 0, "type": "transform", "name": "transform_typing", "t_start": round(start, 1),
              "t_end": round(end, 1), "completed": True, "skipped": False}]
    units = [{"task": 0, "unit": 0, "kind": "transform", "question_id": POOL.index(st["sentence"]),
              "question_type": "capitalize_vowels", "source": st["sentence"], "expected": expected,
              "typed": typed, "t_shown": round(start + 5, 1), "t_first_key": first_key, "t_end": round(end, 1)}]
    if rng.random() < 0.03:
        ui.append([round((start + end) / 2, 1), "window_blur", None])

    start = end + rng.uniform(5000, 12000)
    typed, end = typist.free_task(1, start + 5)
    tasks.append({"idx": 1, "type": "free", "name": "free_text", "t_start": round(start, 1),
                  "t_end": round(end, 1), "completed": True, "skipped": False})
    units.append({"task": 1, "unit": 0, "kind": "free", "prompt_id": "fp5",
                  "prompt": "Describe the place where you grew up.", "expected": None, "typed": typed,
                  "t_shown": round(start + 5, 1), "t_end": round(end, 1), "skipped": False})
    if rng.random() < 0.08:
        ui.append([round((start + end) / 2, 1), "textarea_blur", None])

    start = end + rng.uniform(15000, 25000)
    pvt_rows, end = typist.pvt(start)
    if st["aborted"]:
        pvt_rows, end = pvt_rows[: len(pvt_rows) // 3], start + 60000
    tasks.append({"idx": 2, "type": "pvt", "name": "pvt_main", "t_start": round(start, 1),
                  "t_end": None if st["aborted"] else round(end, 1), "completed": not st["aborted"], "skipped": False})
    for task in tasks:
        ui.append([task["t_start"], "task_start", task["name"]])
        if task["t_end"] is not None:
            ui.append([task["t_end"], "task_end", task["name"]])
    if st["aborted"]:
        ui.append([round(end, 1), "aborted", None])

    origin = epoch_ms(st["t"]) - t0
    return {
        "schema": "typing-station", "schema_version": 1, "app_version": "simulated",
        "session_id": str(uuid.UUID(bytes=rng.bytes(16), version=4)), "status": "final",
        "completed": not st["aborted"], "recovered": False, "aborted": st["aborted"], "test_mode": False,
        "participant_id": p["id"], "cohort": p["cohort"], "collector": "", "station_id": "SIM",
        "local_session_number": st["n"],
        "clock": {"time_origin_epoch_ms": origin, "start_epoch_ms": round(epoch_ms(st["t"])),
                  "start_perf_ms": round(t0, 1), "start_local_iso": f"{st['t']:%Y-%m-%dT%H:%M:%S}+00:00",
                  "tz_offset_min": 0, "tz_name": "Europe/London"},
        "config": {"durations_s": {"pvt_main_s": 180, "pvt_practice_s": 20, "free_s": 90},
                   "pvt": {"isi_min_ms": 1000, "isi_max_ms": 4000, "feedback_ms": 1000, "lapse_ms": 355,
                           "false_start_ms": 100, "timeout_ms": 10000, "lead_s": 3},
                   "practice_pvt": st["n"] == 1},
        "self_report": {"kss": st["kss"], "mental_fatigue": st["mental_fatigue"], "motivation": st["motivation"],
                        "wake_time": st["wake_time"], "sleep_last_night_h": 7, "nap_since_last_min": st["slept_min"],
                        "caffeine_last3h": str(st["caffeine"]), "caffeine_min_ago": None,
                        "keyboard_type": "builtin", "posture": "desk", "answered_t": round(answered, 1)},
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
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.glob("tsession_*.json"):
            old.unlink()
        world_rng = np.random.default_rng(args.seed + 1)
        count = 0
        for p, states in plan:
            for st in states:
                session = make_session(world_rng, p, st, effect)
                (folder / f"tsession_{p['id']}_{st['t']:%Y%m%d-%H%M%S}.json").write_text(json.dumps(session))
                count += 1
        print(f"{world}: wrote {count} check-ins to {folder}")


if __name__ == "__main__":
    main()
