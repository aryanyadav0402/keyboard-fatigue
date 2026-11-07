import json
import math

import numpy as np
import pandas as pd
import pytest

from features import (add_clock, adjust_for_rule_type, add_hours_awake, build_features, hours_awake, processed_table, pvt_features,
                      raw_table, typing_features)
from load_data import load_sessions, pair_presses
from model import run_models
from report import write_outputs
from simulate import make_people, make_schedule, make_session, make_states
from tasks import POOL, apply_rule, capitalise_vowels, score, sentence_ok


def presses(rows):
    """A press table from (key, down_ms, up_ms) tuples."""
    return pd.DataFrame(rows, columns=["key", "down_ms", "up_ms"]).assign(caps_lock=False)


def test_capitalise_vowels():
    assert capitalise_vowels("a curious raccoon found a shiny golden key") == "A cUrIOUs rAccOOn fOUnd A shIny gOldEn kEy"
    assert capitalise_vowels("why try") == "why try"  # y is not a vowel here
    assert capitalise_vowels("") == ""


def test_pool_meets_the_rules():
    assert len(POOL) == 40
    assert all(sentence_ok(s) for s in POOL)
    assert len(set(POOL)) == len(POOL)


@pytest.mark.parametrize("typed, expected, counts", [
    ("hEllO", "hEllO", (0, 0, 0, 0)),          # (distance, missed, wrong, other)
    ("hEllo", "hEllO", (1, 1, 0, 0)),          # forgot a capital
    ("HEllO", "hEllO", (1, 0, 1, 0)),          # capitalised a consonant
    ("hEllO wOrld", "hEllO wOrd", (1, 0, 0, 1)),  # an extra letter
    ("hEO", "hEllO", (2, 0, 0, 2)),            # two missing letters
    ("hellO", "hEllO", (1, 1, 0, 0)),
])
def test_score_counts_errors(typed, expected, counts):
    s = score(typed, expected)
    assert (s["distance"], s["missed_capitals"], s["wrong_capitals"], s["other_errors"]) == counts
    assert s["accuracy"] == pytest.approx(1 - counts[0] / len(expected))


def test_gap_and_hold_times_from_a_hand_made_table():
    # gaps 200, 250, 150, 300, 100; holds 80-130; "e" is still down when "f" goes down
    rows = [("a", 0, 80), ("b", 200, 290), ("c", 450, 550), ("d", 600, 710), ("e", 900, 1020), ("f", 1000, 1130)]
    f = typing_features(presses(rows), "abcdef", "abcdef", rule=False)
    assert f["gap_med"] == 200
    assert f["gap_iqr"] == 100 and f["gap_cv"] == pytest.approx(0.5)
    assert f["hold_med"] == 105
    assert f["rollover_share"] == pytest.approx(0.2)
    assert f["pause500_share"] == 0
    assert f["wpm"] == pytest.approx(6 / 5 / (1130 / 60000))
    assert f["accuracy"] == 1


def test_rule_cost_and_shift_lead_leave_shift_out_of_the_gaps():
    rows = [("a", 0, 50), ("b", 100, 150), ("c", 200, 250), ("Shift", 300, 450), ("E", 400, 450),
            ("d", 600, 650), ("f", 700, 750), ("Shift", 750, 950), ("I", 900, 950), ("g", 1100, 1150),
            ("h", 1200, 1250)]
    f = typing_features(presses(rows), "abcEdfIgh", "abcEdfIgh", rule=True)
    assert f["n_keys"] == 9  # Shift isn't counted
    assert f["ord_gap_med"] == 100 and f["touch_gap_med"] == 200
    assert f["rule_cost"] == pytest.approx(2.0)
    assert f["shift_lead_med"] == pytest.approx(125)


def test_corrections_are_left_out_of_the_gaps():
    rows = [("a", 0, 50), ("b", 100, 150), ("x", 200, 250), ("Backspace", 900, 950), ("c", 1500, 1550),
            ("d", 1600, 1650), ("e", 1700, 1750), ("f", 1800, 1850), ("g", 1900, 1950)]
    f = typing_features(presses(rows), "abcdefg", "abcdefg", rule=False)
    assert f["longest_gap"] == 100  # the gaps around x and Backspace are skipped
    assert f["backspace_per100"] == pytest.approx(100 / 9)


def test_hours_awake_across_midnight():
    assert hours_awake(pd.Timestamp("2026-11-08 02:30"), "08:00") == pytest.approx(18.5)
    assert hours_awake(pd.Timestamp("2026-11-07 23:00"), "07:30") == pytest.approx(15.5)
    assert hours_awake(pd.Timestamp("2026-11-08 00:10"), "23:50") == pytest.approx(1 / 3)
    assert math.isnan(hours_awake(pd.Timestamp("2026-11-08 00:10"), "soon"))


def test_sleep_resets_hours_awake_even_with_a_stale_wake_time():
    f = pd.DataFrame({
        "person": ["P"] * 3,
        "timestamp": pd.to_datetime(["2026-11-08 02:00", "2026-11-08 05:30", "2026-11-08 07:00"]),
        "wake_time": ["08:00", "08:00", "08:00"],  # never updated after sleeping
        "nap_since_last_min": [0, 180, 0],
    })
    assert add_hours_awake(f).tolist() == pytest.approx([18.0, 0.5, 2.0])


def test_key_events_pair_up_and_skip_auto_repeats():
    events = [[0, 0, "d", 10, "KeyA", "a", 0], [0, 0, "d", 60, "KeyA", "a", 32], [0, 0, "u", 90, "KeyA", "a", 0],
              [0, 0, "d", 100, "KeyB", "b", 0]]
    p = pair_presses(events)
    assert [(x["key"], x["down"], x["up"]) for x in p[:1]] == [("a", 10, 90)]
    assert len(p) == 2 and math.isnan(p[1]["up"])


@pytest.fixture
def few_sessions(tmp_path):
    """Three simulated check-ins from one panel person, in the app's file format."""
    rng = np.random.default_rng(1)
    person = make_people(rng)[0]
    times, sleep = make_schedule(rng, person)
    for st in make_states(rng, person, times[:3], sleep):
        session = make_session(rng, person, st, effect=True)
        (tmp_path / f"{session['session_id']}.json").write_text(json.dumps(session))
    return tmp_path


def test_partial_data_skips_models_without_crashing(few_sessions, tmp_path):
    tables = load_sessions(few_sessions)
    f = build_features(tables)
    assert len(f) == 3 and f.vt_rule_cost.notna().all()
    r = run_models(f, n_perm=10)
    assert r["missing"]  # one person isn't enough
    report = write_outputs(tables, f, r, tmp_path / "out", str(few_sessions))
    assert "Not enough data" in report.read_text()


def test_processed_file_measures_change_from_first_two_checkins(few_sessions):
    f = build_features(load_sessions(few_sessions))
    processed = processed_table(f)
    assert processed.is_baseline.tolist() == [True, True, False]
    assert processed.vt_wpm_dev.iloc[:2].mean() == pytest.approx(0)
    # the change is measured on the rule-adjusted version, because each visit shows a different rule
    assert processed.vt_wpm_dev.iloc[2] == pytest.approx(f.vt_wpm_adj.iloc[2] - f.vt_wpm_adj.iloc[:2].mean())
    assert processed.kss_delta.iloc[2] == pytest.approx(f.kss.iloc[2] - f.kss.iloc[:2].mean())
    assert "backspace_per100_adj" in processed and "backspace_per100_dev" not in processed
    assert not any(c.endswith(("_dev", "_adj")) for c in raw_table(f))


@pytest.mark.parametrize("rule_type, source, expected", [
    ("dashes_for_spaces", "a sleepy panda shared a sweet red apple with a tiny duck",
     "a-sleepy-panda-shared-a-sweet-red-apple-with-a-tiny-duck"),
    ("dots_and_vowels", "the baby otter floated down the calm blue river", "thE.bAby.OttEr.flOAtEd.dOwn.thE.cAlm.blUE.rIvEr"),
    ("dashes_and_vowels", "the polite pig invited the hedgehog to tea", "thE-pOlItE-pIg-InvItEd-thE-hEdgEhOg-tO-tEA"),
    # the team's hand-written answer had "O F" here; the rule gives "OF"
    ("capitalize_first_and_last", "the clumsy bear dropped his favorite jar of honey",
     "ThE ClumsY BeaR DroppeD HiS FavoritE JaR OF HoneY"),
])
def test_rules_match_the_apps_examples(rule_type, source, expected):
    assert apply_rule(rule_type, source) == expected


def test_rule_cost_works_for_dashes():
    rows = [("a", 0, 50), ("b", 100, 150), ("-", 300, 350), ("c", 500, 550), ("d", 600, 650), ("-", 800, 850),
            ("e", 1000, 1050), ("f", 1100, 1150)]
    f = typing_features(presses(rows), "ab-cd-ef", "ab-cd-ef", source="ab cd ef", rule=True)
    assert f["ord_gap_med"] == 100 and f["touch_gap_med"] == 200
    assert f["rule_cost"] == pytest.approx(2.0)


def test_a_20_second_reaction_test_still_counts():
    trials = pd.DataFrame({"phase": "main", "rt_ms": [250, 270, 300, 380, 260, 290, 310],
                           "outcome": ["valid", "valid", "valid", "lapse", "valid", "valid", "valid"]})
    out = pvt_features(trials, lapse_ms=355, false_start_ms=100, timeout_ms=10000)
    assert out["rt_trials"] == 7 and out["rt_median"] == 290
    assert out["rt_lapse_rate"] == pytest.approx(1 / 7)


def test_in_progress_checkpoints_and_duplicate_copies_are_skipped(few_sessions):
    final = json.loads(next(few_sessions.glob("*.json")).read_text())
    (few_sessions / "partial").mkdir()
    (few_sessions / "partial" / "copy.json").write_text(json.dumps(dict(final, status="in_progress", keys=[])))
    (few_sessions / "downloaded_copy.json").write_text(json.dumps(final))
    tables = load_sessions(few_sessions)
    assert len(tables.sessions) == 3 and tables.skipped_partial == 1
    assert (tables.sessions.status == "final").all()


def test_clock_falls_back_to_hours_since_first_checkin():
    f = pd.DataFrame({"person": ["P"] * 3,
                      "timestamp": pd.to_datetime(["2026-11-07 14:00", "2026-11-07 15:30", "2026-11-07 18:00"]),
                      "hours_awake": [np.nan] * 3})
    out = add_clock(f)
    assert out.hours_clock.tolist() == pytest.approx([0, 1.5, 4.0])
    assert (out.clock_basis == "hours since first check-in").all()


def test_rule_features_are_measured_against_the_panels_typical_value_for_that_rule():
    f = pd.DataFrame({
        "participant_type": ["panel"] * 6 + ["oneoff"],
        "question_type": ["capitalize_vowels"] * 3 + ["dashes_for_spaces"] * 4,
        "vt_rule_cost": [1.4, 1.5, 1.6, 1.1, 1.2, 1.3, 2.4],
        "vt_wpm": [30, 32, 34, 40, 42, 44, 10],
    })
    out = adjust_for_rule_type(f)
    # rule cost is divided by the panel's median for that rule, so 1.0 = normal for that rule
    assert out.vt_rule_cost_adj.iloc[:6].tolist() == pytest.approx([1.4 / 1.5, 1, 1.6 / 1.5, 1.1 / 1.2, 1, 1.3 / 1.2])
    assert out.vt_rule_cost_adj.iloc[6] == pytest.approx(2.0)  # the walk-up doesn't move the reference
    # other features: minus the rule's panel median, plus the panel's overall median (37)
    assert out.vt_wpm_adj.iloc[:3].tolist() == pytest.approx([35, 37, 39])
