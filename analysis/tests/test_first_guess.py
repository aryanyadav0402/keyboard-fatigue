import json

import numpy as np
import pandas as pd
import pytest

from compare_first_guess import CANDIDATES, compare
from demo import score_session
from features import build_features, typing_features
from first_guess import FIRST_GUESS, LABELS, FatigueIndex, describe
from load_data import load_sessions
from model import no_baseline_model
from simulate import make_people, make_schedule, make_session, make_states


@pytest.fixture(scope="module")
def night(tmp_path_factory):
    """A simulated night in the app's format: 7 teammates and 20 walk-ups."""
    folder = tmp_path_factory.mktemp("night")
    rng, world = np.random.default_rng(5), np.random.default_rng(6)
    for p in make_people(rng):
        times, sleep = make_schedule(rng, p)
        for st in make_states(rng, p, times, sleep):
            session = make_session(world, p, st, effect=True)
            (folder / f"{session['session_id']}.json").write_text(json.dumps(session))
    return folder


def presses(rows):
    return pd.DataFrame(rows, columns=["key", "down_ms", "up_ms"]).assign(caps_lock=False)


def test_pauses_are_measured_against_the_persons_own_pace():
    # a steady typist (100 ms gaps) with one 400 ms gap: that gap is over 3 times their pace
    keys = "abcdefgh"
    downs = [0, 100, 200, 300, 700, 800, 900, 1000]
    steady = typing_features(presses([(k, d, d + 50) for k, d in zip(keys, downs)]), keys, keys)
    assert steady["pause_rel_share"] == pytest.approx(1 / 7)
    assert steady["pause1s_share"] == 0  # nothing reaches a whole second
    # the same rhythm, 3 times slower: the same relative pause share
    slow = typing_features(presses([(k, 3 * d, 3 * d + 50) for k, d in zip(keys, downs)]), keys, keys)
    assert slow["pause_rel_share"] == pytest.approx(steady["pause_rel_share"])


def test_fatigue_index_follows_the_fixed_directions():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, len(FIRST_GUESS)))
    signs = np.array([1, 1, 1, 1, -1])  # accuracy (last) counts the other way
    y = 5 + (X * signs).mean(axis=1) * 2
    model = FatigueIndex(FIRST_GUESS).fit(X, y)
    assert model.slope_ > 0
    tired, alert = np.array([signs * 1.0]), np.array([signs * -1.0])
    assert model.predict(tired)[0] > model.predict(alert)[0]
    # a row with nothing measured is treated as a typical attempt
    assert model.predict(np.full((1, len(FIRST_GUESS)), np.nan))[0] == pytest.approx(model.intercept_)


def test_fatigue_index_falls_back_to_the_average_when_the_data_disagree():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, len(FIRST_GUESS)))
    y = 5 - (X * np.array([1, 1, 1, 1, -1])).mean(axis=1) * 2  # the opposite of the fixed directions
    model = FatigueIndex(FIRST_GUESS).fit(X, y)
    assert model.slope_ == 0 and model.predict(X) == pytest.approx(np.full(60, y.mean()))


def test_no_baseline_model_is_the_fatigue_index(night):
    model, cols = no_baseline_model(build_features(load_sessions(night)))
    assert isinstance(model, FatigueIndex) and cols == FIRST_GUESS


def test_comparison_predicts_each_panel_person_as_a_stranger(night):
    table = compare(build_features(load_sessions(night)))
    assert list(table.index) == list(CANDIDATES)
    assert table.loc["Fatigue index, 5 features", "MAE"] <= table.loc["Always guess the panel average", "MAE"] + 0.05
    assert table["Walk-up MAE"].notna().all()


def test_describe_ranks_an_attempt_against_the_panel_and_explains_it():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(100, len(FIRST_GUESS)))
    signs = np.array([1, 1, 1, 1, -1])
    model = FatigueIndex(FIRST_GUESS).fit(X, 5 + (X * signs).mean(axis=1) * 2)
    typical = describe(model, X, model.center_)
    tired = describe(model, X, model.center_ + 1.5 * model.scale_ * signs)
    assert 40 <= typical["percentile"] <= 60 and tired["percentile"] >= 90
    assert "more tired than" in tired["text"] and tired["panel_checkins"] == 100
    assert [d["label"] for d in tired["drivers"]] and all(d["label"] in LABELS.values() for d in tired["drivers"])
    assert all(d["pushes"] == "more tired" for d in tired["drivers"])
    sizes = [abs(d["z"]) for d in tired["drivers"]]
    assert sizes == sorted(sizes, reverse=True)
    # an attempt where nothing could be measured has no index
    assert describe(model, X, np.full(len(FIRST_GUESS), np.nan)) is None


def demo_body(folder, username=None):
    """The short form the demo page sends, from the first simulated session in folder."""
    session = json.loads(sorted(folder.glob("*.json"))[0].read_text())
    unit = session["units"][0]
    keys = [{"type": "down" if k[2] == "d" else "up", "key": k[5], "code": k[4], "t": k[3]}
            for k in session["keys"] if k[0] == 0]
    return {"username": username, "sentence": unit["source"], "rule": unit["question_type"],
            "typed": unit["typed"], "shown_at": unit["t_shown"], "keys": keys}


def test_demo_reply_includes_the_fatigue_index(night, tmp_path):
    train, db = load_sessions(night), tmp_path / "demo.sqlite"
    first = score_session(demo_body(night, "a judge"), train, db)
    assert first["model"] == "no-baseline model"
    info = first["fatigue_index"]
    assert 0 <= info["percentile"] <= 100 and info["text"].startswith("Your typing looks more tired than")
    assert {d["feature"] for d in info["drivers"]} <= set(FIRST_GUESS) and info["drivers"]
    json.dumps(first, allow_nan=False)
    again = score_session(demo_body(night, "a judge"), train, db)  # compared with their best now
    assert again["model"] == "best-attempt model" and again["fatigue_index"] is None  # only themselves
