import json
from datetime import datetime, timezone

import numpy as np
import pytest

from ananya_model import export_ananya, local_hour, phase_of, score_ananya, windows
from attempt import BadSession
from attempt_store import MemoryStore
from frozen_model import teammate_key
from simulate import make_people, make_schedule, make_session, make_states

COLS = ["log_iki_med", "log_iki_sd", "iki_cv", "cps", "pauses_pm", "pause_frac", "log_dwell", "bs_rate"]
COEF = [0.1, -0.1, 0.2, 0.05, 0.3, -0.2, 0.5, -0.3]


@pytest.fixture(scope="module")
def sessions():
    """Three simulated check-ins from one teammate, in the app's format."""
    rng, world = np.random.default_rng(21), np.random.default_rng(22)
    p = make_people(rng)[0]
    times, sleep = make_schedule(rng, p)
    return [make_session(world, p, st, effect=True) for st in make_states(rng, p, times, sleep)[:3]]


@pytest.fixture(scope="module")
def model(sessions, tmp_path_factory):
    """Her package for one made-up teammate, Zoe, whose usual typing is her first check-in."""
    folder = tmp_path_factory.mktemp("final_pkg")
    wf = windows(sessions[0]["keys"])
    usual = {c: float(np.mean([f[c] for f in wf])) for c in COLS}
    (folder / "model_weights.json").write_text(json.dumps({
        "feature_order": COLS, "alpha": 10.0, "windows": 2, "scaler_mean": [0.0] * 8, "scaler_scale": [1.0] * 8,
        "coef": COEF, "intercept": 0.25}))
    (folder / "person_baselines.json").write_text(json.dumps({"ZOE": {
        "n_clean_sessions": 3, "usual_features": usual, "usual_kss_by_phase": {"pre": 6.0, "post": 3.0}}}))
    return json.loads(json.dumps(export_ananya(folder, "ananya-test")))


def as_page_sends(session, username, hour):
    """The whole session Christy's page posts, started at hour:30 UTC on 8 Nov."""
    start = int(datetime(2026, 11, 8, hour, 30, tzinfo=timezone.utc).timestamp() * 1000)
    clock = dict(session["clock"], start_epoch_ms=start, tz_offset_min=0, start_local_iso=f"2026-11-08T{hour:02d}:30:00.000Z")
    return dict(session, participant_id=username, metadata={"username": username}, clock=clock)


def passes_page_checks(r):
    """What ui/src/lib/predictionApi.ts (isPredictionResponse) accepts."""
    if r["model"] == "not enough data yet":
        return r["score"] is None and r["label"] is None and r["level"] is None and isinstance(r["note"], str)
    return (r["model"] in ("best-attempt model", "no-baseline model") and isinstance(r["score"], float)
            and 1 <= r["score"] <= 9 and isinstance(r["label"], str) and bool(r["label"].strip())
            and isinstance(r["level"], str) and isinstance(r["scale"], str) and isinstance(r["note"], str))


def test_typing_like_usual_gives_the_usual_rating_plus_the_intercept(sessions, model):
    day = score_ananya(as_page_sends(sessions[0], "Zoe", 12), model)
    assert passes_page_checks(day) and day["model"] == "best-attempt model" and "Ananya" in day["note"]
    assert day["phase"] == "post" and day["usual_kss"] == 3.0 and day["shift"] == pytest.approx(0.25, abs=1e-3)
    night = score_ananya(as_page_sends(sessions[0], "zoe", 22), model)  # names match however they're typed
    assert night["phase"] == "pre" and night["usual_kss"] == 6.0 and night["score"] == pytest.approx(6.25, abs=0.051)


def test_other_typing_moves_the_score_by_the_model(sessions, model):
    usual = model["people"][teammate_key("Zoe")]["usual_features"]
    X = np.array([[f[c] - usual[c] for c in COLS] for f in windows(sessions[2]["keys"])])
    r = score_ananya(as_page_sends(sessions[2], "Zoe", 12), model)
    assert r["shift"] == pytest.approx(float((X @ np.array(COEF) + 0.25).mean()), abs=1e-3)
    assert r["model_inputs"] == pytest.approx(dict(zip(COLS, X.mean(axis=0))), abs=1e-5)


def test_someone_without_a_baseline_gets_not_enough_data(sessions, model):
    r = score_ananya(as_page_sends(sessions[0], "A Judge", 12), model)
    assert passes_page_checks(r) and r["model"] == "not enough data yet" and "Ananya" in r["note"]


def test_the_hour_picks_the_phase():
    assert [phase_of(h) for h in (16, 17, 23, 0, 3, 4, 12)] == ["post", "pre", "pre", "pre", "pre", "post", "post"]
    start = int(datetime(2026, 11, 8, 3, 30, tzinfo=timezone.utc).timestamp() * 1000)
    assert local_hour({"clock": {"start_epoch_ms": start, "tz_offset_min": 60}}) == 4
    assert local_hour({"clock": {"start_local_iso": "2026-11-07T18:46:41+00:00"}}) == 18


def test_too_little_typing_is_refused(sessions, model):
    s = as_page_sends(sessions[0], "Zoe", 12)
    with pytest.raises(BadSession):
        score_ananya(dict(s, keys=s["keys"][:10]), model)


def test_scores_are_stored_under_the_scrambled_key(sessions, model):
    store = MemoryStore()
    r = score_ananya(as_page_sends(sessions[1], "Zoe", 12), model, store=store)
    rows = store.history(teammate_key("Zoe"))
    assert r["history"]["saved"] and len(rows) == 1 and rows[0]["model"] == "ananya model" and rows[0]["adj"] == {}
    score_ananya(as_page_sends(sessions[1], "A Judge", 12), model, store=store)  # no score, nothing stored
    assert len(store.rows) == 1


def test_model_file_holds_no_names(model):
    text = json.dumps(model)
    assert "ZOE" not in text.upper() and model["kind"] == "ananya" and model["key_tasks"] is None
