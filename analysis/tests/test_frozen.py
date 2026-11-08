import json

import numpy as np
import pytest

from attempt import BadSession
from demo import score_session
from features import build_features
from frozen_model import describe_model, export_model, score_attempt
from load_data import load_sessions
from simulate import make_people, make_schedule, make_session, make_states


@pytest.fixture(scope="module")
def night(tmp_path_factory):
    """A simulated night in the app's format: 7 teammates and 20 walk-ups."""
    folder = tmp_path_factory.mktemp("night")
    rng, world = np.random.default_rng(11), np.random.default_rng(12)
    for p in make_people(rng):
        times, sleep = make_schedule(rng, p)
        for st in make_states(rng, p, times, sleep):
            session = make_session(world, p, st, effect=True)
            (folder / f"{p['id']}_{st['n']:02d}_{session['session_id']}.json").write_text(json.dumps(session))
    return folder


@pytest.fixture(scope="module")
def frozen(night):
    """The exported model, after a trip through JSON like the real file."""
    return json.loads(json.dumps(export_model(build_features(load_sessions(night)), "test-model")))


def demo_body(folder, who, n=1, username=None):
    """The short form the demo page sends, from one simulated session's typing."""
    session = json.loads(sorted(folder.glob(f"{who}_{n:02d}_*.json"))[0].read_text())
    unit = session["units"][0]
    keys = [{"type": "down" if k[2] == "d" else "up", "key": k[5], "code": k[4], "t": k[3]}
            for k in session["keys"] if k[0] == 0]
    return {"username": username, "sentence": unit["source"], "rule": unit["question_type"],
            "typed": unit["typed"], "shown_at": unit["t_shown"], "keys": keys}


def walkups(folder):
    return sorted({p.name.split("_")[0] for p in folder.glob("WALKUP-*.json")})


def test_stored_history_gives_a_baseline_without_the_page_sending_anything(night, frozen):
    from attempt_store import MemoryStore
    store = MemoryStore()
    first = demo_body(night, walkups(night)[0], username="Visitor One")
    r1 = score_attempt(first, frozen, store=store)
    assert r1["model"] == "no-baseline model" and r1["history"] == {"saved": True, "attempts": 1}
    r2 = score_attempt(first, frozen, store=store)  # typing just like the first try, nothing sent back
    assert r2["model"] == "best-attempt model" and "first attempts" in r2["note"]
    assert r2["score"] == pytest.approx(r1["score"], abs=0.051) and r2["history"]["attempts"] == 2
    r3 = score_attempt(demo_body(night, walkups(night)[1], username="visitor one"), frozen, store=store)
    assert r3["model"] == "best-attempt model" and r3["history"] == {"saved": True, "attempts": 3}
    # numbers under one scrambled key: no name, keystrokes or typed text
    dump = json.dumps(store.rows)
    assert "VISITOR" not in dump.upper() and '"keys"' not in dump and first["typed"] not in dump
    assert len({r["user_key"] for r in store.rows}) == 1


def test_a_database_outage_still_scores(night, frozen):
    class Down:
        def history(self, key):
            raise ConnectionError("down")

        def add(self, record):
            raise ConnectionError("down")

    reply = score_attempt(demo_body(night, walkups(night)[0], username="someone"), frozen, store=Down())
    assert reply["model"] == "no-baseline model" and reply["history"] == {"saved": False, "attempts": None}


def test_database_urls_keep_only_settings_libpq_knows():
    from attempt_store import clean_url
    url = "postgres://u:p@db.example:6543/postgres?sslmode=require&supa=base-pooler.x&pgbouncer=true"
    assert clean_url(url) == "postgres://u:p@db.example:6543/postgres?sslmode=require"


def test_teammate_gets_the_personal_model_from_the_first_attempt(night, frozen):
    from model import display_models
    f = build_features(load_sessions(night))
    main = display_models(f)
    who = sorted({p.name.split("_")[0] for p in night.glob("PANEL-*.json")})[1]
    row = f[(f.person == who) & ~f.is_baseline.astype(bool) & f.kss.notna()].sort_values("timestamp").iloc[2]
    path = next(night.glob(f"{who}_*_{row.checkin_id}.json"))
    n = int(path.name.split("_")[1])
    expected = np.clip(row.kss_base + main["main"].predict(row[main["typing"]].to_numpy(float).reshape(1, -1))[0], 1, 9)
    reply = score_attempt(demo_body(night, who, n, username=who.lower()), frozen)  # typed in any case
    assert reply["model"] == "best-attempt model" and "last night" in reply["note"]  # the label pages know
    assert reply["fatigue_index"] is None
    assert reply["score"] == pytest.approx(expected, abs=0.06)
    stranger = score_attempt(demo_body(night, who, n, username="someone new"), frozen)
    assert stranger["model"] == "no-baseline model"
    assert len(frozen["teammates"]) == f.loc[f.participant_type.eq("panel"), "person"].nunique()


def test_frozen_first_guess_matches_the_local_server(night, frozen):
    train = load_sessions(night)
    for who in walkups(night)[:5]:
        body = demo_body(night, who)
        local, hosted = score_session(body, train), score_attempt(body, frozen)
        assert hosted["model"] == local["model"] == "no-baseline model"
        assert hosted["score"] == pytest.approx(local["score"], abs=0.05)
        assert hosted["fatigue_index"]["percentile"] == pytest.approx(local["fatigue_index"]["percentile"], abs=1)
        assert hosted["features"] == local["features"]
        assert hosted["model_version"] == "test-model"


def test_frozen_best_attempt_matches_the_local_server(night, frozen, tmp_path):
    assert frozen["personal"], "the simulated night should have enough data for the personal model"
    train, db = load_sessions(night), tmp_path / "local.sqlite"
    first, second = demo_body(night, "PANEL-01", 1, "zoe"), demo_body(night, "PANEL-01", 9, "zoe")
    score_session(first, train, db)
    local = score_session(second, train, db)
    hosted_first = score_attempt(first, frozen)
    hosted = score_attempt(second, frozen, hosted_first["best_reference"])
    assert local["model"] == hosted["model"] == "best-attempt model"
    assert hosted["score"] == pytest.approx(local["score"], abs=0.05)
    assert hosted["best"]["attempts"] == local["best"]["attempts"] == 2
    assert hosted["fatigue_index"] is None and hosted["scaled_features"]


def test_best_reference_keeps_the_most_alert_attempt(night, frozen):
    body = demo_body(night, "PANEL-02", 1, "sam")
    first = score_attempt(body, frozen)
    assert first["best"] == {"score": first["score"], "attempts": 1, "new_best": True, "best_at": first["best"]["best_at"]}
    again = score_attempt(body, frozen, first["best_reference"])  # typing exactly like the best attempt
    assert again["score"] == pytest.approx(first["score"]) and not again["best"]["new_best"]
    assert again["best"]["attempts"] == 2 and again["best_reference"]["score"] == first["best_reference"]["score"]
    no_name = score_attempt(dict(body, username=None), frozen, first["best_reference"])
    assert no_name["best"] is None and no_name["model"] == "no-baseline model"


def test_best_from_another_model_starts_the_user_again(night, frozen):
    body = demo_body(night, "PANEL-02", 1, "sam")
    first = score_attempt(body, frozen)
    old = dict(first["best_reference"], model_version="an-older-model")
    again = score_attempt(body, frozen, old)
    assert again["model"] == "no-baseline model" and again["fatigue_index"] is not None
    assert again["best"]["attempts"] == 1 and again["best"]["new_best"]
    assert again["best_reference"]["model_version"] == frozen["name"]


def test_bad_requests_are_refused(night, frozen):
    with pytest.raises(BadSession):
        score_attempt([], frozen)
    with pytest.raises(BadSession):
        score_attempt(demo_body(night, "PANEL-01", 1, "zoe"), frozen, {"score": "high"})


def test_model_file_holds_only_numbers_no_names(night, frozen):
    text = json.dumps({k: v for k, v in frozen.items() if k != "created_at"})  # the export's own time is fine
    assert "PANEL-" not in text and "WALKUP-" not in text and "2026-11-0" not in text
    summary = describe_model(frozen)
    assert summary["first_guess_ready"] and summary["personal_ready"] and len(summary["rules"]) == 5
