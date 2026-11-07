"""Freeze the trained models into a small file of numbers, and score demo attempts with that file.

export_model() runs on a laptop that has the team's check-ins: it fits the models the usual way
and keeps only numbers (the typical value of each feature per rule, the first-guess index, the
personal model's weights, and the team's index values for percentiles). No names, timestamps or
keystrokes go into the file.

score_attempt() runs anywhere, including the hosted API in service/: it needs only the model file
and the feature code, not the team's data or scikit-learn. Its reply has the same shape as the
local POST /api/score (demo.py), plus `best_reference` and `model_version`. Because a hosted API
can't keep a database, the page keeps `best_reference` for each username and sends it back with
that user's next attempt.

    uv run python export_model.py --data ../data/sessions --name real-2026-11-08
"""

from datetime import datetime

import numpy as np
import pandas as pd

from attempt import (NOTES, REFERENCE, SCALE, SHOWN, BadSession, check_typing, clean, compared_with_best,
                     kss_wording, read_attempt)
from features import apply_rule_references, build_features, rule_references
from first_guess import LABELS, FatigueIndex, describe_from_index
from load_data import Tables

FORMAT = "keyboard-fatigue-model/1"


def plain(x):
    """Numbers and lists for JSON: NaN becomes None."""
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [plain(v) for v in x]
    if x is None or (isinstance(x, (float, np.floating)) and np.isnan(x)):
        return None
    if isinstance(x, (np.integer, int)) and not isinstance(x, bool):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return float(x)
    return x


def floats(values) -> np.ndarray:
    return np.array([np.nan if v is None else float(v) for v in values], float)


def export_model(f: pd.DataFrame, name: str, note: str = "") -> dict:
    """The trained models as plain numbers. Needs the team's features and scikit-learn (laptop only)."""
    from model import display_models, scored  # scikit-learn: never needed where the model is served

    m = display_models(f)
    panel = f[f.participant_type.eq("panel") & f.kss.notna()]
    first_guess = None
    if m["fallback"] is not None:
        fg = m["fallback"]
        first_guess = {
            "columns": list(fg.columns), "center": plain(fg.center_), "scale": plain(fg.scale_),
            "signs": plain(fg.signs_), "slope": float(fg.slope_), "intercept": float(fg.intercept_),
            # the team's index values, sorted, for percentiles; just numbers, nothing per person
            "panel_index": plain(np.sort(fg.index(panel[m["fallback_cols"]].to_numpy(float)))),
        }
    personal = None
    if m["main"] is not None:
        steps = m["main"].named_steps
        personal = {
            "features": list(m["typing"]),
            "fill": plain(np.nan_to_num(steps["simpleimputer"].statistics_, nan=0.0)),
            "mean": plain(steps["standardscaler"].mean_), "scale": plain(steps["standardscaler"].scale_),
            "coef": plain(steps["ridgecv"].coef_), "intercept": float(steps["ridgecv"].intercept_),
        }
    typical = {}
    if "question_type" in panel:
        for rule, group in panel.groupby("question_type"):
            typical[str(rule)] = {label: clean(group[col].median()) for col, label in SHOWN.items()
                                  if col in group and group[col].notna().any()}
    return plain({
        "format": FORMAT,
        "name": name,
        "note": note,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "trained_on": {"checkins": len(f), "panel_checkins": len(panel),
                       "panel_people": int(panel.person.nunique()), "scored_checkins": len(scored(f))},
        "rule_references": rule_references(f),
        "first_guess": first_guess,
        "personal": personal,
        "typical_for_rule": typical,
    })


def check_model(model: dict) -> dict:
    if not isinstance(model, dict) or model.get("format") != FORMAT:
        raise ValueError(f"not a {FORMAT} file")
    return model


def first_guess_index(model: dict) -> FatigueIndex | None:
    params = model.get("first_guess")
    if not params:
        return None
    fg = FatigueIndex(params["columns"])
    fg.center_, fg.scale_, fg.signs_ = floats(params["center"]), floats(params["scale"]), floats(params["signs"])
    fg.slope_, fg.intercept_ = float(params["slope"]), float(params["intercept"])
    return fg


def personal_predict(params: dict, x) -> float:
    """The personal model's prediction: fill gaps, standardise, then the ridge regression's weights."""
    x = floats(x)
    x = np.where(np.isnan(x), floats(params["fill"]), x)
    return float((x - floats(params["mean"])) / floats(params["scale"]) @ floats(params["coef"]) + params["intercept"])


def attempt_row(parsed: dict, model: dict) -> pd.Series:
    """This attempt's features, adjusted for its rule against the model's typical values."""
    one = Tables(sessions=pd.DataFrame([parsed["session"]]), units=pd.DataFrame(parsed["units"]),
                 keys=pd.DataFrame(parsed["keys"]), pvt=pd.DataFrame(parsed["pvt"]))
    f = apply_rule_references(build_features(one), model["rule_references"])
    return f.iloc[0]


def check_best_reference(best) -> dict | None:
    if best is None:
        return None
    if not isinstance(best, dict) or not isinstance(best.get("score"), (int, float)) \
            or not isinstance(best.get("adj"), dict):
        raise BadSession("'best_reference' should be the object a previous reply returned")
    return best


def score_attempt(body: dict, model: dict, best_reference: dict | None = None) -> dict:
    """Score one posted attempt with a frozen model, comparing it with the user's best attempt if sent."""
    parsed, username = read_attempt(body)
    row = attempt_row(parsed, model)
    check_typing(row, parsed)
    best = check_best_reference(best_reference if best_reference is not None else body.get("best_reference"))
    best = best if username else None  # like the local server: comparisons need a username
    if best is not None and best.get("model_version") != model["name"]:
        best = None  # a best from another model isn't comparable, so the user starts again with this one

    fg = first_guess_index(model)
    score, model_name, inputs = None, "not enough data yet", {}
    if best is not None and model.get("personal"):
        params = model["personal"]
        attempt, repeat = compared_with_best(row, best["adj"], params["features"])
        change = personal_predict(params, attempt) - personal_predict(params, repeat)
        score, model_name = float(np.clip(best["score"] + change, 1, 9)), "best-attempt model"
        inputs = dict(zip(params["features"], attempt))
    elif fg is not None:
        X = pd.to_numeric(row.reindex(fg.columns), errors="coerce").to_numpy(float).reshape(1, -1)
        score, model_name = float(np.clip(fg.predict(X)[0], 1, 9)), "no-baseline model"
        inputs = dict(zip(fg.columns, X[0]))

    shown = {label: clean(row.get(col)) for col, label in SHOWN.items()}  # every key, null when not measured
    best_info = new_reference = None
    if username and score is not None:
        now = datetime.now().isoformat(timespec="seconds")
        new_best = best is None or score < best["score"]
        attempts = int(best.get("attempts", 0)) + 1 if best else 1
        if new_best:
            # unrounded, so repeating the best attempt exactly gives exactly the best score
            adj = {c: float(row[c]) if pd.notna(row.get(c)) else None for c in REFERENCE if c in row.index}
            new_reference = {"score": score, "rule": row.question_type, "adj": adj, "raw": shown,
                             "attempts": attempts, "best_at": now, "model_version": model["name"]}
        else:
            new_reference = dict(best, attempts=attempts)
        best_info = {"score": round(new_reference["score"], 1), "attempts": attempts, "new_best": new_best,
                     "best_at": new_reference.get("best_at")}
    scaled = None
    if best is not None:
        best_raw = best.get("raw") or {}
        scaled = {label: (round(value / best_raw[label], 3) if value is not None and best_raw.get(label) else None)
                  for label, value in shown.items()}
    index_info = None  # the team comparison is only for people with no baseline
    if model_name == "no-baseline model":
        index_info = describe_from_index(fg, model["first_guess"]["panel_index"], X)
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
        "typical_for_this_rule": model.get("typical_for_rule", {}).get(str(row.question_type), {}),
        "model_inputs": {k: clean(v) for k, v in inputs.items()},
        "best": best_info,
        "best_reference": new_reference,
        "model_version": model["name"],
        "saved_as_training_data": False,
    }


def describe_model(model: dict) -> dict:
    """A short public summary of a model file, for GET /api/models."""
    return {"name": model["name"], "note": model.get("note", ""), "created_at": model.get("created_at"),
            "trained_on": model.get("trained_on"), "first_guess_ready": bool(model.get("first_guess")),
            "personal_ready": bool(model.get("personal")),
            "first_guess_follows_tiredness": bool(model.get("first_guess") and model["first_guess"]["slope"] > 0),
            "rules": sorted(model.get("typical_for_rule", {})), "labels": LABELS}
