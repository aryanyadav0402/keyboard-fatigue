"""Write the exact numbers of the candidate this track would freeze (ridge_fg5) fitted on all 33 panel
check-ins up to the cut-off, so it can be scored on the morning batch without refitting.

    uv run --project <analysis> python frozen_spec.py

Prediction for one attempt:
  1. raw features from the rule task (features.py's typing_features): vt_rule_cost, vt_gap_cv,
     vt_pause_rel_share, backspace_per100, vt_accuracy;
  2. rule adjustment with the references below: rule cost and accuracy divided by the rule's
     typical value; the other three minus the rule's typical value plus the overall typical value
     (a rule type not listed uses the overall value);
  3. a missing feature takes the training median; z = (x - mean) / sd;
  4. KSS = intercept + sum(weight * z), clipped to 1..9.
Writes frozen_spec.json (numbers only: no names, times or keystrokes).
"""

import json

import numpy as np

from candidates import ridge_predict
from common import FG5, HERE, RATIO_COLS, load, rule_adjust
from diagnostics import chosen_alpha

RAW = {"vt_rule_cost_adj": "vt_rule_cost", "vt_gap_cv_adj": "vt_gap_cv", "vt_pause_rel_share_adj": "vt_pause_rel_share",
       "backspace_per100_adj": "backspace_per100", "vt_accuracy_adj": "vt_accuracy"}


def main():
    panel, walk = load()
    y = panel.kss.to_numpy(float)
    d = rule_adjust(panel, panel)
    alpha = chosen_alpha(d, y, FG5)
    X = d[FG5].to_numpy(float)
    med = np.nanmedian(X, 0)
    Xf = np.where(np.isnan(X), med, X)
    mu, sd = Xf.mean(0), Xf.std(0)
    A = (Xf - mu) / sd
    w = np.linalg.solve(A.T @ A + alpha * np.eye(len(FG5)), A.T @ (y - y.mean()))
    refs = {raw: {"rule": "divide" if raw in RATIO_COLS else "subtract, add overall",
                  "typical": {k: float(v) for k, v in panel.groupby("question_type")[raw].median().items()},
                  "overall": float(panel[raw].median())} for raw in RAW.values()}
    spec = {"name": "ridge_fg5", "fitted_on": "33 panel check-ins up to 2026-11-08 00:53:51 (P1-P5), baselines included",
            "features": FG5, "rule_references": refs, "impute_median": dict(zip(FG5, med.round(6))),
            "mean": dict(zip(FG5, mu.round(6))), "sd": dict(zip(FG5, sd.round(6))),
            "alpha": alpha, "weight_per_sd": dict(zip(FG5, w.round(4))), "intercept": round(float(y.mean()), 4),
            "clip": [1, 9]}
    # self-check: the formula above reproduces the fitted model
    p_formula = np.clip(y.mean() + ((np.where(np.isnan(X), med, X) - mu) / sd) @ w, 1, 9)
    p_model = np.clip(ridge_predict(X, y, X, alpha), 1, 9)
    assert np.allclose(p_formula, p_model)
    wx = rule_adjust(panel, walk)[FG5].to_numpy(float)
    spec["walkup_predictions"] = np.round(np.clip(y.mean() + ((np.where(np.isnan(wx), med, wx) - mu) / sd) @ w, 1, 9), 2).tolist()
    spec["panel_prediction_range_in_sample"] = [round(float(p_model.min()), 2), round(float(p_model.max()), 2)]
    (HERE / "frozen_spec.json").write_text(json.dumps(spec, indent=2, default=float) + "\n")
    print(json.dumps({k: spec[k] for k in ("alpha", "weight_per_sd", "intercept", "walkup_predictions",
                                           "panel_prediction_range_in_sample")}, default=float))


if __name__ == "__main__":
    main()
