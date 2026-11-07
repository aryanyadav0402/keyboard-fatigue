"""Compare first-guess models for people with no baseline, leaving one panel person out at a time.

    uv run python compare_first_guess.py --data ../data/sessions

Each panel person is predicted as if they were a stranger: by a model trained on the other panel
people's check-ins, using only features that need no baseline. Walk-ups, if there are any, are
predicted by models trained on the whole panel.
"""

import argparse

import numpy as np
import pandas as pd

from features import build_features
from first_guess import FIRST_GUESS, FatigueIndex
from load_data import load_sessions
from model import metrics, present, ridge

OLD = ["vt_rule_cost_adj", "vt_gap_cv_adj", "vt_accuracy_adj", "backspace_per100_adj"]
CANDIDATES = {
    "Always guess the panel average": ("mean", []),
    "Ridge, 4 features (before)": ("ridge", OLD),
    "Ridge, 5 features": ("ridge", FIRST_GUESS),
    "Fatigue index, 5 features": ("index", FIRST_GUESS),
}


def fit_predict(kind: str, cols: list[str], train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    if kind == "mean" or not cols:
        return np.full(len(test), train.kss.mean())
    model = ridge() if kind == "ridge" else FatigueIndex(cols)
    model.fit(train[cols].to_numpy(float), train.kss.to_numpy(float))
    return np.clip(model.predict(test[cols].to_numpy(float)), 1, 9)


def compare(f: pd.DataFrame) -> pd.DataFrame:
    """One row per candidate: errors predicting each panel person as a stranger, and walk-ups."""
    panel = f[f.participant_type.eq("panel") & f.kss.notna()].reset_index(drop=True)
    walkups = f[f.participant_type.eq("oneoff") & f.kss.notna()].reset_index(drop=True)
    rows = {}
    for name, (kind, wanted) in CANDIDATES.items():
        cols = present(panel, wanted)
        pred = np.full(len(panel), np.nan)
        for person in panel.person.unique():
            test = (panel.person == person).to_numpy()
            pred[test] = fit_predict(kind, cols, panel[~test], panel[test])
        row = metrics(panel, pred)
        if len(walkups):
            walkup_pred = fit_predict(kind, cols, panel, walkups)
            row["Walk-up MAE"] = float(np.mean(np.abs(walkup_pred - walkups.kss.to_numpy(float))))
        rows[name] = row
    return pd.DataFrame(rows).T


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare first-guess (no-baseline) tiredness models.")
    parser.add_argument("--data", required=True, help="folder of typing-station session files (*.json)")
    args = parser.parse_args()
    f = build_features(load_sessions(args.data))
    panel = f[f.participant_type.eq("panel") & f.kss.notna()]
    print(f"{len(panel)} panel check-ins from {panel.person.nunique()} people, each predicted as a stranger; "
          f"{int(f.participant_type.eq('oneoff').sum())} walk-up check-ins")
    print(compare(f).round(2).to_string())


if __name__ == "__main__":
    main()
