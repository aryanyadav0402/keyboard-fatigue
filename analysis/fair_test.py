"""A fair test of our model (v1) against a single SWELL stress score (v3), fixed before new data came in.

At 01:00 on 8 Nov we compared seven models on the 34 check-ins collected by 00:37. One SWELL stress
score did about as well as our 6-feature model, but we picked it after trying seven versions on 22
scored check-ins. So this test was committed before any later check-in was looked at: every model is
fitted only on check-ins up to CUTOFF and scored only on panel check-ins after it. Don't change it
after seeing its results.

    uv run python export_features.py --data <folder> --out results/<name>
    uv run python fair_test.py results/<name>

v2 is Ananya's two numbers, typing speed and backspaces. v3 mixes the same two numbers with her
SWELL stress recipe (`swell_model_weights.json`, commit 1879084) and learns one weight from our team.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from model import CLOCK, TYPING, ridge, scored

CUTOFF = pd.Timestamp("2026-11-08 00:18:48")  # the latest check-in in the 00:37 batch
# Ananya's SWELL stress recipe, in units of each number's typical change within a person
SWELL_STRESS = {"rate": -0.09978783562013131, "errr": -0.32602494462135545}
MODELS = {  # name: (features, whether it predicts the change from the person's baseline rating)
    "Guess the average": ([], False),
    "Clock": (CLOCK, True),
    "v1 ours: 6 typing features": (TYPING, True),
    "v2 hers: speed + backspaces": (["h_rate_dev", "h_errr_dev"], True),
    "Backspaces only": (["h_errr_dev"], True),
    "v3 SWELL stress score": (["swell_stress"], True),
}


def add_her_numbers(f: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Speed (log words per minute) and backspaces (square root per key), as changes from each
    person's first two check-ins."""
    raw = raw.set_index("checkin_id").reindex(f.checkin_id)
    her = pd.DataFrame({"rate": np.log(raw.vt_wpm.to_numpy(float)),
                        "errr": np.sqrt(raw.vt_n_backspace.to_numpy(float) / raw.vt_n_keys.to_numpy(float))},
                       index=f.index)
    f = f.copy()
    for col in her:
        f[f"h_{col}_dev"] = her[col] - her[col].where(f.is_baseline).groupby(f.person).transform("mean")
    return f


def fair_test(f: pd.DataFrame, raw: pd.DataFrame, cutoff: pd.Timestamp = CUTOFF) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit every model on scored check-ins up to the cut-off and score the ones after it."""
    d = scored(add_her_numbers(f.assign(is_baseline=f.is_baseline.astype(bool)), raw))
    train = pd.to_datetime(d.timestamp).le(cutoff).to_numpy()
    test = d[~train]
    if test.empty:
        return pd.DataFrame(), test
    spread = {c: d.loc[train, f"h_{c}_dev"].std() for c in SWELL_STRESS}  # training check-ins only
    d["swell_stress"] = sum(w * d[f"h_{c}_dev"] / spread[c] for c, w in SWELL_STRESS.items())
    kss = test.kss.to_numpy(float)
    guess_error = ((d.kss[train].mean() - kss) ** 2).sum()
    rows = {}
    for name, (cols, anchored) in MODELS.items():
        y = (d.kss_delta if anchored else d.kss).to_numpy(float)
        if cols:
            X = d[cols].to_numpy(float)
            pred = ridge().fit(X[train], y[train]).predict(X[~train])
        else:
            pred = np.full(len(test), y[train].mean())
        if anchored:
            pred = pred + test.kss_base.to_numpy(float)
        err = np.clip(pred, 1, 9) - kss
        rows[name] = {"error (KSS)": np.abs(err).mean(),
                      "skill": 1 - (err ** 2).sum() / guess_error if guess_error > 0 else np.nan}
    return pd.DataFrame(rows).T, test


def main() -> None:
    out = Path(sys.argv[1])
    table, test = fair_test(pd.read_csv(out / "features_processed.csv"), pd.read_csv(out / "features_raw.csv"))
    if table.empty:
        print(f"No scored panel check-ins after {CUTOFF} yet.")
        return
    print(f"Fitted on check-ins up to {CUTOFF}; scored on {len(test)} panel check-ins after it, "
          f"from {test.person.nunique()} people. Error is in KSS points; skill 0 = guessing the average.\n")
    print(table.round(2).to_string())


if __name__ == "__main__":
    main()
