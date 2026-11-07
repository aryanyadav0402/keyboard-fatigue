"""The fatigue index: a first-guess tiredness model for someone with no baseline, such as a judge
trying the demo once.

Without a person's own normal, their typing can only be compared with everyone else's, so typing
skill and habit get mixed up with tiredness. With a panel of only a few people, a model with many
fitted weights mostly learns who those few people are. This one has very few moving parts:

1. Five features from the one attempt, each measured against the person's own typing in that
   attempt and adjusted for the rule type: the rule's cost, rhythm unevenness, pauses relative to
   their own pace, backspaces and accuracy.
2. Each is turned into a robust z-score against the panel's spread, given a direction fixed in
   advance (a bigger rule cost, a more uneven rhythm, more pauses, more corrections or lower
   accuracy all mean more tired), and averaged into one index.
3. Two numbers fitted on the panel, a slope that can't go negative and an intercept, turn the index
   into KSS. If the panel shows no link, the slope is 0 and everyone gets the panel's average.
"""

import warnings

import numpy as np

# +1 means a higher value suggests more tired. Fixed before seeing any real data.
DIRECTIONS = {
    "vt_rule_cost_adj": 1,
    "vt_gap_cv_adj": 1,
    "vt_pause_rel_share_adj": 1,
    "backspace_per100_adj": 1,
    "vt_accuracy_adj": -1,
}
FIRST_GUESS = list(DIRECTIONS)
Z_LIMIT = 4  # one wild feature shouldn't decide the score
LABELS = {  # for showing people what drove their index
    "vt_rule_cost_adj": "slowdown from the rule",
    "vt_gap_cv_adj": "uneven rhythm",
    "vt_pause_rel_share_adj": "pauses",
    "backspace_per100_adj": "corrections",
    "vt_accuracy_adj": "accuracy",
}


class FatigueIndex:
    """Average of signed robust z-scores, mapped to KSS with a fitted line (scikit-learn style)."""

    def __init__(self, columns: list[str]):
        self.columns = list(columns)

    def fit(self, X, y) -> "FatigueIndex":
        X, y = np.asarray(X, float), np.asarray(y, float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # a column with nothing measured
            self.center_ = np.nanmedian(X, axis=0)
            mad = 1.4826 * np.nanmedian(np.abs(X - self.center_), axis=0)
            spread = np.nanstd(X, axis=0)
        self.center_ = np.nan_to_num(self.center_)
        self.scale_ = np.where(mad > 1e-9, mad, np.where(spread > 1e-9, spread, 1.0))
        self.signs_ = np.array([DIRECTIONS.get(c, 1) for c in self.columns], float)
        index = self.index(X)
        ok = ~np.isnan(index) & ~np.isnan(y)
        self.slope_, self.intercept_ = 0.0, float(np.nanmean(y)) if ok.any() else 5.0
        if ok.sum() >= 3 and np.ptp(index[ok]) > 0:
            slope, intercept = np.polyfit(index[ok], y[ok], 1)
            if slope > 0:  # a negative slope would contradict the directions, so fall back to the average
                self.slope_, self.intercept_ = float(slope), float(intercept)
            else:
                self.intercept_ = float(y[ok].mean())
        return self

    def contributions(self, X) -> np.ndarray:
        """Each feature's signed, capped z-score: above 0 pushes the index towards more tired."""
        return np.clip((np.asarray(X, float) - self.center_) / self.scale_ * self.signs_, -Z_LIMIT, Z_LIMIT)

    def index(self, X) -> np.ndarray:
        """The fatigue index: 0 is the panel's typical attempt, higher is more tired-looking."""
        z = self.contributions(X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # a row with nothing measured gives NaN
            index = np.nanmean(z, axis=1)  # features that couldn't be measured are skipped
        return index

    def predict(self, X) -> np.ndarray:
        index = np.nan_to_num(self.index(X))  # nothing measurable: treat as a typical attempt
        return self.intercept_ + self.slope_ * index


def describe(model: FatigueIndex, panel_X, x) -> dict | None:
    """One attempt's fatigue index as a percentile of the panel's check-ins, with what drove it.

    This is what the demo shows someone with no baseline: how their typing compares with the team's
    check-ins over a night, rather than a tiredness rating converted from a handful of people.
    """
    return describe_from_index(model, model.index(np.asarray(panel_X, float)), x)


def describe_from_index(model: FatigueIndex, panel_index, x) -> dict | None:
    """describe(), given the panel's index values instead of their features (as a frozen model keeps them)."""
    x = np.asarray(x, float).reshape(1, -1)
    value = model.index(x)[0]
    panel = np.asarray(panel_index, float)
    panel = panel[~np.isnan(panel)]
    if np.isnan(value) or len(panel) == 0:
        return None
    percentile = int(round(100 * float(np.mean(panel < value))))
    drivers = []
    for col, z, you, typical in zip(model.columns, model.contributions(x)[0], x[0], model.center_):
        if np.isnan(z):
            continue  # this feature couldn't be measured in the attempt
        drivers.append({"feature": col, "label": LABELS.get(col, col), "z": round(float(z), 2),
                        "pushes": "more tired" if z > 0 else "less tired" if z < 0 else "neither way",
                        "strength": "clearly" if abs(z) >= 1.5 else "a bit" if abs(z) >= 0.5 else "about typical",
                        "you": round(float(you), 3), "typical": round(float(typical), 3)})
    drivers.sort(key=lambda d: -abs(d["z"]))
    return {
        "value": round(float(value), 2),
        "percentile": percentile,
        "text": f"Your typing looks more tired than {percentile}% of check-ins from our team's all-nighter.",
        "panel_checkins": int(len(panel)),
        "drivers": drivers,
    }
