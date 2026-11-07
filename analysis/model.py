"""Tiredness models and their evaluation, leaving one panel person out at a time.

Predictions use typing only, because the demo can't ask people how long they've been awake.
Hours awake is still recorded, and the evaluation uses it in two ways:

- as a benchmark: typing alone is compared with hours awake alone, under the same
  leave-one-person-out split;
- as a clock check: practice, tiredness and the clock all drift together over one night, so a
  permutation test shuffles the typing features within each person, keeping the ratings and hours
  awake in place, and asks whether typing adds information beyond the clock.

The original test from the plan (shuffling ratings within each person) is kept for comparison,
because typing that only drifts with practice can fool it.

People use the KSS scale differently, so models predict each person's change from their own
baseline rating and then add the baseline back ("anchored").
"""

import re
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, permutation_test_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ALPHAS = np.logspace(-2, 3, 30)
# The 6 typing features picked on 7 Nov, before any real results (free-text thinking pauses were
# dropped because the app has no free-text task); no feature selection happens on
# the data. They are columns of features_processed.csv: changes from each person's first two
# check-ins (_dev), and backspaces per 100 keys as they are.
TYPING = ["vt_rule_cost_dev", "backspace_per100", "vt_accuracy_dev", "vt_gap_cv_dev",
          "vt_pause1s_share_dev", "vt_hold_med_dev"]
RULE = ["vt_rule_cost_dev"]
RT = ["rt_lapse_rate_dev"]  # the reaction test, for one comparison row
CLOCK = ["hours_rel"]  # hours awake relative to the baseline check-ins: a benchmark, never a prediction input
EXTERNAL = ["vt_rule_cost", "vt_gap_cv", "vt_accuracy", "backspace_per100"]  # need no baseline
MAIN = "Typing only (main)"
MIN_PEOPLE, MIN_CHECKINS, MIN_PER_PERSON = 3, 15, 3
MIXED = {
    "Tiredness": "kss ~ hours_awake + log_checkin + vt_gap_cv_dev + vt_rule_cost_dev",
    "Rule cost": "vt_rule_cost ~ hours_awake + log_checkin",
}


def ridge():
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                         RidgeCV(alphas=ALPHAS))


def logistic():
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                         LogisticRegression(class_weight="balanced", max_iter=1000))


def present(f: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in f and f[c].notna().any()]


def scored(f: pd.DataFrame) -> pd.DataFrame:
    """Panel check-ins we predict: rated, not a baseline check-in, and with some typing data."""
    typing = [c for c in TYPING if c in f]
    if f.empty or not typing:
        return f.iloc[0:0]
    keep = f.participant_type.eq("panel") & ~f.is_baseline & f.kss.notna() & f[typing].notna().any(axis=1)
    return f[keep].reset_index(drop=True)


def missing_data(d: pd.DataFrame) -> list[str]:
    """What's missing before models can be fitted; empty when there's enough."""
    per_person = d.groupby("person").size() if len(d) else pd.Series(dtype=int)
    people = int((per_person >= MIN_PER_PERSON).sum())
    gaps = []
    if people < MIN_PEOPLE:
        gaps.append(f"{people} panel people have {MIN_PER_PERSON}+ scored check-ins; models need {MIN_PEOPLE}")
    if len(d) < MIN_CHECKINS:
        gaps.append(f"{len(d)} scored panel check-ins (after each person's 2 baseline ones); models need {MIN_CHECKINS}")
    return gaps


def out_of_fold(X: np.ndarray, y: np.ndarray, person: np.ndarray) -> np.ndarray:
    """Predict each person from a model trained on everyone else."""
    pred = np.empty(len(y))
    for p in np.unique(person):
        test = person == p
        pred[test] = ridge().fit(X[~test], y[~test]).predict(X[test]) if X.shape[1] else y[~test].mean()
    return pred


def predict(d: pd.DataFrame, features: list[str], anchored: bool = True) -> np.ndarray:
    y = (d.kss_delta if anchored else d.kss).to_numpy(float)
    pred = out_of_fold(d[features].to_numpy(float), y, d.person.to_numpy())
    if anchored:
        pred = pred + d.kss_base.to_numpy(float)
    return np.clip(pred, 1, 9)


def metrics(d: pd.DataFrame, pred: np.ndarray) -> dict:
    kss = d.kss.to_numpy(float)
    within, varies = [], False
    for p in d.person.unique():
        m = (d.person == p).to_numpy()
        varies |= bool(np.ptp(pred[m]) > 0)
        if m.sum() >= 4 and np.ptp(kss[m]) > 0 and np.ptp(pred[m]) > 0:
            within.append(stats.spearmanr(pred[m], kss[m]).statistic)
    return {
        "MAE": float(np.mean(np.abs(pred - kss))),
        # A prediction that never changes within a person has no ranking worth reporting.
        "Spearman": float(stats.spearmanr(pred, kss).statistic) if varies else np.nan,
        "Spearman within person": float(np.median(within)) if within else np.nan,
    }


def compare_models(d: pd.DataFrame, typing: list[str], rt: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    no_rule = [c for c in typing if c not in RULE]
    preds = {
        "Training mean": predict(d, [], anchored=False),
        "Own baseline rating": np.clip(d.kss_base.to_numpy(float), 1, 9),
        "Hours awake (benchmark)": predict(d, CLOCK),
        MAIN: predict(d, typing),
        "Typing + reaction test": predict(d, typing + rt),
        "Typing without rule cost": predict(d, no_rule),
        "Typing, not anchored": predict(d, typing, anchored=False),
        "Typing + hours awake (check only)": predict(d, CLOCK + typing),
    }
    table = pd.DataFrame({name: metrics(d, p) for name, p in preds.items()}).T
    return table, pd.DataFrame(preds, index=d.index)


def beyond_clock_test(d: pd.DataFrame, typing: list[str], n_perm: int, seed: int) -> dict:
    """Does adding typing to hours awake cut the error by more than shuffled typing does?"""
    rng = np.random.default_rng(seed)
    y, base, kss = d.kss_delta.to_numpy(float), d.kss_base.to_numpy(float), d.kss.to_numpy(float)
    person, clock, X = d.person.to_numpy(), d[CLOCK].to_numpy(float), d[typing].to_numpy(float)

    def error(features):
        return float(np.mean(np.abs(np.clip(out_of_fold(features, y, person) + base, 1, 9) - kss)))

    clock_mae = error(clock)
    observed = clock_mae - error(np.column_stack([clock, X]))
    groups = [np.flatnonzero(person == p) for p in np.unique(person)]
    null = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = X.copy()
        for idx in groups:
            shuffled[idx] = X[rng.permutation(idx)]
        null[i] = clock_mae - error(np.column_stack([clock, shuffled]))
    return {"clock_mae": clock_mae, "main_mae": clock_mae - observed, "improvement": observed,
            "null": null, "p": float((1 + np.sum(null >= observed)) / (1 + n_perm))}


def original_test(d: pd.DataFrame, typing: list[str], n_perm: int, seed: int) -> dict:
    """The plan's test: shuffle ratings within each person (scikit-learn's groups does exactly that)."""
    score, perm, p = permutation_test_score(
        ridge(), d[typing].to_numpy(float), d.kss.to_numpy(float), groups=d.person.to_numpy(),
        cv=LeaveOneGroupOut(), n_permutations=n_perm, scoring="neg_mean_absolute_error",
        random_state=seed, n_jobs=-1)
    return {"mae": -score, "null": -perm, "p": float(p)}


def tired_labels(d: pd.DataFrame) -> np.ndarray:
    """1 when a check-in is above that person's median KSS."""
    return (d.kss > d.groupby("person").kss.transform("median")).to_numpy(int)


def classify(d: pd.DataFrame, features: list[str]) -> tuple[float, np.ndarray]:
    tired, X, person = tired_labels(d), d[features].to_numpy(float), d.person.to_numpy()
    prob = np.full(len(d), np.nan)
    for p in np.unique(person):
        test = person == p
        if len(np.unique(tired[~test])) == 2:
            prob[test] = logistic().fit(X[~test], tired[~test]).predict_proba(X[test])[:, 1]
    ok = ~np.isnan(prob)
    auc = roc_auc_score(tired[ok], prob[ok]) if len(np.unique(tired[ok])) == 2 else np.nan
    return float(auc), prob


def mixed_models(f: pd.DataFrame) -> dict:
    panel = f[f.participant_type.eq("panel")]
    out = {}
    for name, formula in MIXED.items():
        cols = re.findall(r"[A-Za-z_]\w*", formula)
        if not all(c in panel for c in cols):
            out[name] = {"formula": formula, "error": "a column is missing from the data"}
            continue
        data = panel.dropna(subset=cols)
        if data.person.nunique() < MIN_PEOPLE or len(data) < MIN_CHECKINS:
            out[name] = {"formula": formula,
                         "error": f"not enough data ({len(data)} check-ins from {data.person.nunique()} people)"}
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = smf.mixedlm(formula, data, groups=data["person"]).fit(method=["powell", "lbfgs"])
        except Exception as err:  # small samples can make the fit fail outright
            out[name] = {"formula": formula, "error": f"did not fit ({err})"}
            continue
        terms = fit.fe_params.index
        ci = fit.conf_int().loc[terms]
        out[name] = {"formula": formula, "n": len(data), "people": data.person.nunique(),
                     "converged": bool(fit.converged),
                     "table": pd.DataFrame({"coef": fit.fe_params, "ci_low": ci[0], "ci_high": ci[1],
                                            "p": fit.pvalues.loc[terms]})}
    return out


def rule_control(f: pd.DataFrame) -> dict | None:
    """Positive control: are gaps next to a capital longer than ordinary gaps?"""
    if not {"vt_touch_gap_med", "vt_ord_gap_med"} <= set(f):
        return None
    d = f.dropna(subset=["vt_touch_gap_med", "vt_ord_gap_med"])
    if len(d) < 6:
        return None
    out = {"n": len(d), "median_ratio": float((d.vt_touch_gap_med / d.vt_ord_gap_med).median()),
           "p": float(stats.wilcoxon(d.vt_touch_gap_med, d.vt_ord_gap_med).pvalue)}
    if {"vt_wpm", "pt_wpm"} <= set(f):
        both = f.dropna(subset=["vt_wpm", "pt_wpm"])
        if len(both) >= 6:
            out["plain_wpm_ratio"] = float((both.vt_wpm / both.pt_wpm).median())
            out["plain_p"] = float(stats.wilcoxon(both.vt_wpm, both.pt_wpm).pvalue)
    return out


def sleep_pairs(f: pd.DataFrame, min_slept: float = 60) -> pd.DataFrame:
    """The check-in before and after each sleep of at least min_slept minutes."""
    rows = []
    panel = f[f.participant_type.eq("panel")].sort_values(["person", "timestamp"])
    for person, g in panel.groupby("person"):
        for i in range(1, len(g)):
            before, after = g.iloc[i - 1], g.iloc[i]
            if after.nap_since_last_min >= min_slept:
                rows.append({"person": person, "slept_min": after.nap_since_last_min,
                             "kss_before": before.kss, "kss_after": after.kss,
                             "rule_cost_before": before.get("vt_rule_cost"), "rule_cost_after": after.get("vt_rule_cost"),
                             "rt_median_before": before.get("rt_median"), "rt_median_after": after.get("rt_median")})
    return pd.DataFrame(rows)


def no_baseline_model(f: pd.DataFrame):
    """A model on features that need no baseline, for people with too few check-ins."""
    panel = f[f.participant_type.eq("panel") & f.kss.notna()]
    feats = present(panel, EXTERNAL)
    if len(panel) < MIN_CHECKINS or not feats:
        return None, feats
    return ridge().fit(panel[feats].to_numpy(float), panel.kss.to_numpy(float)), feats


def external_test(f: pd.DataFrame) -> dict | None:
    """Train on the panel, then predict the one-off people, who have no baseline."""
    oneoff = f[f.participant_type.eq("oneoff") & f.kss.notna()]
    model, feats = no_baseline_model(f)
    if model is None or oneoff.empty:
        return None
    pred = np.clip(model.predict(oneoff[feats].to_numpy(float)), 1, 9)
    kss, mean = oneoff.kss.to_numpy(float), f[f.participant_type.eq("panel")].kss.mean()
    return {"n": len(oneoff), "people": oneoff.person.nunique(), "features": feats,
            "MAE": float(np.mean(np.abs(pred - kss))), "constant_MAE": float(np.mean(np.abs(mean - kss))),
            "pred": pd.Series(pred, index=oneoff.index)}


def run_models(f: pd.DataFrame, n_perm: int = 500, seed: int = 0) -> dict:
    d = scored(f)
    r = {"scored": d, "missing": missing_data(d)}
    r["typing"] = present(d, TYPING)
    r["inactive"] = [c for c in TYPING if c not in r["typing"]]
    r["rt"] = present(d, RT)
    if f.empty:
        return r
    r["mixed"] = mixed_models(f)
    r["rule_control"] = rule_control(f)
    r["sleep"] = sleep_pairs(f)
    r["external"] = external_test(f)
    if r["missing"] or not r["typing"]:
        return r
    r["table"], r["preds"] = compare_models(d, r["typing"], r["rt"])
    r["beyond_clock"] = beyond_clock_test(d, r["typing"], n_perm, seed)
    r["original"] = original_test(d, r["typing"], n_perm, seed)
    r["auc"], r["tired_prob"] = classify(d, r["typing"])
    r["auc_clock"], _ = classify(d, CLOCK)
    return r


def latest_predictions(f: pd.DataFrame) -> pd.DataFrame:
    """Each person's latest check-in with predicted KSS, from models trained on all panel data."""
    d = scored(f)
    typing = cols = present(d, TYPING)  # no hours awake: the demo can't ask for it
    main = clf = None
    if not missing_data(d) and typing:
        main = ridge().fit(d[cols].to_numpy(float), d.kss_delta.to_numpy(float))
        tired = tired_labels(d)
        if len(np.unique(tired)) == 2:
            clf = logistic().fit(d[cols].to_numpy(float), tired)
    fallback, fallback_cols = no_baseline_model(f)
    rows = []
    for person, g in f.sort_values("timestamp").groupby("person"):
        last = g.iloc[[-1]]
        pred, prob, source = np.nan, np.nan, "not enough data yet"
        if last.participant_type.iloc[0] == "panel" and main is not None and pd.notna(last.kss_base.iloc[0]):
            X = last[cols].to_numpy(float)
            pred, source = main.predict(X)[0] + last.kss_base.iloc[0], "panel model"
            if clf is not None:
                prob = clf.predict_proba(X)[0, 1]
        elif fallback is not None:
            pred, source = fallback.predict(last[fallback_cols].to_numpy(float))[0], "no-baseline model"
        rows.append({"person": person, "participant_type": last.participant_type.iloc[0],
                     "timestamp": last.timestamp.iloc[0], "kss_reported": last.kss.iloc[0],
                     "tiredness_pred": float(np.clip(pred, 1, 9)) if np.isfinite(pred) else np.nan,
                     "tired_prob": float(prob), "model": source})
    return pd.DataFrame(rows)
