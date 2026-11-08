"""Shared harness for the overnight modelling experiments (8 Nov). Read this before adding an experiment.

Data: the 36 check-ins collected by 00:53 on 8 Nov: 33 from 5 panel people (24 scored once each
person's first two baseline check-ins are set aside) and 3 one-off walk-ups. Morning check-ins are the
confirmation set for whatever gets frozen, so nothing here may read check-ins after LAST_SEEN.

Ground rules, because 24 scored check-ins from 5 people is tiny:
- Compare every idea with the same baselines (guess, clock, v1, v3) under both evaluations.
- Count every variant you try and log it (n_variants). The best of many is optimistic.
- Tune hyperparameters inside the training folds only, never on the test person or time.
- Before calling anything a win, run perm_test() on the final procedure.
- Never put names in outputs: use pid (P1..P5 for the panel, W1..W3 for walk-ups).
- Predictions may use typing (and the reaction test). The clock and check-in number are benchmarks or
  nuisance covariates only, never inputs, because the demo can't know them.

Usage:
    from harness import load_table, scored_panel, evaluate, compare, perm_test, log_result, BASELINES
    t = load_table(); d = scored_panel(t)
    print(compare(d, {"my idea": my_fit_predict}))
A fit_predict(train, test) gets two DataFrames of scored panel rows and returns absolute KSS
predictions for test. Use anchored() / plain() to wrap an sklearn-style model factory.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ANALYSIS = Path("/Users/aryan/Projects/hacktheburgh/keyboard-fatigue-nobaseline/analysis")
sys.path.insert(0, str(ANALYSIS))
EXP = ANALYSIS / "results" / "experiments"
DATA = Path("/Users/aryan/Projects/hacktheburgh/keyboard-fatigue/data/real-2026-11-08-0110/train_data")
FEATURES = ANALYSIS / "results" / "real-2026-11-08-0110"
LAST_SEEN = pd.Timestamp("2026-11-08 00:53:51")  # the morning batch is the confirmation set
MIN_TRAIN_ROLLING = 10

from fair_test import SWELL_STRESS, add_her_numbers  # noqa: E402
from model import CLOCK, TYPING, ridge  # noqa: E402
from model import scored as _scored  # noqa: E402


def raw_sessions() -> dict:
    """The raw session JSON for every check-in up to LAST_SEEN, keyed by checkin_id (= session_id)."""
    out = {}
    for path in sorted(DATA.glob("*.json")):
        if path.name.startswith("._"):
            continue
        s = json.loads(path.read_text())
        out[s["session_id"]] = s
    return out


def load_table() -> pd.DataFrame:
    """One row per check-in: processed features, raw features (suffix-free where not duplicated),
    the extra self-ratings, Ananya's two numbers, and an anonymous pid."""
    f = pd.read_csv(FEATURES / "features_processed.csv")
    raw = pd.read_csv(FEATURES / "features_raw.csv")
    t = f.merge(raw[["checkin_id"] + [c for c in raw.columns if c not in f.columns]], on="checkin_id")
    t["is_baseline"] = t.is_baseline.astype(bool)
    t = add_her_numbers(t, raw)
    t["timestamp"] = pd.to_datetime(t.timestamp)
    sessions = raw_sessions()
    for key in ("mental_fatigue", "motivation", "sleep_last_night_h", "caffeine_last3h"):
        t[key] = t.checkin_id.map(lambda c: (sessions.get(c, {}).get("self_report") or {}).get(key))
    panel = sorted(t.loc[t.participant_type.eq("panel"), "person"].unique())
    walkups = sorted(t.loc[~t.participant_type.eq("panel"), "person"].unique())
    codes = {p: f"P{i + 1}" for i, p in enumerate(panel)} | {p: f"W{i + 1}" for i, p in enumerate(walkups)}
    t["pid"] = t.person.map(codes)
    t = t[t.timestamp <= LAST_SEEN]
    return t.sort_values("timestamp").reset_index(drop=True)


def scored_panel(t: pd.DataFrame) -> pd.DataFrame:
    """The panel check-ins every evaluation predicts: rated, not a baseline check-in, with typing."""
    return _scored(t).sort_values("timestamp").reset_index(drop=True)


# ---- model wrappers -------------------------------------------------------------------------

def anchored(model_factory, cols):
    """Fit the change from the person's baseline rating; predict absolute KSS."""
    def fit_predict(train, test):
        m = model_factory().fit(train[cols].to_numpy(float), train.kss_delta.to_numpy(float))
        return m.predict(test[cols].to_numpy(float)) + test.kss_base.to_numpy(float)
    return fit_predict


def plain(model_factory, cols):
    """Fit absolute KSS directly."""
    def fit_predict(train, test):
        m = model_factory().fit(train[cols].to_numpy(float), train.kss.to_numpy(float))
        return m.predict(test[cols].to_numpy(float))
    return fit_predict


def guess(train, test):
    return np.full(len(test), train.kss.mean())


def swell_stress(train, frame):
    """Ananya's SWELL stress recipe on her two numbers, scaled by the training rows' spread."""
    spread = {c: train[f"h_{c}_dev"].std() for c in SWELL_STRESS}
    return sum(w * frame[f"h_{c}_dev"] / spread[c] for c, w in SWELL_STRESS.items()).to_numpy(float)


def v3(train, test):
    m = ridge().fit(swell_stress(train, train).reshape(-1, 1), train.kss_delta.to_numpy(float))
    return m.predict(swell_stress(train, test).reshape(-1, 1)) + test.kss_base.to_numpy(float)


BASELINES = {"guess": guess, "clock": anchored(ridge, CLOCK), "v1": anchored(ridge, TYPING), "v3": v3}


# ---- evaluation -----------------------------------------------------------------------------

def splits(d: pd.DataFrame, scheme: str):
    """lopo: each person predicted by a model trained on the others.
    rolling: each check-in predicted by a model trained on every scored check-in before it
    (all people, including that person's own earlier ones), once there are MIN_TRAIN_ROLLING."""
    if scheme == "lopo":
        for p in d.pid.unique():
            yield (d.pid != p).to_numpy(), (d.pid == p).to_numpy()
    elif scheme == "rolling":
        times = d.timestamp.to_numpy()
        for i in range(len(d)):
            train = times < times[i]
            if train.sum() >= MIN_TRAIN_ROLLING:
                test = np.zeros(len(d), bool)
                test[i] = True
                yield train, test
    else:
        raise ValueError(scheme)


def predict(d: pd.DataFrame, fit_predict, scheme: str) -> tuple[np.ndarray, np.ndarray]:
    pred, base = np.full(len(d), np.nan), np.full(len(d), np.nan)
    for train, test in splits(d, scheme):
        pred[test] = np.clip(fit_predict(d[train], d[test]), 1, 9)
        base[test] = d.kss[train].mean()
    return pred, base


def metrics(d: pd.DataFrame, pred: np.ndarray, base: np.ndarray) -> dict:
    """MAE; skill = 1 - squared error / squared error of guessing the training mean; Spearman;
    within-person Spearman (median over people with 4+ predictions); within-person concordance =
    share of same-person pairs with different KSS that the predictions put in the right order
    (ties count half): the number a nap rule depends on, 0.5 = coin flip."""
    m = ~np.isnan(pred)
    y, p, g = d.kss.to_numpy(float)[m], pred[m], base[m]
    out = {"n": int(m.sum()), "MAE": float(np.mean(np.abs(p - y))),
           "skill": float(1 - ((p - y) ** 2).sum() / ((g - y) ** 2).sum()),
           "spearman": float(stats.spearmanr(p, y).statistic) if np.ptp(p) > 0 else float("nan")}
    sub = d[m].assign(pred=p)
    rhos, good, pairs = [], 0.0, 0
    for _, grp in sub.groupby("pid"):
        k, q = grp.kss.to_numpy(float), grp.pred.to_numpy(float)
        if len(grp) >= 4 and np.ptp(k) > 0 and np.ptp(q) > 0:
            rhos.append(stats.spearmanr(q, k).statistic)
        for i in range(len(k)):
            for j in range(i + 1, len(k)):
                if k[i] != k[j]:
                    pairs += 1
                    s = np.sign(q[i] - q[j]) * np.sign(k[i] - k[j])
                    good += 1.0 if s > 0 else 0.5 if s == 0 else 0.0
    out["within_rho"] = float(np.median(rhos)) if rhos else float("nan")
    out["within_concordance"] = good / pairs if pairs else float("nan")
    return out


def evaluate(d: pd.DataFrame, fit_predict, schemes=("lopo", "rolling")) -> dict:
    return {s: metrics(d, *predict(d, fit_predict, s)) for s in schemes}


def compare(d: pd.DataFrame, candidates: dict, schemes=("lopo", "rolling"), baselines=True) -> pd.DataFrame:
    """A table of every model (baselines first) under every scheme."""
    models = (BASELINES | candidates) if baselines else candidates
    rows = {}
    for name, fn in models.items():
        res = evaluate(d, fn, schemes)
        rows[name] = {f"{s}:{k}": v for s, r in res.items() for k, v in r.items() if k != "n"}
    return pd.DataFrame(rows).T.round(3)


def perm_test(d: pd.DataFrame, fit_predict, scheme="lopo", stat="skill", n=200, seed=0):
    """Shuffle KSS among each person's scored check-ins and re-run the whole procedure.
    Returns (observed, p-value, null values). A small p means the features carry information about
    which of a person's check-ins were more tired. It doesn't rule out practice: anything that drifts
    over the night passes this test too."""
    observed = evaluate(d, fit_predict, (scheme,))[scheme][stat]
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n):
        dd = d.copy()
        dd["kss"] = dd.groupby("pid").kss.transform(lambda s: rng.permutation(s.to_numpy()))
        dd["kss_delta"] = dd.kss - dd.kss_base
        null.append(evaluate(dd, fit_predict, (scheme,))[scheme][stat])
    null = np.asarray(null, float)
    return observed, float((np.sum(null >= observed) + 1) / (len(null) + 1)), null


def log_result(track: str, name: str, n_variants: int, results: dict, notes: str = "") -> None:
    """Append one experiment to results.jsonl. results is evaluate()'s output (or similar)."""
    record = {"time": datetime.now().isoformat(timespec="seconds"), "track": track, "name": name,
              "n_variants": n_variants, "results": results, "notes": notes}
    with open(EXP / "results.jsonl", "a") as fh:
        fh.write(json.dumps(record, default=float) + "\n")


if __name__ == "__main__":
    t = load_table()
    d = scored_panel(t)
    print(f"{len(t)} check-ins, {len(d)} scored from {d.pid.nunique()} panel people "
          f"({', '.join(f'{p}: {n}' for p, n in d.pid.value_counts().sort_index().items())})")
    print(compare(d, {}).to_string())
