"""Shared pieces for the first-guess (stranger) track.

Stranger evaluation: every panel check-in, baseline check-ins included (a stranger has no baseline),
is predicted from single-attempt features only (no _dev features, no kss_base) by a model trained on
the other panel people (leave one person out). The rule-type references behind the _adj features
are recomputed inside each fold from the training people only, so the left-out person never shapes
them (the shipped references use every panel check-in).

Never carries names: load() drops the name column straight after harness.load_table().
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from harness import load_table, log_result  # noqa: E402,F401
from fair_test import SWELL_STRESS  # noqa: E402  (harness puts analysis/ on sys.path)

TRACK = "first_guess"
RATIO_COLS = ("vt_rule_cost", "vt_accuracy")  # divided by the rule's typical value, as features.py does
SUB_COLS = ("vt_gap_cv", "vt_pause_rel_share", "backspace_per100", "log_wpm", "sqrt_bsp", "vt_hold_med",
            "vt_pause1s_share")  # typical value for the rule subtracted, overall typical value added back
FG5 = ["vt_rule_cost_adj", "vt_gap_cv_adj", "vt_pause_rel_share_adj", "backspace_per100_adj", "vt_accuracy_adj"]
RATIO3 = ["vt_rule_cost_adj", "vt_pause_rel_share_adj", "vt_accuracy_adj"]
WIDE = FG5 + ["log_wpm_adj", "vt_hold_med_adj", "vt_pause1s_share_adj", "log_rt"]
# +1 = higher means more tired. The first five are analysis/first_guess.py's DIRECTIONS (fixed before
# real data); the last four are textbook directions (slower typing, longer key holds, more long
# pauses, slower reactions), written down before any run in this folder.
PRIOR = {"vt_rule_cost_adj": 1, "vt_gap_cv_adj": 1, "vt_pause_rel_share_adj": 1, "backspace_per100_adj": 1,
         "vt_accuracy_adj": -1, "log_wpm_adj": -1, "vt_hold_med_adj": 1, "vt_pause1s_share_adj": 1, "log_rt": 1}
Z_LIMIT = 4
KEEP = ["pid", "participant_type", "timestamp", "checkin_n", "is_baseline", "question_type", "kss",
        *RATIO_COLS, *SUB_COLS, "log_rt", "vt_wpm"]


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(panel, walk-ups): rated check-ins up to the harness cut-off, raw single-attempt columns, pid only."""
    t = load_table().drop(columns=["person"])
    t["log_wpm"] = np.log(t.vt_wpm)
    t["sqrt_bsp"] = np.sqrt(t.vt_n_backspace / t.vt_n_keys)
    t["log_rt"] = np.log(t.rt_median)
    t = t[t.kss.notna()]
    panel = t[t.participant_type.eq("panel")][KEEP].reset_index(drop=True)
    walk = t[~t.participant_type.eq("panel")][KEEP].reset_index(drop=True)
    return panel, walk


def rule_adjust(ref: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """Add _adj columns to frame, measured against ref's median for each rule type (and overall)."""
    out = frame.copy()
    for col in RATIO_COLS + SUB_COLS:
        overall = ref[col].median()
        typical = ref.groupby("question_type")[col].median()
        reference = frame.question_type.map(typical).astype(float).fillna(overall)
        if col in RATIO_COLS:
            out[col + "_adj"] = frame[col] / reference.where(reference > 0)
        else:
            out[col + "_adj"] = frame[col] - reference + overall
    return out


def make_folds(panel: pd.DataFrame, full_refs: bool = False) -> list:
    """[(test mask, train frame, test frame)] for leave-one-person-out. full_refs=True mimics the
    shipped features (rule references from every panel check-in, the left-out person included)."""
    folds = []
    for p in sorted(panel.pid.unique()):
        te = panel.pid.eq(p).to_numpy()
        ref = panel if full_refs else panel[~te]
        folds.append((te, rule_adjust(ref, panel[~te]).reset_index(drop=True),
                      rule_adjust(ref, panel[te]).reset_index(drop=True)))
    return folds


# ---- building blocks for candidates -----------------------------------------------------------

def robust_scale(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        centre = np.nanmedian(X, axis=0)
        mad = 1.4826 * np.nanmedian(np.abs(X - centre), axis=0)
        sd = np.nanstd(X, axis=0)
    scale = np.where(mad > 1e-9, mad, np.where(sd > 1e-9, sd, 1.0))
    return np.nan_to_num(centre), scale


def zscores(Xtr, Xte):
    c, s = robust_scale(Xtr)
    return np.clip((Xtr - c) / s, -Z_LIMIT, Z_LIMIT), np.clip((Xte - c) / s, -Z_LIMIT, Z_LIMIT)


def ranks(Xtr, Xte):
    """Each value's mid-rank percentile among the training values of that column, centred on 0."""
    def pct(ref, X):
        out = np.full(X.shape, np.nan)
        for j in range(X.shape[1]):
            r = ref[:, j][~np.isnan(ref[:, j])]
            if len(r) == 0:
                continue
            x = X[:, j]
            ok = ~np.isnan(x)
            out[ok, j] = ((r[None, :] < x[ok, None]).sum(1) + 0.5 * (r[None, :] == x[ok, None]).sum(1)) / len(r) - 0.5
        return out
    return pct(Xtr, Xtr), pct(Xtr, Xte)


def signed_mean(Z: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """Average of signed scores; a sign of 0 leaves the column out; all-missing rows give NaN."""
    Z = Z * signs
    Z[:, signs == 0] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(Z, axis=1)


def map_slope(itr, y, ite, mode="nonneg"):
    """KSS = intercept + slope * index, fitted on training rows; 'nonneg' falls back to the mean when
    the slope isn't positive (as analysis/first_guess.py does). Missing index = typical attempt (0)."""
    ok = ~np.isnan(itr)
    slope, icpt = 0.0, float(np.mean(y))
    if ok.sum() >= 3 and np.ptp(itr[ok]) > 0:
        s, i = np.polyfit(itr[ok], y[ok], 1)
        if mode == "free" or s > 0:
            slope, icpt = float(s), float(i)
        else:
            icpt = float(y[ok].mean())
    return icpt + slope * np.nan_to_num(ite)


def matched_deviations(v: np.ndarray, n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """v minus its mean over training rows with the same check-in number (rows whose check-in number
    appears once are dropped). Check-in number is a nuisance covariate here, never an input."""
    dev = np.full(len(v), np.nan)
    for k in np.unique(n):
        m = (n == k) & ~np.isnan(v)
        if m.sum() >= 2:
            dev[m] = v[m] - v[m].mean()
    return dev, ~np.isnan(dev)


def map_matched(itr, y, ite, n):
    """Slope (>= 0) from between-person contrasts at the same check-in number; intercept keeps the
    training mean."""
    di, oki = matched_deviations(itr, n)
    dy, _ = matched_deviations(np.where(np.isnan(itr), np.nan, y.astype(float)), n)
    ok = oki & ~np.isnan(dy)
    slope = float(np.sum(di[ok] * dy[ok]) / np.sum(di[ok] ** 2)) if ok.sum() >= 3 and np.sum(di[ok] ** 2) > 0 else 0.0
    slope = max(slope, 0.0)
    mean_i = float(np.nanmean(itr)) if np.any(~np.isnan(itr)) else 0.0
    return float(np.mean(y)) + slope * (np.where(np.isnan(ite), mean_i, ite) - mean_i)


def map_quantile(itr, y, ite):
    """The test index's percentile among the training index values, turned into that quantile of
    the training ratings (the 'percentile against the team' read as a KSS)."""
    r = itr[~np.isnan(itr)]
    if len(r) == 0:
        return np.full(len(ite), float(np.mean(y)))
    pct = np.array([0.5 if np.isnan(v) else (np.mean(r < v) + 0.5 * np.mean(r == v)) for v in ite])
    return np.quantile(np.asarray(y, float), pct)


def spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~np.isnan(a) & ~np.isnan(b)
    if ok.sum() < 3 or np.ptp(a[ok]) == 0 or np.ptp(b[ok]) == 0:
        return float("nan")
    return float(stats.spearmanr(a[ok], b[ok]).statistic)


# ---- evaluation ----------------------------------------------------------------------------------

def run(fn, folds, y: np.ndarray, n_rows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(prediction, index or NaN, flat guess) for every panel row, each from its own fold."""
    pred, idx, flat = np.full(n_rows, np.nan), np.full(n_rows, np.nan), np.full(n_rows, np.nan)
    for te, train, test in folds:
        p, i = fn(train, y[~te], test)
        pred[te] = np.clip(p, 1, 9)
        if i is not None:
            idx[te] = i
        flat[te] = y[~te].mean()
    return pred, idx, flat


def pair_masks(panel: pd.DataFrame) -> dict:
    pid, n = panel.pid.to_numpy(), panel.checkin_n.to_numpy()
    same = pid[:, None] == pid[None, :]
    upper = np.triu(np.ones((len(pid), len(pid)), bool), 1)
    return {"between": upper & ~same, "matched": upper & ~same & (n[:, None] == n[None, :]), "within": upper & same}


def concordance(score: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    """Share of pairs (in mask) with different KSS that score puts in the right order; ties count half."""
    ds = np.sign(score[:, None] - score[None, :])
    dy = np.sign(y[:, None] - y[None, :])
    m = mask & (dy != 0) & ~np.isnan(ds)
    if not m.any():
        return float("nan")
    s = (ds * dy)[m]
    return float(np.mean(np.where(s > 0, 1.0, np.where(s == 0, 0.5, 0.0))))


def metrics(panel: pd.DataFrame, y: np.ndarray, pred: np.ndarray, idx: np.ndarray, flat: np.ndarray,
            masks: dict) -> dict:
    first = panel.checkin_n.eq(1).to_numpy()
    out = {"MAE": float(np.mean(np.abs(pred - y))), "MAE_flat": float(np.mean(np.abs(flat - y))),
           "skill": float(1 - np.sum((pred - y) ** 2) / np.sum((flat - y) ** 2)),
           "spearman": spearman(pred, y),
           "conc_between": concordance(pred, y, masks["between"]),
           "conc_matched": concordance(pred, y, masks["matched"]),
           "conc_within": concordance(pred, y, masks["within"]),
           "first_MAE": float(np.mean(np.abs(pred[first] - y[first]))),
           "first_MAE_flat": float(np.mean(np.abs(flat[first] - y[first]))),
           "first_bias": float(np.mean(pred[first] - y[first])),
           "rho_pred_checkin": spearman(pred, panel.checkin_n.to_numpy(float))}
    if np.any(~np.isnan(idx)):
        out |= {"idx_spearman": spearman(idx, y), "idx_conc_between": concordance(idx, y, masks["between"]),
                "idx_conc_matched": concordance(idx, y, masks["matched"]),
                "idx_conc_within": concordance(idx, y, masks["within"])}
    return out


def walkup_check(fn, panel: pd.DataFrame, walk: pd.DataFrame) -> dict:
    """Train on every panel check-in, predict the walk-ups (3 rows: a sanity check, not a test)."""
    train, test = rule_adjust(panel, panel), rule_adjust(panel, walk)
    y = panel.kss.to_numpy(float)
    p, _ = fn(train, y, test)
    p = np.clip(p, 1, 9)
    k = walk.kss.to_numpy(float)
    return {"walk_pred": [round(float(v), 2) for v in p], "walk_kss": [float(v) for v in k],
            "walk_MAE": float(np.mean(np.abs(p - k))), "walk_MAE_flat": float(np.mean(np.abs(y.mean() - k)))}
