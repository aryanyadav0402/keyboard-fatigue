"""Every first-guess variant this track tries, registered before any of them was run.

Honesty labels:
- prior: directions/weights fixed before this data (first_guess.py's DIRECTIONS, textbook
  directions, the SWELL stress weights from office-worker data, fair_test.SWELL_STRESS); only a slope/intercept is fitted.
- fold: anything data-driven (directions, feature drops, alpha, k) is decided inside the training
  folds only.
- evidence: a direction chosen because of correlations already computed on all of tonight's data
  (the backspace and rhythm flips). Their scores are optimistic; reported, not eligible to be chosen.
- check: the shipped spec with rule references from every panel check-in, to show what the in-fold
  references change. Not eligible.

Selection rule (fixed before running): the eligible (prior or fold) candidate with the highest pooled
Spearman between LOPO predictions and KSS; ties broken by skill. That candidate gets the shuffle test.
"""

import numpy as np

from common import (FG5, PRIOR, RATIO3, SWELL_STRESS, WIDE, map_matched, map_quantile, map_slope,
                    matched_deviations, ranks, robust_scale, signed_mean, spearman, zscores)


# ---- direction rules -------------------------------------------------------------------------

def prior_signs(cols, override=None):
    s = {c: PRIOR[c] for c in cols} | (override or {})
    return lambda train, y, cols_: np.array([s[c] for c in cols_], float)


def _sign(rho):
    return 0.0 if np.isnan(rho) else float(np.sign(rho))


def sign_pooled(train, y, cols):
    return np.array([_sign(spearman(train[c].to_numpy(float), y)) for c in cols])


def sign_within(train, y, cols):
    pid = train.pid.to_numpy()
    multi = np.isin(pid, [p for p in np.unique(pid) if (pid == p).sum() >= 2])
    out = []
    for c in cols:
        x = train[c].to_numpy(float)
        xc, yc = np.full(len(x), np.nan), np.full(len(x), np.nan)
        for p in np.unique(pid[multi]):
            m = pid == p
            xc[m], yc[m] = x[m] - np.nanmean(x[m]), y[m] - y[m].mean()
        out.append(_sign(spearman(xc, yc)))
    return np.array(out)


def sign_between(train, y, cols):
    g = train.assign(_y=y).groupby("pid")
    my = g._y.mean().to_numpy()
    return np.array([_sign(spearman(g[c].mean().to_numpy(float), my)) for c in cols])


def sign_matched(train, y, cols):
    n = train.checkin_n.to_numpy()
    dy, _ = matched_deviations(y.astype(float), n)
    return np.array([_sign(spearman(matched_deviations(train[c].to_numpy(float), n)[0], dy)) for c in cols])


def drop_contradicted(train, y, cols):
    prior = np.array([PRIOR[c] for c in cols], float)
    return np.where(sign_pooled(train, y, cols) == prior, prior, 0.0)


# ---- model families --------------------------------------------------------------------------

def index_model(cols, signs, scoring="z", mapping="nonneg"):
    def fn(train, y, test):
        s = signs(train, y, cols)
        if not np.any(s):
            return np.full(len(test), y.mean()), np.full(len(test), np.nan)
        score = zscores if scoring == "z" else ranks
        Ztr, Zte = score(train[cols].to_numpy(float), test[cols].to_numpy(float))
        itr, ite = signed_mean(Ztr, s), signed_mean(Zte, s)
        if mapping == "quantile":
            return map_quantile(itr, y, ite), ite
        if mapping == "matched":
            return map_matched(itr, y, ite, train.checkin_n.to_numpy()), ite
        return map_slope(itr, y, ite, mapping), ite
    return fn


def swell_model(adjusted=False, mapping="nonneg"):
    """The SWELL stress weights (fair_test.SWELL_STRESS) on uncentred log speed and sqrt(backspaces per key), each in
    units of its typical within-person spread among the training people."""
    cols = {"log_wpm": SWELL_STRESS["rate"], "sqrt_bsp": SWELL_STRESS["errr"]}

    def fn(train, y, test):
        pid = train.pid.to_numpy()
        s_tr, s_te = np.zeros(len(train)), np.zeros(len(test))
        for c, w in cols.items():
            c = c + "_adj" if adjusted else c
            x, xt = train[c].to_numpy(float), test[c].to_numpy(float)
            centred = np.concatenate([x[pid == p] - x[pid == p].mean() for p in np.unique(pid) if (pid == p).sum() >= 2])
            sd = centred.std(ddof=1)
            med = np.median(x)
            s_tr += w * (x - med) / sd
            s_te += w * (xt - med) / sd
        return map_slope(s_tr, y, s_te, mapping), s_te
    return fn


def _fill(Xtr, Xte):
    med = np.nanmedian(Xtr, axis=0)
    return np.where(np.isnan(Xtr), med, Xtr), np.where(np.isnan(Xte), med, Xte)


def ridge_predict(Xtr, y, Xte, alpha):
    Xtr, Xte = _fill(Xtr, Xte)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    w = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ (y - y.mean()))
    return y.mean() + B @ w


def knn_predict(Xtr, y, Xte, k):
    c, s = robust_scale(Xtr)
    A = np.nan_to_num(np.clip((Xtr - c) / s, -4, 4))
    B = np.nan_to_num(np.clip((Xte - c) / s, -4, 4))
    d = ((B[:, None, :] - A[None, :, :]) ** 2).sum(-1)
    near = np.argsort(d, axis=1, kind="stable")[:, :min(k, len(y))]
    return y[near].mean(1)


def inner_lopo(predict, grid):
    """Pick the grid value with the lowest squared error leaving one training person out at a time."""
    def fn_factory(cols):
        def fn(train, y, test):
            X, Xt, pid = train[cols].to_numpy(float), test[cols].to_numpy(float), train.pid.to_numpy()
            best, best_err = grid[0], np.inf
            for g in grid:
                err = 0.0
                for p in np.unique(pid):
                    m = pid == p
                    err += np.sum((predict(X[~m], y[~m], X[m], g) - y[m]) ** 2)
                if err < best_err:
                    best, best_err = g, err
            return predict(X, y, Xt, best), None
        return fn
    return fn_factory


ridge_model = inner_lopo(ridge_predict, list(np.logspace(-2, 3, 11)))
knn_model = inner_lopo(knn_predict, [1, 3, 5, 7, 11, 15, 21])


def flat(train, y, test):
    return np.full(len(test), y.mean()), None


FG4 = [c for c in FG5 if c != "backspace_per100_adj"]
CANDIDATES = {
    # name: (fit_predict, honesty, description)
    "fg5_prior": (index_model(FG5, prior_signs(FG5)), "prior", "shipped spec: 5 features, fixed directions, slope >= 0"),
    "fg5_flip_bsp": (index_model(FG5, prior_signs(FG5, {"backspace_per100_adj": -1})), "evidence", "backspaces -1"),
    "fg5_flip_bsp_cv": (index_model(FG5, prior_signs(FG5, {"backspace_per100_adj": -1, "vt_gap_cv_adj": -1})),
                        "evidence", "backspaces -1, rhythm unevenness -1"),
    "fg4_no_bsp": (index_model(FG4, prior_signs(FG4)), "evidence", "drop backspaces"),
    "fg5_drop_contra": (index_model(FG5, drop_contradicted), "fold", "drop features whose training-fold Spearman contradicts the prior direction"),
    "fg5_sign_pooled": (index_model(FG5, sign_pooled), "fold", "directions = sign of pooled training Spearman"),
    "fg5_sign_within": (index_model(FG5, sign_within), "fold", "directions = sign of within-person training Spearman"),
    "fg5_sign_between": (index_model(FG5, sign_between), "fold", "directions = sign of Spearman over training people's means"),
    "fg5_sign_matched": (index_model(FG5, sign_matched, mapping="matched"), "fold",
                         "directions and slope from between-person contrasts at the same check-in number"),
    "fg5_prior_free_slope": (index_model(FG5, prior_signs(FG5), mapping="free"), "fold", "fixed directions, slope may be negative"),
    "fg5_prior_matched_slope": (index_model(FG5, prior_signs(FG5), mapping="matched"), "fold",
                                "fixed directions, slope (>= 0) from same-check-in-number contrasts"),
    "fg5_rank_prior": (index_model(FG5, prior_signs(FG5), scoring="rank"), "prior", "rank (percentile) scores instead of z"),
    "fg5_rank_flip_bsp": (index_model(FG5, prior_signs(FG5, {"backspace_per100_adj": -1}), scoring="rank"), "evidence",
                          "rank scores, backspaces -1"),
    "fg5_rank_sign_pooled": (index_model(FG5, sign_pooled, scoring="rank"), "fold", "rank scores, pooled training directions"),
    "fg5_prior_quantile": (index_model(FG5, prior_signs(FG5), mapping="quantile"), "prior",
                           "percentile of the index among training check-ins -> that quantile of training KSS"),
    "fg5_flip_bsp_quantile": (index_model(FG5, prior_signs(FG5, {"backspace_per100_adj": -1}), mapping="quantile"),
                              "evidence", "quantile mapping, backspaces -1"),
    "ratio3_prior": (index_model(RATIO3, prior_signs(RATIO3)), "prior", "rule cost, relative pauses, accuracy: fixed directions"),
    "ratio3_sign_pooled": (index_model(RATIO3, sign_pooled), "fold", "the three ratios, pooled training directions"),
    "swell_raw": (swell_model(False), "prior", "SWELL stress weights on raw log speed and sqrt backspaces, slope >= 0"),
    "swell_raw_free": (swell_model(False, "free"), "fold", "same, slope may be negative (like v3)"),
    "swell_adj": (swell_model(True), "prior", "SWELL stress weights on rule-adjusted speed and backspaces, slope >= 0"),
    "rt_prior": (index_model(["log_rt"], prior_signs(["log_rt"])), "prior", "reaction time alone, slower = more tired"),
    "fg5_rt_prior": (index_model(FG5 + ["log_rt"], prior_signs(FG5 + ["log_rt"])), "prior", "5 features + reaction time"),
    "ridge_fg5": (ridge_model(FG5), "fold", "ridge on the 5 features, alpha by inner LOPO"),
    "ridge_wide": (ridge_model(WIDE), "fold", "ridge on 9 single-attempt features, alpha by inner LOPO"),
    "knn_fg5": (knn_model(FG5), "fold", "k nearest panel check-ins on the 5 features, k by inner LOPO"),
    "knn_wide": (knn_model(WIDE), "fold", "k nearest panel check-ins on 9 features, k by inner LOPO"),
    "fg5_prior_fullrefs": (index_model(FG5, prior_signs(FG5)), "check", "shipped spec with rule references from all panel rows"),
}
N_VARIANTS = len(CANDIDATES)  # the flat guess is the reference, not a variant
ELIGIBLE = [n for n, (_, h, _) in CANDIDATES.items() if h in ("prior", "fold")]
