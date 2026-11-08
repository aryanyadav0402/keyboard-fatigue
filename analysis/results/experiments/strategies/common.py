"""Training-strategy variants for the tiny-data track (8 Nov). Uses the shared harness unchanged.

Every builder returns a fit_predict(train, test) -> absolute KSS, as the harness expects. Nothing here
reads test.kss / test.kss_delta / test.mental_fatigue; test rows only supply typing features, the
person's baseline rating (kss_base, as v1 uses) and, for the one labelled oracle, the check-in number.
Hyperparameters are chosen by leave-one-training-person-out CV inside the training rows (inner_splits).
"""

import sys
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

sys.path.insert(0, "/Users/aryan/Projects/hacktheburgh/keyboard-fatigue-nobaseline/analysis/results/experiments")
import harness as H  # noqa: E402
from model import ALPHAS, TYPING, ridge  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.cross_decomposition import PLSRegression  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import (BayesianRidge, ElasticNet, Lasso, LogisticRegression,  # noqa: E402
                                  Ridge, RidgeCV)
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

warnings.filterwarnings("ignore")

BASE_ROWS = None  # panel baseline check-ins, set by load(); used by the augmentation / practice variants


# ---- data -----------------------------------------------------------------------------------

def load():
    """(table, scored panel rows, baseline rows). Names are dropped straight away; only pid is kept."""
    global BASE_ROWS
    t = H.load_table().drop(columns=["person"])
    for c in ("mental_fatigue", "motivation", "sleep_last_night_h", "caffeine_last3h"):
        t[c] = pd.to_numeric(t[c], errors="coerce")
    t["log_n"] = np.log(t.checkin_n.astype(float))
    t["kss_delta"] = t.kss - t.kss_base
    base = t[t.participant_type.eq("panel") & t.is_baseline]
    mf_base = base.groupby("pid").mental_fatigue.mean()
    t["mf_base"] = t.pid.map(mf_base)
    t["mf_delta"] = t.mental_fatigue - t.mf_base
    d = H.scored_panel(t)
    assert np.allclose(d.kss_delta, d.kss - d.kss_base)
    n_base = base.groupby("pid").size()
    B = t[t.participant_type.eq("panel") & t.is_baseline & t.pid.isin(n_base[n_base >= 2].index)].copy()
    BASE_ROWS = B.reset_index(drop=True)
    return t, d, BASE_ROWS


def aug_rows(train, test):
    """Baseline check-ins of people already in the training rows and recorded before the test check-in(s)."""
    B = BASE_ROWS
    keep = B.pid.isin(train.pid.unique()) & (B.timestamp < test.timestamp.min())
    return B[keep]


# ---- small utilities ------------------------------------------------------------------------

def inner_splits(train):
    """Leave one training person out (the tuning CV used everywhere in this track)."""
    pids = train.pid.to_numpy()
    u = np.unique(pids)
    if len(u) < 2:
        return []
    return [(pids != p, pids == p) for p in u]


def group_mean(a, groups):
    a = np.asarray(a, float)
    out = np.empty_like(a)
    for g in np.unique(groups):
        m = groups == g
        out[m] = np.nanmean(a[m], axis=0)
    return out


def prep(train, cols):
    """Median-impute and standardize on the training rows; returns a transform for any frame."""
    Xtr = train[cols].to_numpy(float)
    imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(Xtr)
    sc = StandardScaler().fit(imp.transform(Xtr))
    return lambda fr: sc.transform(imp.transform(fr[cols].to_numpy(float)))


def concordance(frame, pred):
    """Within-person concordance, as in harness.metrics."""
    good, pairs = 0.0, 0
    k_all, p_all, g_all = frame.kss.to_numpy(float), np.asarray(pred, float), frame.pid.to_numpy()
    for g in np.unique(g_all):
        m = g_all == g
        k, q = k_all[m], p_all[m]
        for i in range(len(k)):
            for j in range(i + 1, len(k)):
                if k[i] != k[j]:
                    pairs += 1
                    s = np.sign(q[i] - q[j]) * np.sign(k[i] - k[j])
                    good += 1.0 if s > 0 else 0.5 if s == 0 else 0.0
    return good / pairs if pairs else np.nan


def tuned(family, grid, criterion="mse"):
    """Pick a hyperparameter by inner leave-one-person-out on the training rows, then refit.
    Ties go to the earlier grid value. Records the choices in fp.chosen."""
    def fp(train, test):
        splits = inner_splits(train)
        y = train.kss.to_numpy(float)
        best_g, best_s = grid[0], np.inf
        if splits:
            for g in grid:
                pred = np.full(len(train), np.nan)
                for tr, te in splits:
                    pred[te] = family(g)(train[tr], train[te])
                pred = np.clip(pred, 1, 9)
                s = np.mean((pred - y) ** 2) if criterion == "mse" else -concordance(train, pred)
                if s < best_s - 1e-12:
                    best_g, best_s = g, s
        fp.chosen.append(best_g)
        return family(best_g)(train, test)
    fp.chosen = []
    return fp


def mean_of(*fps):
    return lambda train, test: np.mean([np.clip(f(train, test), 1, 9) for f in fps], axis=0)


# ---- linear models with any amount of anchoring ---------------------------------------------

def lin(cols=TYPING, a=1.0, factory=ridge, aug=False):
    """pred = a*kss_base + f(x), with f fitted on kss - a*kss_base (a=1: v1, anchored; a=0: unanchored)."""
    def fp(train, test):
        tr = pd.concat([train, aug_rows(train, test)]) if aug else train
        y = tr.kss.to_numpy(float) - a * tr.kss_base.to_numpy(float)
        m = factory().fit(tr[cols].to_numpy(float), y)
        return np.ravel(m.predict(test[cols].to_numpy(float))) + a * test.kss_base.to_numpy(float)
    return fp


def pipe(est):
    return lambda: make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                                 clone(est))


def pls(k):
    return pipe(PLSRegression(n_components=k, scale=False))


def topk(cols, k, a):
    """Keep the k features most correlated (absolute) with the target in the training rows, then RidgeCV."""
    def fp(train, test):
        y = train.kss.to_numpy(float) - a * train.kss_base.to_numpy(float)
        X = train[cols].to_numpy(float)
        r = [abs(np.corrcoef(X[:, j], y)[0, 1]) if np.nanstd(X[:, j]) > 0 else 0 for j in range(X.shape[1])]
        keep = [cols[j] for j in np.argsort(r)[::-1][:k]]
        return lin(keep, a=a)(train, test)
    return fp


def within_ridge(cols=TYPING, a=1.0):
    """Slopes from within-person variation only (person-demeaned features and target, as fixed effects),
    with the pooled intercept for prediction; pred = a*kss_base + c + w.x."""
    def fp(train, test):
        f = prep(train, cols)
        Xtr, Xte = f(train), f(test)
        y = train.kss.to_numpy(float) - a * train.kss_base.to_numpy(float)
        pids = train.pid.to_numpy()
        m = RidgeCV(alphas=ALPHAS, fit_intercept=False).fit(Xtr - group_mean(Xtr, pids), y - group_mean(y, pids))
        c = y.mean() - Xtr.mean(0) @ m.coef_
        return c + Xte @ m.coef_ + a * test.kss_base.to_numpy(float)
    return fp


# ---- pairwise within-person ranking ---------------------------------------------------------

def _pairs(X, y, pids, weight):
    D, L, W = [], [], []
    for p in np.unique(pids):
        idx = np.flatnonzero(pids == p)
        for ii in range(len(idx)):
            for jj in range(ii + 1, len(idx)):
                i, j = idx[ii], idx[jj]
                if y[i] != y[j]:
                    diff, lab = X[i] - X[j], int(y[i] > y[j])
                    w = abs(y[i] - y[j]) if weight else 1.0
                    D += [diff, -diff]
                    L += [lab, 1 - lab]
                    W += [w, w]
    return np.array(D), np.array(L), np.array(W)


def pairwise(cols=TYPING, C=1.0, a=1.0, weight=False, aug=False):
    """Logistic regression on within-person feature differences (label: which check-in had higher KSS).
    The score is standardized on the fitting rows; KSS = a*kss_base + c + b*score, with b (>= 0) and c
    fitted on inner leave-one-person-out scores of the training rows."""
    def scorer(rows):
        f = prep(rows, cols)
        X = f(rows)
        D, L, W = _pairs(X, rows.kss.to_numpy(float), rows.pid.to_numpy(), weight)
        if len(D) == 0:
            return None
        w = LogisticRegression(C=C, fit_intercept=False, max_iter=2000).fit(D, L, sample_weight=W).coef_.ravel()
        s = X @ w
        mu, sd = s.mean(), (s.std() if s.std() > 0 else 1.0)
        return lambda fr: (f(fr) @ w - mu) / sd

    def fp(train, test):
        rows = pd.concat([train, aug_rows(train, test)]) if aug else train
        score = scorer(rows)
        target = train.kss.to_numpy(float) - a * train.kss_base.to_numpy(float)
        if score is None:
            return np.full(len(test), target.mean()) + a * test.kss_base.to_numpy(float)
        s_oof = np.full(len(train), np.nan)
        for tr, te in inner_splits(train):
            inner = train[tr]
            if aug:
                inner = pd.concat([inner, aug_rows(inner, train[te])])
            sc = scorer(inner)
            if sc is not None:
                s_oof[te] = sc(train[te])
        ok = ~np.isnan(s_oof)
        b, c = 0.0, target.mean()
        if ok.sum() >= 3 and np.ptp(s_oof[ok]) > 0:
            b, c = np.polyfit(s_oof[ok], target[ok], 1)
            if b <= 0:
                b, c = 0.0, target.mean()
        fp.slopes.append(b)
        return c + b * score(test) + a * test.kss_base.to_numpy(float)
    fp.slopes = []
    return fp


# ---- ordinal regression ---------------------------------------------------------------------

class OrdinalLogit:
    """Cumulative-logit (proportional odds) model with an L2 penalty on the slopes. Predicts the
    expected category. Categories are the distinct training values."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        self.cats = np.unique(y)
        K, n, p = len(self.cats), len(y), X.shape[1]
        self.p = p
        if K == 1:
            self.w, self.theta = np.zeros(p), np.array([])
            return self
        yi = np.searchsorted(self.cats, y)

        def unpack(par):
            return par[:p], np.concatenate([[par[p]], par[p] + np.cumsum(np.exp(par[p + 1:])) ])

        rows = np.arange(n)
        upper_ok, lower_ok = yi <= K - 2, yi >= 1  # row's category has a finite upper / lower threshold

        def nll(par):
            """Penalized negative log-likelihood and its analytic gradient."""
            w, th = unpack(par)
            eta = X @ w
            z = th[None, :] - eta[:, None]
            cdf = np.hstack([np.zeros((n, 1)), expit(z), np.ones((n, 1))])
            pdf = cdf[:, 1:-1] * (1 - cdf[:, 1:-1])
            pdf = np.hstack([np.zeros((n, 1)), pdf, np.zeros((n, 1))])
            pr = np.clip(cdf[rows, yi + 1] - cdf[rows, yi], 1e-12, None)
            f_up, f_lo = pdf[rows, yi + 1], pdf[rows, yi]
            g_eta = (f_up - f_lo) / pr  # d(-log pr)/d eta
            g_th = np.zeros(K - 1)
            np.add.at(g_th, yi[upper_ok], -f_up[upper_ok] / pr[upper_ok])
            np.add.at(g_th, yi[lower_ok] - 1, f_lo[lower_ok] / pr[lower_ok])
            grad = np.empty_like(par)
            grad[:p] = X.T @ g_eta + self.alpha * w
            grad[p] = g_th.sum()
            grad[p + 1:] = np.exp(par[p + 1:]) * np.cumsum(g_th[::-1])[::-1][1:]
            return -np.sum(np.log(pr)) + 0.5 * self.alpha * w @ w, grad

        q = np.clip(np.array([(yi <= k).mean() for k in range(K - 1)]), 0.02, 0.98)
        th0 = np.log(q / (1 - q))
        th0 = np.maximum.accumulate(th0 + np.arange(K - 1) * 1e-3)
        inc = np.log(np.clip(np.diff(th0), 1e-3, None))
        x0 = np.concatenate([np.zeros(p), [th0[0]], inc])
        res = minimize(nll, x0, method="L-BFGS-B", jac=True)
        self.w, self.theta = unpack(res.x)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, float)
        if len(self.cats) == 1:
            return np.ones((len(X), 1))
        cum = expit(self.theta[None, :] - (X @ self.w)[:, None])
        cum = np.hstack([np.zeros((len(X), 1)), cum, np.ones((len(X), 1))])
        return np.diff(cum, axis=1)

    def predict(self, X):
        return self.predict_proba(X) @ self.cats


def ordinal(cols=TYPING, alpha=1.0, a=0.0):
    """Ordinal model on kss (a=0) or on kss_delta (a=1, then add kss_base back)."""
    def fp(train, test):
        f = prep(train, cols)
        y = train.kss.to_numpy(float) - a * train.kss_base.to_numpy(float)
        m = OrdinalLogit(alpha).fit(f(train), y)
        return m.predict(f(test)) + a * test.kss_base.to_numpy(float)
    return fp


def ordered_on_score(cols=TYPING):
    """statsmodels OrderedModel (logit) on one number: the unanchored ridge score, computed out of fold
    (inner leave-one-person-out) for the training rows so the mapping isn't fitted on overfit scores."""
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    def fp(train, test):
        y = train.kss.to_numpy(float)
        full = ridge().fit(train[cols].to_numpy(float), y)
        s_test = full.predict(test[cols].to_numpy(float))
        s_oof = np.full(len(train), np.nan)
        for tr, te in inner_splits(train):
            s_oof[te] = ridge().fit(train[tr][cols].to_numpy(float), y[tr]).predict(train[te][cols].to_numpy(float))
        ok = ~np.isnan(s_oof)
        cats = np.unique(y[ok])
        try:
            if len(cats) < 3:
                raise ValueError("too few categories")
            endog = pd.Series(pd.Categorical(y[ok], categories=cats, ordered=True))
            res = OrderedModel(endog, s_oof[ok].reshape(-1, 1), distr="logit").fit(method="bfgs", disp=False, maxiter=500)
            pr = np.asarray(res.model.predict(res.params, exog=s_test.reshape(-1, 1)))
            return pr @ cats
        except Exception:
            return s_test
    return fp


# ---- practice removal -----------------------------------------------------------------------

def practice(cols=TYPING, a=1.0, oracle=False):
    """Residualize the training features on log(check-in number) within person (pooled slope per feature,
    estimated on the training rows plus those people's baseline check-ins), fit ridge on the residualized
    features, and predict from raw test features (deployable). oracle=True also residualizes the test
    features with their check-in number: NOT deployable, an upper bound only."""
    def fp(train, test):
        est = pd.concat([train, aug_rows(train, test)])
        X = est[cols].to_numpy(float)
        X = np.where(np.isnan(X), np.nanmedian(X, axis=0), X)
        l, pids = est.log_n.to_numpy(float), est.pid.to_numpy()
        Xd, ld = X - group_mean(X, pids), l - group_mean(l, pids)
        beta = (Xd * ld[:, None]).sum(0) / (ld ** 2).sum()
        lbar = train.log_n.mean()
        Xtr = train[cols].to_numpy(float) - np.outer(train.log_n.to_numpy(float) - lbar, beta)
        Xte = test[cols].to_numpy(float)
        if oracle:
            Xte = Xte - np.outer(test.log_n.to_numpy(float) - lbar, beta)
        y = train.kss.to_numpy(float) - a * train.kss_base.to_numpy(float)
        return ridge().fit(Xtr, y).predict(Xte) + a * test.kss_base.to_numpy(float)
    return fp


# ---- composite label ------------------------------------------------------------------------

def composite(cols=TYPING, a=1.0, factory=ridge):
    """Fit the mean of z-scored KSS and mental fatigue (both as changes from baseline when a=1),
    then map the prediction back to KSS with the training rows' regression of KSS on the composite."""
    def fp(train, test):
        if a == 1.0:
            yk, ym = train.kss_delta.to_numpy(float), train.mf_delta.to_numpy(float)
        else:
            yk, ym = train.kss.to_numpy(float), train.mental_fatigue.to_numpy(float)
        zk = (yk - yk.mean()) / yk.std()
        zm = (ym - np.nanmean(ym)) / np.nanstd(ym) if np.nanstd(ym) > 0 else zk
        zm = np.where(np.isnan(zm), zk, zm)
        comp = (zk + zm) / 2
        m = factory().fit(train[cols].to_numpy(float), comp)
        b, c = np.polyfit(comp, yk, 1)
        out = c + b * np.ravel(m.predict(test[cols].to_numpy(float)))
        return out + (test.kss_base.to_numpy(float) if a == 1.0 else 0.0)
    return fp


# ---- not deployable reference ---------------------------------------------------------------

def person_offset(cols=TYPING, k=2.0):
    """NOT deployable (uses the person's own earlier KSS ratings beyond the baseline): unanchored ridge plus
    the person's mean earlier residual, shrunk by n/(n+k). In LOPO it equals unanchored ridge."""
    def fp(train, test):
        y = train.kss.to_numpy(float)
        m = ridge().fit(train[cols].to_numpy(float), y)
        res = y - m.predict(train[cols].to_numpy(float))
        out = m.predict(test[cols].to_numpy(float))
        for i, p in enumerate(test.pid.to_numpy()):
            r = res[train.pid.to_numpy() == p]
            if len(r):
                out[i] += r.sum() / (len(r) + k)
        return out
    return fp
