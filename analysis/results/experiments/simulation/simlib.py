"""Feature-level simulator + fast copies of the harness evaluation (LOPO / rolling / perm test).

Why feature-level: simulate.py writes raw keystrokes (slow: seconds per world). The questions here need
thousands of worlds, so this module draws the processed table directly (the same columns the harness
uses: the 6 TYPING features as changes from each person's first two check-ins, hours_rel, the two SWELL
inputs, kss, kss_base) with spreads and correlations taken from the real scored table. simulate.py is
used separately (calib_pipeline.py) to check that this shortcut agrees with the full pipeline.

fast_ridge() reproduces model.ridge() (standardise, RidgeCV with GCV over the same alphas); checked
against the harness in check_fast_ridge.py.
"""

from types import SimpleNamespace

import numpy as np

ALPHAS = np.logspace(-2, 3, 30)
SWELL = {"rate": -0.09978783562013131, "errr": -0.32602494462135545}

# ---- fast ridge ------------------------------------------------------------------------------


def fast_ridge(Xtr, ytr, Xte, alpha=None):
    """Standardise columns, centre y, ridge with intercept; alpha by leave-one-out (GCV) unless given."""
    n = len(ytr)
    if Xtr.shape[1] == 0:
        return np.full(len(Xte), ytr.mean())
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Xs, Xt = (Xtr - mu) / sd, (Xte - mu) / sd
    ym = ytr.mean()
    yc = ytr - ym
    U, d, Vt = np.linalg.svd(Xs, full_matrices=False)
    uty = U.T @ yc
    d2 = d ** 2
    if alpha is None:
        S = d2[None, :] / (d2[None, :] + ALPHAS[:, None])
        yhat = (S * uty) @ U.T + ym
        h = 1.0 / n + S @ (U ** 2).T
        loo = ((ytr - yhat) / (1 - h)) ** 2
        alpha = ALPHAS[np.argmin(loo.mean(1))]
    coef = Vt.T @ (d / (d2 + alpha) * uty)
    return Xt @ coef + ym


# ---- world generator -------------------------------------------------------------------------

# order: rule_cost, backspace, accuracy, gap_cv, pause1s, hold, (SWELL inputs) rate, errr
NAMES = ["rule", "bs", "acc", "cv", "pause", "hold", "rate", "errr"]
FSIGN = np.array([1, 1, -1, 1, 1, 1, -1, 1], float)    # direction of a fatigue effect
PSIGN = np.array([-1, -1, 1, -1, -1, -1, 1, -1], float)  # direction of practice (faster, cleaner)
# Real scored-table correlation of the 8 features (24 rows), shrunk halfway to identity below.
REAL_CORR = np.array([
    [1.00, 0.07, 0.04, -0.15, 0.12, 0.14, 0.09, -0.02],
    [0.07, 1.00, -0.56, 0.30, -0.65, 0.02, -0.11, 0.81],
    [0.04, -0.56, 1.00, 0.37, 0.72, -0.09, -0.20, -0.22],
    [-0.15, 0.30, 0.37, 1.00, -0.07, 0.16, -0.45, 0.63],
    [0.12, -0.65, 0.72, -0.07, 1.00, -0.18, -0.07, -0.52],
    [0.14, 0.02, -0.09, 0.16, -0.18, 1.00, 0.03, 0.16],
    [0.09, -0.11, -0.20, -0.45, -0.07, 0.03, 1.00, -0.41],
    [-0.02, 0.81, -0.22, 0.63, -0.52, 0.16, -0.41, 1.00]])
NOISE_CORR = 0.5 * REAL_CORR + 0.5 * np.eye(8)
NOISE_CHOL = np.linalg.cholesky(NOISE_CORR)
SIGMA_N = 0.72     # within-person noise sd per feature (units of the feature's real sd)
N_EXTRA = 12       # pure-noise pool features for the selection-bias runs
H_STEP = 0.8       # hours between check-ins (real hours_rel at check-in 3 is ~1.0, mean 3.9 over check-ins 3-8)
H_AWAKE0 = 8.0     # hours awake at the first check-in (only matters for the clock after sleep)

DEFAULT = dict(
    rho=0.0,        # loading of every feature on true tiredness (correlation of feature with T)
    pi=0.0,         # loading of every feature on practice (log check-in number)
    slope=0.6, slope_sd=0.25,  # KSS rise per hour awake (per person)
    sd_a=0.9,       # between-person KSS level
    sd_u=0.8,       # true tiredness fluctuation that typing can see
    sd_e=0.7,       # self-report noise typing cannot see
    L=4.2, bump=1.6,  # mean KSS level; extra KSS at the first two (baseline) check-ins
    extra=0,        # number of pure-noise extra features
)
_SCALE = {}


def _raw_world(rng, counts, p, morning=None, sleepers=0.6):
    """Draw every check-in. counts = scored night check-ins per person (plus 2 baseline each).
    morning = morning check-ins per person (list or int), after sleep for a share `sleepers` of people."""
    N = len(counts)
    if morning is None:
        morning = [0] * N
    elif np.isscalar(morning):
        morning = [morning] * N
    nsleep = int(round(sleepers * N))
    rows = []
    for i in range(N):
        n_night = 2 + counts[i]
        L = p["L"] + rng.normal(0, p["sd_a"])
        s = max(0.05, rng.normal(p["slope"], p["slope_sd"]))
        h = np.concatenate([[0.0], np.cumsum(np.maximum(0.3, rng.normal(H_STEP, 0.15, n_night - 1)))])
        hb = h[:2].mean()
        slept = i < nsleep
        u_prev = rng.normal(0, p["sd_u"])
        K = n_night + morning[i]
        h_all = np.zeros(K)
        awake = np.zeros(K)
        T_det = np.zeros(K)
        h_all[:n_night] = h
        awake[:n_night] = H_AWAKE0 + h
        T_det[:n_night] = s * (h - hb)
        for m in range(morning[i]):
            k = n_night + m
            if slept:
                hw = 1.0 + H_STEP * m
                awake[k] = hw
                h_all[k] = h[-1] + 8 + H_STEP * m
                T_det[k] = -1.0 + 0.5 * s * hw  # rested: low tiredness, creeping up again
            else:
                h_all[k] = h[-1] + H_STEP * (m + 1) + 1.0
                awake[k] = H_AWAKE0 + h_all[k]
                T_det[k] = s * (h_all[k] - hb)
        u = np.zeros(K)
        for k in range(K):
            u_prev = 0.4 * u_prev + np.sqrt(1 - 0.16) * rng.normal(0, p["sd_u"])
            u[k] = u_prev
        T = T_det + u
        e = rng.normal(0, p["sd_e"], K)
        kss = np.clip(np.round(L + T + p["bump"] * (np.arange(K) < 2) + e), 1, 9)
        hours_rel = awake - awake[:2].mean()
        rows.append(dict(i=i, K=K, n_night=n_night, T=T, kss=kss, hours_rel=hours_rel, lp=np.log(np.arange(1, K + 1)),
                         t=h_all + rng.uniform(0, 0.4), slept=slept))
    return rows


def _scales():
    """sd of T and of the practice term over scored night rows, so loadings are correlations."""
    if not _SCALE:
        rng = np.random.default_rng(12345)
        r = _raw_world(rng, [6] * 400, DEFAULT)
        T = np.concatenate([x["T"][2:] for x in r])
        lp = np.concatenate([x["lp"][2:] - x["lp"][:2].mean() for x in r])
        _SCALE["T"], _SCALE["P"] = T.std(), lp.std()
    return _SCALE


def make_world(rng, counts, morning=None, sleepers=0.6, **kw):
    """Return (all rows as a Tab with `phase` 0 baseline / 1 night / 2 morning, `scored` mask)."""
    p = DEFAULT | kw
    sc = _scales()
    rows = _raw_world(rng, counts, p, morning, sleepers)
    out = {k: [] for k in ["pid", "kss", "base", "clk", "lp", "t", "phase", "slept", "X", "P", "ck"]}
    for r in rows:
        K = r["K"]
        zT = r["T"] / sc["T"]
        lp0 = r["lp"] - r["lp"][:2].mean()
        zP = (r["lp"]) / sc["P"]
        sig = np.sqrt(max(0.04, (0.75 - p["rho"] ** 2 - p["pi"] ** 2) / 1.5))  # keep dev spread at the real size
        eps = rng.normal(size=(K, 8)) @ NOISE_CHOL.T * sig
        load = p["rho"] * FSIGN[None, :] * np.where(np.arange(8) == 7, 0.8, 1.0)
        x = load * zT[:, None] + p["pi"] * PSIGN[None, :] * zP[:, None] + eps
        dev = x - x[:2].mean(0)
        if p["extra"]:
            f = rng.normal(size=(K, 1))
            ex = SIGMA_N * (0.45 * f + np.sqrt(1 - 0.45 ** 2) * rng.normal(size=(K, p["extra"])))
            ex = ex - ex[:2].mean(0)
            pool = np.hstack([dev, ex])
        else:
            pool = dev
        base = r["kss"][:2].mean()
        ph = np.where(np.arange(K) < 2, 0, np.where(np.arange(K) < r["n_night"], 1, 2))
        out["pid"] += [r["i"]] * K
        out["kss"].append(r["kss"]); out["base"].append(np.full(K, base))
        out["clk"].append(r["hours_rel"]); out["lp"].append(lp0)
        out["t"].append(r["t"]); out["phase"].append(ph); out["slept"].append(np.full(K, r["slept"]))
        out["X"].append(dev); out["P"].append(pool); out["ck"].append(np.arange(1, K + 1))
    cat = lambda k: np.concatenate(out[k])
    tab = SimpleNamespace(pid=np.array(out["pid"]), kss=cat("kss"), base=cat("base"), clk=cat("clk"), lp=cat("lp"),
                          t=cat("t"), phase=cat("phase"), slept=cat("slept"), X=np.vstack(out["X"]),
                          P=np.vstack(out["P"]), ck=cat("ck"))
    return tab


def sub(tab, mask):
    mask = np.asarray(mask)
    return SimpleNamespace(**{k: v[mask] for k, v in vars(tab).items()})


def scored(tab, morning=True):
    return sub(tab, tab.phase >= 1 if morning else tab.phase == 1)


# ---- models (same definitions as the harness baselines) ----------------------------------------

TYPING6 = [0, 1, 2, 3, 4, 5]


def anchored_model(cols_fn):
    def fp(tr, te):
        return fast_ridge(cols_fn(tr), tr.kss - tr.base, cols_fn(te)) + te.base
    return fp


def guess(tr, te):
    return np.full(len(te.kss), tr.kss.mean())


v1 = anchored_model(lambda t: t.X[:, TYPING6])
clock = anchored_model(lambda t: t.clk[:, None])
practice = anchored_model(lambda t: t.lp[:, None])


def _stress(tr, t):
    sp = [tr.X[:, 6].std(ddof=1), tr.X[:, 7].std(ddof=1)]
    return (SWELL["rate"] * t.X[:, 6] / sp[0] + SWELL["errr"] * t.X[:, 7] / sp[1])[:, None]


def v3(tr, te):
    return fast_ridge(_stress(tr, tr), tr.kss - tr.base, _stress(tr, te)) + te.base


def v1_plus_practice(tr, te):
    f = lambda t: np.hstack([t.X[:, TYPING6], t.lp[:, None]])
    return fast_ridge(f(tr), tr.kss - tr.base, f(te)) + te.base


BASELINES = {"guess": guess, "clock": clock, "v1": v1, "v3": v3}

# ---- evaluation --------------------------------------------------------------------------------


def lopo_predict(d, fp):
    pred, g = np.full(len(d.kss), np.nan), np.full(len(d.kss), np.nan)
    for p in np.unique(d.pid):
        te = d.pid == p
        tr = ~te
        a, b = sub(d, tr), sub(d, te)
        pred[te] = np.clip(fp(a, b), 1, 9)
        g[te] = a.kss.mean()
    return pred, g


def rolling_predict(d, fp, min_train=10):
    order = np.argsort(d.t, kind="stable")
    pred, g = np.full(len(d.kss), np.nan), np.full(len(d.kss), np.nan)
    ts = d.t
    for i in range(len(ts)):
        tr = ts < ts[i]
        if tr.sum() >= min_train:
            a, b = sub(d, tr), sub(d, np.arange(len(ts)) == i)
            pred[i] = np.clip(fp(a, b), 1, 9)[0]
            g[i] = a.kss.mean()
    return pred, g


def skill_of(y, pred, g):
    m = ~np.isnan(pred)
    return 1 - ((pred[m] - y[m]) ** 2).sum() / ((g[m] - y[m]) ** 2).sum()


def concordance(pid, y, pred):
    good, pairs = 0.0, 0
    for p in np.unique(pid):
        m = pid == p
        k, q = y[m], pred[m]
        dk = np.sign(k[:, None] - k[None, :])
        dq = np.sign(q[:, None] - q[None, :])
        use = np.triu(dk != 0, 1)
        s = (dk * dq)[use]
        good += (s > 0).sum() + 0.5 * (s == 0).sum()
        pairs += use.sum()
    return good / pairs if pairs else np.nan


def metrics(d, pred, g, conc=True):
    m = ~np.isnan(pred)
    y = d.kss
    out = {"MAE": float(np.mean(np.abs(pred[m] - y[m]))), "skill": float(skill_of(y, pred, g))}
    if conc:
        out["conc"] = float(concordance(d.pid[m], y[m], pred[m]))
    return out


def perm_p(d, fp, n=200, rng=None, scheme="lopo"):
    """The harness perm_test: shuffle KSS within person, keep kss_base, rerun the procedure."""
    rng = rng or np.random.default_rng(0)
    pr = lopo_predict if scheme == "lopo" else rolling_predict
    obs = skill_of(d.kss, *pr(d, fp))
    null = np.empty(n)
    idx = {p: np.where(d.pid == p)[0] for p in np.unique(d.pid)}
    dd = SimpleNamespace(**vars(d))
    for b in range(n):
        k = d.kss.copy()
        for p, ii in idx.items():
            k[ii] = rng.permutation(k[ii])
        dd.kss = k
        null[b] = skill_of(k, *pr(dd, fp))
    return obs, (np.sum(null >= obs) + 1) / (n + 1)
