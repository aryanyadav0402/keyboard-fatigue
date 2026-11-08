"""Kernel / distribution-distance models on log-IKI and log-hold distributions."""
import itertools, json
from scipy.stats import wasserstein_distance as w1
from sklearn.linear_model import Ridge
from common import *

ids = [c for c in SEQ if T.set_index("checkin_id").pid[c] in BASE]; ix = {c: i for i, c in enumerate(ids)}
pid_of = dict(zip(T.checkin_id, T.pid))
def pool(p, k):  # baseline pool for person p, channel k (0 iki, 1 hold)
    return np.concatenate([SEQ[c][k] for c in BASE[p]])
MED = {p: (np.median(pool(p, 0)), np.median(pool(p, 1))) for p in BASE}
def cen(c, k): return SEQ[c][k] - MED[pid_of[c]][k]
# pairwise W1 matrices: raw and baseline-centred, for iki and hold
def pw(k, centred):
    arrs = [cen(c, k) if centred else SEQ[c][k] for c in ids]
    n = len(ids); M = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n): M[i, j] = M[j, i] = w1(arrs[i], arrs[j])
    return M
DM = {(k, c): pw(k, c) for k in (0, 1) for c in (False, True)}
# distance of each check-in to own baseline pool, signed shifts
def dist_feats(c):
    p = pid_of[c]; f = []
    for k in (0, 1):
        a, b = SEQ[c][k], pool(p, k)
        qs = [10, 50, 90]
        f += [w1(a, b), np.median(a) - np.median(b)] + [np.percentile(a, q) - np.percentile(b, q) for q in qs]
        f += [a.std() - b.std()]
    return f
DF = {c: dist_feats(c) for c in ids}  # [w1,medshift,q10,q50,q90,sd] x (iki, hold)
names = [f"{ch}_{n}" for ch in ("iki", "hold") for n in ("w1", "med", "q10", "q50", "q90", "sd")]
col = {n: i for i, n in enumerate(names)}

def getX(frame, cols):
    return np.array([[DF[c][col[n]] for n in cols] for c in frame.checkin_id])
def target(train, anch): return (train.kss_delta if anch else train.kss).to_numpy(float)
def back(pred, test, anch): return pred + (test.kss_base.to_numpy(float) if anch else 0)

def inner_lopo(train, fitfn, grid):
    """pick the grid point with best inner leave-one-person-out MAE; return the winner"""
    best, bs = None, 1e9
    for g in grid:
        err = []
        for p in train.pid.unique():
            tr, te = train[train.pid != p], train[train.pid == p]
            if tr.pid.nunique() < 2: continue
            err.append(np.abs(fitfn(tr, te, g) - te.kss.to_numpy(float)).mean())
        s = np.mean(err)
        if s < bs: best, bs = g, s
    return best

def make_ridge(cols, anch=True, alphas=(1, 10, 100, 1000)):
    def fit(tr, te, a):
        X = getX(tr, cols); mu, sd = X.mean(0), X.std(0) + 1e-9
        m = Ridge(alpha=a).fit((X - mu) / sd, target(tr, anch))
        return back(m.predict((getX(te, cols) - mu) / sd), te, anch)
    return lambda tr, te: fit(tr, te, inner_lopo(tr, fit, alphas))

def make_krr(chs, centred=True, anch=True, mults=(0.5, 1, 2), alphas=(0.1, 1, 10)):
    mats = [DM[(k, centred)] for k in chs]
    def fit(tr, te, g):
        m, a = g
        K = sum(np.exp(-M / (m * np.median(M[M > 0]))) for M in mats) / len(mats)
        i, j = [ix[c] for c in tr.checkin_id], [ix[c] for c in te.checkin_id]
        y = target(tr, anch); mu = y.mean()
        al = np.linalg.solve(K[np.ix_(i, i)] + a * np.eye(len(i)), y - mu)
        return back(mu + K[np.ix_(j, i)] @ al, te, anch)
    grid = list(itertools.product(mults, alphas))
    return lambda tr, te: fit(tr, te, inner_lopo(tr, fit, grid))

CANDS = {
 "ridge: iki w1 (anch)": make_ridge(["iki_w1"]),
 "ridge: iki med shift (anch)": make_ridge(["iki_med"]),
 "ridge: iki w1+hold w1 (anch)": make_ridge(["iki_w1", "hold_w1"]),
 "ridge: iki+hold shifts q10/50/90 (anch)": make_ridge(["iki_q10", "iki_q50", "iki_q90", "hold_q10", "hold_q50", "hold_q90"]),
 "ridge: all 12 baseline-dist feats (anch)": make_ridge(names),
 "ridge: all 12 baseline-dist feats (abs)": make_ridge(names, anch=False),
 "krr W1 iki centred (anch)": make_krr([0]),
 "krr W1 iki+hold centred (anch)": make_krr([0, 1]),
 "krr W1 iki+hold centred (abs)": make_krr([0, 1], anch=False),
 "krr W1 iki+hold RAW uncentred (anch)": make_krr([0, 1], centred=False),
}
if __name__ == "__main__":
    out = {}
    for n, f in CANDS.items():
        r = h.evaluate(D, f); out[n] = r
        print(f"{n:45s} lopo MAE {r['lopo']['MAE']:.3f} skill {r['lopo']['skill']:+.3f} conc {r['lopo']['within_concordance']:.3f} | roll MAE {r['rolling']['MAE']:.3f} skill {r['rolling']['skill']:+.3f} conc {r['rolling']['within_concordance']:.3f}", flush=True)
    json.dump(out, open("exp1_results.json", "w"), default=float)
