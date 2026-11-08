"""Window-length variants, and v1 features + sequence distance features."""
import json, warnings
warnings.filterwarnings("ignore")
from exp2 import make, MLPRegressor
import exp1
from exp1 import *
from model import TYPING
V1 = TYPING
def getv1(fr): return fr[V1].to_numpy(float)
def make_v1plus(extra, anch=True, alphas=(1, 10, 30, 100, 300)):
    def X(fr): return np.hstack([getv1(fr)] + ([exp1.getX(fr, extra)] if extra else []))
    def fit(tr, te, a):
        A = X(tr); mu, sd = A.mean(0), A.std(0) + 1e-9
        m = Ridge(alpha=a).fit((A - mu) / sd, exp1.target(tr, anch))
        return exp1.back(m.predict((X(te) - mu) / sd), te, anch)
    return lambda tr, te: fit(tr, te, exp1.inner_lopo(tr, fit, alphas))
CANDS = {
 "win10 ridge (abs)": make("ridge", False, W=10, S=3),
 "win24 ridge (abs)": make("ridge", False, W=24, S=4),
 "win10 cnn (abs)": make("cnn", False, W=10, S=3),
 "win24 cnn (abs)": make("cnn", False, W=24, S=4),
 "win16 mlp (abs, fixed)": make("mlp", False),
 "v1 ridge, inner-tuned alpha (control)": make_v1plus([]),
 "v1 + iki_w1,hold_w1 (anch)": make_v1plus(["iki_w1", "hold_w1"]),
 "v1 + 12 baseline-dist feats (anch)": make_v1plus(exp1.names),
}
if __name__ == "__main__":
    out = {}
    for n, f in CANDS.items():
        r = h.evaluate(D, f); out[n] = r
        print(f"{n:40s} lopo MAE {r['lopo']['MAE']:.3f} skill {r['lopo']['skill']:+.3f} conc {r['lopo']['within_concordance']:.3f} | roll MAE {r['rolling']['MAE']:.3f} skill {r['rolling']['skill']:+.3f} conc {r['rolling']['within_concordance']:.3f}", flush=True)
    json.dump(out, open("exp3_results.json", "w"), default=float)
