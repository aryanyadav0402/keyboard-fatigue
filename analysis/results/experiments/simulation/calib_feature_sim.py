"""Calibrate the feature-level simulator to the real scored table: spreads, KSS stats, baseline skills vs rho."""
import sys; sys.path.insert(0, ".")
import numpy as np, simlib as S
from multiprocessing import Pool
REAL = [7, 5, 6, 6]
def one(args):
    seed, rho, pi = args
    rng = np.random.default_rng(seed)
    w = S.make_world(rng, REAL, rho=rho, pi=pi)
    d = S.scored(w)
    out = {"klp": 0, "sd": d.X.std(0), "kss_mean": d.kss.mean(), "kss_sd": d.kss.std(), "base": d.base.mean(), "delta": (d.kss-d.base).mean(), "dsd": (d.kss-d.base).std()}
    for n, fp in S.BASELINES.items():
        m = S.metrics(d, *S.lopo_predict(d, fp)); out[n+"_skill"]=m["skill"]; out[n+"_MAE"]=m["MAE"]; out[n+"_conc"]=m["conc"]
    return out
if __name__ == "__main__":
    real_sd = np.array([0.341, 4.485/ 4.485, 0.044/0.044, 0.464, 0.073, 12.395, 0.372, 0.118])
    with Pool(8) as p:
        for rho, pi in [(0,0),(0.5,0),(0.7,0),(0.85,0),(0,0.5),(0,0.7),(0,0.85),(0.5,0.5)]:
            r = p.map(one, [(s, rho, pi) for s in range(600)])
            g = lambda k: np.mean([x[k] for x in r])
            print(f"rho={rho} pi={pi}: dev-feature sd (units of real sd) mean {np.mean([x['sd'] for x in r]):.2f}; KSS mean {g('kss_mean'):.2f} sd {g('kss_sd'):.2f} base {g('base'):.2f} delta {g('delta'):.2f} dsd {g('dsd'):.2f}")
            print("   " + "  ".join(f"{n}: skill {g(n+'_skill'):+.2f} (sd {np.std([x[n+'_skill'] for x in r]):.2f}) MAE {g(n+'_MAE'):.2f} conc {g(n+'_conc'):.2f}" for n in S.BASELINES))
