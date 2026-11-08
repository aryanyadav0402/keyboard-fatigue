"""Extra: (1) how often null / practice / fatigue worlds reproduce the REAL v1 numbers (LOPO skill .217, concordance .837);
(2) the sign diagnostic: share of v1's standardised coefficients pointing in the fatigue direction (slower/noisier/more errors = more tired)."""
import sys; sys.path.insert(0, "/Users/aryan/Projects/hacktheburgh/keyboard-fatigue-nobaseline/analysis/results/experiments/simulation")
sys.path.insert(0, "/Users/aryan/Projects/hacktheburgh/keyboard-fatigue-nobaseline/analysis/results/experiments")
import numpy as np, pandas as pd
import simlib as S
from multiprocessing import Pool
REAL = [7, 5, 6, 6]

def coefs(d):
    X = d.X[:, :6]; y = d.kss - d.base
    Xs = (X - X.mean(0)) / X.std(0)
    # ridge at the alpha RidgeCV would pick, via the same GCV code path: fit by predicting with unit vectors
    base = S.fast_ridge(X, y, np.vstack([X.mean(0)]))[0]
    c = np.array([S.fast_ridge(X, y, (X.mean(0) + np.eye(6)[j] * X.std(0)[None, :])[:1])[0] - base for j in range(6)])
    return c   # change in predicted delta per +1 sd of feature j

def one(a):
    rho, pi, seed = a
    d = S.scored(S.make_world(np.random.default_rng(seed), REAL, rho=rho, pi=pi), morning=False)
    m = S.metrics(d, *S.lopo_predict(d, S.v1))
    c = coefs(d)
    agree = int((np.sign(c) == S.FSIGN[:6]).sum())
    return dict(skill=m["skill"], conc=m["conc"], agree=agree)

if __name__ == "__main__":
    import harness as H
    d = H.scored_panel(H.load_table())
    from types import SimpleNamespace
    tab = SimpleNamespace(kss=d.kss.to_numpy(float), base=d.kss_base.to_numpy(float), X=d[H.TYPING].to_numpy(float))
    c = coefs(tab)
    print("REAL v1 coefficient signs (rule_cost, backspace, accuracy, gap_cv, pause1s, hold):", np.sign(c).astype(int).tolist(),
          "-> in fatigue direction:", int((np.sign(c) == S.FSIGN[:6]).sum()), "of 6 (fatigue direction = +,+,-,+,+,+)")
    rows = []
    with Pool(9) as p:
        for name, (rho, pi) in {"null": (0, 0), "practice .5": (0, .5), "practice .7": (0, .7), "fatigue .4": (.4, 0), "fatigue .55": (.55, 0)}.items():
            df = pd.DataFrame(p.map(one, [(rho, pi, 70000 + i) for i in range(2000)]))
            like = (df.skill >= 0.217) & (df.conc >= 0.837)
            rows.append(dict(world=name, P_skill_ge_real=(df.skill >= .217).mean(), P_conc_ge_real=(df.conc >= .837).mean(),
                             P_both_ge_real=like.mean(), mean_signs_in_fatigue_dir=df.agree.mean(),
                             P_ge5_of6_fatigue_dir=(df.agree >= 5).mean(), P_le1_of6_fatigue_dir=(df.agree <= 1).mean(),
                             P_both_ge_real_and_ge5_fatigue_dir=(like & (df.agree >= 5)).mean()))
    out = pd.DataFrame(rows).round(3); print(out.to_string()); out.to_csv("real_match_and_signs.csv", index=False)
