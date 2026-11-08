"""Morning batch (M=15 over 4 people), varying the share of people who actually slept before their morning check-ins."""
import sys; sys.path.insert(0, "/Users/aryan/Projects/hacktheburgh/keyboard-fatigue-nobaseline/analysis/results/experiments/simulation")
import numpy as np, pandas as pd, simlib as S
from multiprocessing import Pool
REAL = [7, 5, 6, 6]
def one(a):
    sl, rho, pi, seed = a
    w = S.make_world(np.random.default_rng(seed), REAL, morning=[4, 4, 4, 3], sleepers=sl, rho=rho, pi=pi)
    night, morn = S.sub(w, w.phase == 1), S.sub(w, w.phase == 2)
    sse = lambda n: np.sum((np.clip(S.BASELINES[n](night, morn), 1, 9) - morn.kss) ** 2)
    return dict(skill=1 - sse("v1") / sse("guess"), v1_beats_guess=sse("v1") < sse("guess"))
if __name__ == "__main__":
    rows = []
    with Pool(9) as p:
        for sl in (0.0, 0.5, 1.0):
            res = {n: pd.DataFrame(p.map(one, [(sl, r, pi, 123000 + i) for i in range(1500)], chunksize=20)) for n, (r, pi) in
                   {"null": (0, 0), "practice .6": (0, .6), "fatigue .4": (.4, 0)}.items()}
            t95 = res["null"].skill.quantile(.95)
            for n, df in res.items():
                rows.append(dict(share_slept=sl, world=n, mean_frozen_skill=df.skill.mean(), P_v1_beats_guess=df.v1_beats_guess.mean(),
                                 null95=t95, P_skill_gt_null95=(df.skill > t95).mean()))
    out = pd.DataFrame(rows).round(3); print(out.to_string()); out.to_csv("morning_by_sleep_share.csv", index=False)
