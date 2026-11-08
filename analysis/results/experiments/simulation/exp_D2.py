"""Selection bias, same-family version: variants are anchored ridge models on random subsets (2-8) of the 8 typing/SWELL
features with random alpha (what 'tweaking v1' means), plus the right fix: re-run the whole selection inside every shuffle.
Reports the margin a win must have over v1 (skill and LOPO MAE) at K = 5, 20, 50."""
import sys; sys.path.insert(0, "/Users/aryan/Projects/hacktheburgh/keyboard-fatigue-nobaseline/analysis/results/experiments/simulation")
import numpy as np, pandas as pd
import simlib as S
from multiprocessing import Pool
REAL = [7, 5, 6, 6]

def specs_(rng, n):
    out = []
    for _ in range(n):
        k = int(rng.integers(2, 9)); cols = sorted(rng.choice(8, k, replace=False).tolist())
        out.append((cols, None if rng.random() < 0.3 else float(10 ** rng.uniform(-1, 3))))
    return out

def fp_of(spec):
    cols, alpha = spec
    return lambda tr, te: S.fast_ridge(tr.X[:, cols], tr.kss - tr.base, te.X[:, cols], alpha) + te.base

def sweep(d, specs):
    sk, mae = [], []
    for s in specs:
        p, g = S.lopo_predict(d, fp_of(s)); sk.append(S.skill_of(d.kss, p, g)); mae.append(np.mean(np.abs(p - d.kss)))
    return np.array(sk), np.array(mae)

def one(a):
    rho, pi, seed, do_perm = a
    rng = np.random.default_rng(seed)
    d = S.scored(S.make_world(rng, REAL, rho=rho, pi=pi), morning=False)
    specs = specs_(rng, 50)
    sk, mae = sweep(d, specs)
    p1, g1 = S.lopo_predict(d, S.v1); v1s, v1m = S.skill_of(d.kss, p1, g1), np.mean(np.abs(p1 - d.kss))
    r = {"v1_skill": v1s, "v1_MAE": v1m}
    for K in (1, 5, 20, 50):
        r[f"best{K}_skill"] = sk[:K].max(); r[f"best{K}_maegain"] = v1m - mae[:K].min()
    if do_perm:  # whole-procedure permutation test for K=20: shuffle KSS within person, repeat selection, compare the best
        obs = sk[:20].max(); null = []
        idx = {q: np.where(d.pid == q)[0] for q in np.unique(d.pid)}
        import copy
        from types import SimpleNamespace
        dd = SimpleNamespace(**vars(d))
        for b in range(100):
            k = d.kss.copy()
            for q, ii in idx.items(): k[ii] = rng.permutation(k[ii])
            dd.kss = k
            null.append(sweep(dd, specs[:20])[0].max())
        r["whole_procedure_p"] = (np.sum(np.array(null) >= obs) + 1) / 101
    return r

if __name__ == "__main__":
    rows = []
    with Pool(9) as p:
        for name, (rho, pi) in {"null": (0, 0), "practice-only .6": (0, .6), "fatigue .4": (.4, 0)}.items():
            df = pd.DataFrame(p.map(one, [(rho, pi, 80000 + i, False) for i in range(2000)], chunksize=10))
            r = dict(world=name, v1_skill_mean=df.v1_skill.mean())
            for K in (1, 5, 20, 50):
                r[f"K{K}_best_skill_mean"] = df[f"best{K}_skill"].mean(); r[f"K{K}_best_skill_q95"] = df[f"best{K}_skill"].quantile(.95)
                r[f"K{K}_MAEgain_over_v1_q95"] = df[f"best{K}_maegain"].quantile(.95); r[f"K{K}_MAEgain_over_v1_mean"] = df[f"best{K}_maegain"].mean()
                r[f"K{K}_skill_margin_over_v1_q95"] = (df[f"best{K}_skill"] - df.v1_skill).quantile(.95)
            rows.append(r); print(name, flush=True)
        out = pd.DataFrame(rows).round(3); out.to_csv("selection_same_family.csv", index=False)
        print(out.T.to_string())
        res = []
        for name, (rho, pi) in {"null": (0, 0), "fatigue .4": (.4, 0)}.items():
            df = pd.DataFrame(p.map(one, [(rho, pi, 90000 + i, True) for i in range(300)], chunksize=2))
            res.append(dict(world=name, K=20, P_whole_procedure_p_lt_0p05=(df.whole_procedure_p < 0.05).mean(), n_worlds=300))
        print(pd.DataFrame(res)); pd.DataFrame(res).to_csv("selection_whole_procedure_perm.csv", index=False)
