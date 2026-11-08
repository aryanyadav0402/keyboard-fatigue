"""The four simulation experiments. Usage: python run_experiments.py A|B|C|D [reps]
A  power grid (people x check-ins x effect size)           -> power_grid.csv
B  morning batch (frozen on the night, tested on morning)  -> morning.csv
C  practice-only world, and what separates it from fatigue -> practice.csv, practice_sleep.csv
D  selection bias: best of K variants under the null       -> selection.csv
Every world is a fresh draw; seeds are fixed so reruns give the same numbers.
"""

import sys
from multiprocessing import Pool
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/aryan/Projects/hacktheburgh/keyboard-fatigue-nobaseline/analysis/results/experiments/simulation")
import simlib as S

OUT = "/Users/aryan/Projects/hacktheburgh/keyboard-fatigue-nobaseline/analysis/results/experiments/simulation/"
REAL = [7, 5, 6, 6]
NP = 9


def pm(vals, fn):
    return np.array([fn(v) for v in vals])


# ---------------------------------------------------------------- A: power grid
def world_A(a):
    counts, rho, pi, seed = a
    d = S.scored(S.make_world(np.random.default_rng(seed), counts, rho=rho, pi=pi), morning=False)
    r = {}
    for n in ("guess", "clock", "v1", "v3"):
        m = S.metrics(d, *S.lopo_predict(d, S.BASELINES[n]))
        r[n + "_MAE"], r[n + "_skill"], r[n + "_conc"] = m["MAE"], m["skill"], m["conc"]
    return r


def exp_A(reps):
    designs = {"real(7,5,6,6)": REAL}
    for N in (4, 6, 8, 12, 20):
        for m in (4, 6, 10, 15):
            designs[f"{N}x{m}"] = [m] * N
    rows = []
    with Pool(NP) as pool:
        for dn, counts in designs.items():
            for rho in (0.0, 0.25, 0.4, 0.55):
                res = pool.map(world_A, [(counts, rho, 0.0, 1000 * hash_(dn) + 17 * i) for i in range(reps)], chunksize=20)
                df = pd.DataFrame(res)
                df["design"], df["rho"] = dn, rho
                df["n_people"], df["n_scored"] = len(counts), sum(counts)
                rows.append(df)
                print(dn, rho, "v1 skill %.2f" % df.v1_skill.mean(), flush=True)
    allr = pd.concat(rows)
    allr.to_pickle(OUT + "power_raw.pkl")
    summ = []
    for dn, g in allr.groupby("design", sort=False):
        thr = g[g.rho == 0].v1_skill.quantile(0.95)
        thr_mae = g[g.rho == 0].eval("v1_MAE - guess_MAE").quantile(0.05)
        for rho, h in g.groupby("rho"):
            dm = h.v1_MAE - h.v3_MAE
            summ.append(dict(design=dn, n_people=h.n_people.iloc[0], n_scored=h.n_scored.iloc[0], rho=rho,
                             v1_skill=h.v1_skill.mean(), v1_skill_sd=h.v1_skill.std(), v1_conc=h.v1_conc.mean(),
                             P_skill_gt0=(h.v1_skill > 0).mean(), null_thr95=thr, P_skill_gt_thr=(h.v1_skill > thr).mean(),
                             P_MAE_beats_guess=(h.v1_MAE < h.guess_MAE).mean(), P_v1_beats_v3=(dm < 0).mean(),
                             dMAE_v1_v3_mean=dm.mean(), dMAE_v1_v3_sd=dm.std(), P_abs_dMAE_lt_0p02=(dm.abs() < 0.02).mean(),
                             v3_skill=h.v3_skill.mean(), clock_skill=h.clock_skill.mean()))
    pd.DataFrame(summ).round(3).to_csv(OUT + "power_grid.csv", index=False)


def hash_(s):
    return sum(ord(c) * (i + 1) for i, c in enumerate(s)) % 9973


# ---------------------------------------------------------------- B: morning batch
def split_morning(M, N=4):
    base = [M // N + (1 if i < M % N else 0) for i in range(N)]
    return base


def world_B(a):
    M, rho, pi, seed, nperm = a
    rng = np.random.default_rng(seed)
    w = S.make_world(rng, REAL, morning=split_morning(M), sleepers=0.5, rho=rho, pi=pi)
    night, allr = S.sub(w, w.phase == 1), S.scored(w)
    morn = S.sub(w, w.phase == 2)
    r = {}
    # (i) frozen on the 24 night rows, tested on the morning rows
    for n in ("guess", "v1", "v3", "clock"):
        pr = np.clip(S.BASELINES[n](night, morn), 1, 9)
        r[f"fz_{n}_MAE"] = np.mean(np.abs(pr - morn.kss))
        r[f"fz_{n}_sse"] = np.sum((pr - morn.kss) ** 2)
    r["fz_v1_skill"] = 1 - r["fz_v1_sse"] / r["fz_guess_sse"]
    r["fz_v3_skill"] = 1 - r["fz_v3_sse"] / r["fz_guess_sse"]
    # (ii) pooled LOPO over night + morning rows
    for n in ("guess", "v1", "v3"):
        m = S.metrics(allr, *S.lopo_predict(allr, S.BASELINES[n]))
        r[f"pool_{n}_MAE"], r[f"pool_{n}_skill"], r[f"pool_{n}_conc"] = m["MAE"], m["skill"], m["conc"]
    if nperm:
        r["pool_perm_p"] = S.perm_p(allr, S.v1, n=nperm, rng=rng)[1]
    return r


def exp_B(reps):
    rows = []
    with Pool(NP) as pool:
        for M in (0, 10, 15, 25):
            for rho, pi in ((0, 0), (0.25, 0), (0.4, 0), (0.55, 0), (0, 0.6)):
                if M == 0:
                    continue
                res = pool.map(world_B, [(M, rho, pi, 5000 + 31 * i, 100) for i in range(reps)], chunksize=5)
                df = pd.DataFrame(res); df["M"], df["rho"], df["pi"] = M, rho, pi
                rows.append(df)
                print("B", M, rho, pi, "frozen v1 skill %.2f" % df.fz_v1_skill.mean(), flush=True)
    raw = pd.concat(rows)
    raw.to_pickle(OUT + "morning_raw.pkl")
    summ = []
    for (M, rho, pi), g in raw.groupby(["M", "rho", "pi"]):
        null = raw[(raw.M == M) & (raw.rho == 0) & (raw.pi == 0)]
        t_fz = null.fz_v1_skill.quantile(0.95)
        t_pool = null.pool_v1_skill.quantile(0.95)
        summ.append(dict(M=M, rho=rho, pi=pi,
                         fz_v1_skill=g.fz_v1_skill.mean(), P_fz_v1_beats_guess_MAE=(g.fz_v1_MAE < g.fz_guess_MAE).mean(),
                         P_fz_skill_gt_null95=(g.fz_v1_skill > t_fz).mean(), null95_fz=t_fz,
                         P_fz_v1_beats_v3=(g.fz_v1_MAE < g.fz_v3_MAE).mean(), fz_dMAE_v1_v3_sd=(g.fz_v1_MAE - g.fz_v3_MAE).std(),
                         pool_v1_skill=g.pool_v1_skill.mean(), P_pool_skill_gt0=(g.pool_v1_skill > 0).mean(),
                         P_pool_skill_gt_null95=(g.pool_v1_skill > t_pool).mean(), P_pool_perm_p05=(g.pool_perm_p < 0.05).mean(),
                         P_pool_v1_beats_v3=(g.pool_v1_MAE < g.pool_v3_MAE).mean(),
                         pool_dMAE_v1_v3_sd=(g.pool_v1_MAE - g.pool_v3_MAE).std()))
    pd.DataFrame(summ).round(3).to_csv(OUT + "morning.csv", index=False)


# ---------------------------------------------------------------- C: practice
def world_C(a):
    counts, rho, pi, seed, nperm = a
    rng = np.random.default_rng(seed)
    d = S.scored(S.make_world(rng, counts, rho=rho, pi=pi), morning=False)
    m = S.metrics(d, *S.lopo_predict(d, S.v1))
    r = dict(skill=m["skill"], conc=m["conc"], MAE=m["MAE"])
    r["perm_p"] = S.perm_p(d, S.v1, n=nperm, rng=rng)[1]
    return r


def world_C2(a):
    """Practice-vs-fatigue tests when after-sleep check-ins exist (4 people, 2 sleep, 3 morning check-ins each)."""
    rho, pi, seed, mpp = a
    rng = np.random.default_rng(seed)
    w = S.make_world(rng, REAL, morning=mpp, sleepers=0.5, rho=rho, pi=pi)
    night, morn = S.sub(w, w.phase == 1), S.sub(w, w.phase == 2)
    allr = S.scored(w)
    out = {}
    pr_n, pr_m = S.v1(night, night), S.v1(night, morn)  # in-sample on night is fine: only used as a score
    slept = morn.slept
    # D: after sleep (KSS drops) does the frozen v1 predict a lower KSS than for the same person's last night check-in?
    ds = []
    for p in np.unique(morn.pid[slept]):
        last = night.pid == p
        i_last = np.where(last)[0][-1]
        ds.append(np.mean(pr_m[(morn.pid == p)]) - pr_n[i_last])
    out["D_after_sleep"] = np.mean(ds)
    out["kss_drop_true"] = np.mean([morn.kss[morn.pid == p].mean() - night.kss[night.pid == p][-1] for p in np.unique(morn.pid[slept])])
    # frozen skill on the after-sleep rows only
    ms = morn.slept
    a = S.sub(morn, ms)
    pr = np.clip(S.v1(night, a), 1, 9)
    out["frozen_v1_skill_after_sleep"] = 1 - np.sum((pr - a.kss) ** 2) / np.sum((night.kss.mean() - a.kss) ** 2)
    # pooled LOPO: v1 vs a practice-only model (log check-in number), night only vs night + morning
    for tag, dd in (("night", S.scored(w, morning=False)), ("all", allr)):
        s1 = S.skill_of(dd.kss, *S.lopo_predict(dd, S.v1))
        sp = S.skill_of(dd.kss, *S.lopo_predict(dd, S.practice))
        sc = S.skill_of(dd.kss, *S.lopo_predict(dd, S.clock))
        sb = S.skill_of(dd.kss, *S.lopo_predict(dd, S.v1_plus_practice))
        out[f"{tag}_v1"], out[f"{tag}_practice"], out[f"{tag}_clock"], out[f"{tag}_v1pp"] = s1, sp, sc, sb
    return out


WORLDS_C = {"null": (0, 0), "practice .3": (0, 0.3), "practice .5": (0, 0.5), "practice .7": (0, 0.7), "practice .85": (0, 0.85),
            "fatigue .4": (0.4, 0), "fatigue .55": (0.55, 0), "both .4/.5": (0.4, 0.5)}


def exp_C(reps):
    rows, rows2 = [], []
    with Pool(NP) as pool:
        for dn, counts in (("real(7,5,6,6)", REAL), ("5x6", [6] * 5)):
            for wn, (rho, pi) in WORLDS_C.items():
                res = pool.map(world_C, [(counts, rho, pi, 9000 + 13 * i, 200) for i in range(reps)], chunksize=5)
                df = pd.DataFrame(res); df["design"], df["world"] = dn, wn
                rows.append(df)
                print("C", dn, wn, "skill>0 %.2f" % (df.skill > 0).mean(), flush=True)
        for wn, (rho, pi) in WORLDS_C.items():
            for mpp in (3, 5):
                res = pool.map(world_C2, [(rho, pi, 15000 + 7 * i, mpp) for i in range(reps)], chunksize=5)
                df = pd.DataFrame(res); df["world"], df["morning_per_person"] = wn, mpp
                rows2.append(df)
                print("C2", wn, mpp, flush=True)
    raw = pd.concat(rows)
    raw.to_pickle(OUT + "practice_raw.pkl")
    summ = []
    for (dn, wn), g in raw.groupby(["design", "world"], sort=False):
        summ.append(dict(design=dn, world=wn, skill_mean=g.skill.mean(), P_skill_gt0=(g.skill > 0).mean(),
                         P_skill_ge_0p2=(g.skill >= 0.2).mean(), conc_mean=g.conc.mean(), P_conc_ge_0p65=(g.conc >= 0.65).mean(),
                         P_conc_ge_0p8=(g.conc >= 0.8).mean(), P_perm_p05=(g.perm_p < 0.05).mean(),
                         P_looks_real=((g.skill > 0) & (g.conc >= 0.65) & (g.perm_p < 0.05)).mean(),
                         P_looks_like_ours=((g.skill >= 0.2) & (g.conc >= 0.8)).mean()))
    pd.DataFrame(summ).round(3).to_csv(OUT + "practice.csv", index=False)
    raw2 = pd.concat(rows2)
    raw2.to_pickle(OUT + "practice_sleep_raw.pkl")
    s2 = []
    for (wn, mpp), g in raw2.groupby(["world", "morning_per_person"], sort=False):
        s2.append(dict(world=wn, morning_per_person=mpp, true_kss_drop=g.kss_drop_true.mean(), D_after_sleep=g.D_after_sleep.mean(),
                       P_D_negative=(g.D_after_sleep < 0).mean(), frozen_skill_after_sleep=g.frozen_v1_skill_after_sleep.mean(),
                       P_frozen_after_sleep_skill_gt0=(g.frozen_v1_skill_after_sleep > 0).mean(),
                       night_v1=g.night_v1.mean(), night_practice=g.night_practice.mean(), night_v1_minus_practice=(g.night_v1 - g.night_practice).mean(),
                       P_night_v1_beats_practice_by_0p1=((g.night_v1 - g.night_practice) > 0.1).mean(),
                       all_v1=g.all_v1.mean(), all_practice=g.all_practice.mean(), all_clock=g.all_clock.mean(), all_v1_plus_practice=g.all_v1pp.mean(),
                       P_all_v1_beats_practice_by_0p1=((g.all_v1 - g.all_practice) > 0.1).mean(),
                       P_all_v1_skill_gt0=(g.all_v1 > 0).mean()))
    pd.DataFrame(s2).round(3).to_csv(OUT + "practice_sleep.csv", index=False)


# ---------------------------------------------------------------- D: selection bias
KMAX = 50


def variant_fp(spec):
    cols, anch, alpha, clk = spec

    def feats(t):
        X = t.P[:, cols]
        return np.hstack([X, t.clk[:, None]]) if clk else X

    def fp(tr, te):
        y = tr.kss - tr.base if anch else tr.kss
        p = S.fast_ridge(feats(tr), y, feats(te), alpha)
        return p + te.base if anch else p
    return fp


def draw_variants(rng, n, pool_size=20):
    specs = []
    for _ in range(n):
        k = int(rng.integers(1, 9))
        cols = sorted(rng.choice(pool_size, k, replace=False).tolist())
        alpha = None if rng.random() < 0.3 else float(10 ** rng.uniform(-1, 3))
        specs.append((cols, bool(rng.random() < 0.7), alpha, bool(rng.random() < 0.25)))
    return specs


def world_D(a):
    counts, rho, pi, seed = a
    rng = np.random.default_rng(seed)
    d = S.scored(S.make_world(rng, counts, rho=rho, pi=pi, extra=12), morning=False)
    specs = draw_variants(rng, KMAX)
    sk = np.array([S.skill_of(d.kss, *S.lopo_predict(d, variant_fp(s))) for s in specs])
    d6 = S.skill_of(d.kss, *S.lopo_predict(d, S.v1))
    best = int(np.argmax(sk))
    spec = specs[best]
    # naive perm test: the harness's perm_test applied to the winning variant only (selection not repeated)
    _, p_naive = S.perm_p(d, variant_fp(spec), n=100, rng=rng)
    out = {"v1": d6, "naive_p_of_winner_K50": p_naive}
    for K in (1, 5, 20, 50):
        out[f"best{K}"] = sk[:K].max()
        out[f"mean{K}"] = sk[:K].mean()
    return out


def exp_D(reps):
    rows = []
    with Pool(NP) as pool:
        for dn, counts in (("real(7,5,6,6)", REAL), ("5x5", [5] * 5), ("4x6", [6] * 4)):
            for wn, (rho, pi) in (("null", (0, 0)), ("practice-only .6", (0, 0.6)), ("fatigue .4", (0.4, 0))):
                res = pool.map(world_D, [(counts, rho, pi, 30000 + 3 * i) for i in range(reps)], chunksize=5)
                df = pd.DataFrame(res); df["design"], df["world"] = dn, wn
                rows.append(df)
                print("D", dn, wn, "best50 mean %.2f" % df.best50.mean(), flush=True)
    raw = pd.concat(rows)
    raw.to_pickle(OUT + "selection_raw.pkl")
    summ = []
    for (dn, wn), g in raw.groupby(["design", "world"], sort=False):
        r = dict(design=dn, world=wn, v1_skill_mean=g.v1.mean(), v1_skill_q95=g.v1.quantile(0.95))
        for K in (1, 5, 20, 50):
            r[f"K{K}_mean_best"] = g[f"best{K}"].mean()
            r[f"K{K}_q95_best"] = g[f"best{K}"].quantile(0.95)
            r[f"K{K}_q99_best"] = g[f"best{K}"].quantile(0.99)
            r[f"K{K}_q95_best_minus_v1"] = (g[f"best{K}"] - g.v1).quantile(0.95)
            r[f"K{K}_P_best_gt_0.2"] = (g[f"best{K}"] > 0.2).mean()
        r["naive_perm_p05_rate_K50"] = (g.naive_p_of_winner_K50 < 0.05).mean()
        summ.append(r)
    pd.DataFrame(summ).round(3).to_csv(OUT + "selection.csv", index=False)


if __name__ == "__main__":
    which, reps = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 300
    {"A": exp_A, "B": exp_B, "C": exp_C, "D": exp_D}[which](reps)
