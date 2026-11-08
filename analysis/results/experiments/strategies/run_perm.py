"""Permutation tests for the final candidates (n=200), plus a null for each candidate's gain over v1.

    uv run --project /Users/aryan/Projects/hacktheburgh/keyboard-fatigue-nobaseline/analysis python run_perm.py [n_variants]

Per model (one worker process each, 3 at a time):
1. harness.perm_test(d, fn, "lopo", "skill", n=200): the shared test, unchanged.
2. A replay of exactly the same within-person KSS shuffles (seed 0, same transform call as the harness),
   evaluating the model on both schemes per shuffle. Its LOPO-skill p must equal step 1 (asserted).
   Because every worker replays identical shuffles, candidate-minus-v1 per shuffle gives a null for the
   gain over v1: does the gain need the real within-person ordering of KSS, or would it appear anyway
   (for example from person-level calibration, which within-person shuffles leave intact)?
Writes perm_results.json and appends one record per candidate to ../results.jsonl via harness.log_result.
"""

import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
N = 200
STATS = ("MAE", "skill", "within_concordance")
NAMES = ["v1", "H1 v1 + baseline rows (anchored ridge)", "C2 ordinal on kss_delta, alpha tuned", "G1 mean(v1, v3)",
         "R2a ordinal on kss_delta + baseline rows", "R2e pairwise + baseline rows, unanchored",
         "B7 within-person ridge, unanchored"]


def build(name):
    import common as C
    from common import lin, mean_of, ordinal, pairwise, tuned, within_ridge
    from run_round2 import ordinal_aug
    from run_variants import ORD_A
    return {
        "v1": lambda: C.H.BASELINES["v1"],
        NAMES[1]: lambda: lin(a=1.0, aug=True),
        NAMES[2]: lambda: tuned(lambda al: ordinal(alpha=al, a=1.0), ORD_A),
        NAMES[3]: lambda: mean_of(C.H.BASELINES["v1"], C.H.v3),
        NAMES[4]: lambda: tuned(lambda al: ordinal_aug(al), ORD_A),
        NAMES[5]: lambda: pairwise(a=0.0, aug=True),
        NAMES[6]: lambda: within_ridge(a=0.0),
    }[name]()


def worker(name):
    import common as C
    t, d, B = C.load()
    fn = build(name)
    t0 = time.time()
    obs_h, p_h, _ = C.H.perm_test(d, fn, scheme="lopo", stat="skill", n=N)
    observed = C.H.evaluate(d, fn)
    null = {s: {k: [] for k in STATS} for s in ("lopo", "rolling")}
    rng = np.random.default_rng(0)
    for _ in range(N):
        dd = d.copy()
        dd["kss"] = dd.groupby("pid").kss.transform(lambda s: rng.permutation(s.to_numpy()))
        dd["kss_delta"] = dd.kss - dd.kss_base
        res = C.H.evaluate(dd, fn)
        for s in res:
            for k in STATS:
                null[s][k].append(res[s][k])
    print(f"  done {name} in {time.time() - t0:.0f}s (harness LOPO skill p {p_h:.3f})", flush=True)
    return name, {"harness_lopo_skill": {"observed": obs_h, "p": p_h}, "observed": observed, "null": null}


def pval(nv, ob, lower_better):
    nv = np.asarray(nv, float)
    return float(((np.sum(nv <= ob) if lower_better else np.sum(nv >= ob)) + 1) / (len(nv) + 1))


def main():
    import common as C
    n_variants = int(sys.argv[1]) if len(sys.argv) > 1 else 46
    with get_context("spawn").Pool(3) as pool:
        got = dict(pool.map(worker, NAMES))
    out = {}
    for name in NAMES:
        g = got[name]
        rec = {"harness_lopo_skill": g["harness_lopo_skill"], "observed": g["observed"], "p": {}, "gain_vs_v1": {}}
        for s in ("lopo", "rolling"):
            for k in STATS:
                ob, nv = g["observed"][s][k], g["null"][s][k]
                rec["p"][f"{s}:{k}"] = pval(nv, ob, k == "MAE")
                if name != "v1":
                    g_obs = ob - got["v1"]["observed"][s][k]
                    g_null = np.asarray(nv, float) - np.asarray(got["v1"]["null"][s][k], float)
                    rec["gain_vs_v1"][f"{s}:{k}"] = {"observed_gain": float(g_obs),
                                                     "null_mean_gain": float(np.nanmean(g_null)),
                                                     "p": pval(g_null, g_obs, k == "MAE")}
        assert abs(rec["p"]["lopo:skill"] - rec["harness_lopo_skill"]["p"]) < 1e-9, name
        out[name] = rec
    (HERE / "perm_results.json").write_text(json.dumps(out, indent=1, default=float))

    for name, rec in out.items():
        print(f"\n{name}")
        for key, p in rec["p"].items():
            s, k = key.split(":")
            line = f"  {key:28s} obs {rec['observed'][s][k]:.3f}  p {p:.3f}"
            if name != "v1":
                gg = rec["gain_vs_v1"][key]
                line += f"   gain vs v1 {gg['observed_gain']:+.3f} (null mean {gg['null_mean_gain']:+.3f}, p {gg['p']:.3f})"
            print(line)
        if name != "v1":
            C.H.log_result(track="strategies", name=name, n_variants=n_variants,
                           results=rec["observed"] | {"perm": {"p": rec["p"], "gain_vs_v1": rec["gain_vs_v1"],
                                                               "harness_lopo_skill_p": rec["harness_lopo_skill"]["p"],
                                                               "n_perm": N}},
                           notes="strategies track candidate. p: within-person KSS shuffles (harness scheme, seed 0, "
                                 "n=200). gain_vs_v1 p: share of the same shuffles where the candidate beat v1 by at "
                                 "least as much. Baseline-row variants add the panel's baseline check-ins (training "
                                 "people only in LOPO; past ones in rolling) as extra training rows.")


if __name__ == "__main__":
    main()
