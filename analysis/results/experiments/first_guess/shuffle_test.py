"""Label-shuffle tests for the selected first-guess candidate (and a selection-aware version).

    uv run --project <analysis> python shuffle_test.py [n]

Each shuffle permutes KSS and re-runs the whole procedure, in-fold tuning included. Three shuffles:
- across: KSS permuted across all 33 panel check-ins (the required test). Kills everything, so a
  model that only tracks practice over the night still passes it.
- same_checkin: KSS permuted only among check-ins with the same check-in number (same time of night,
  same amount of practice). Keeps the night's drift; kills who-was-more-tired-at-that-moment, which
  is what a stranger's single attempt has to get right.
- within_person: KSS permuted within each person (the harness's perm_test); kills the night's order
  but keeps each person's average.
'selection-aware' also re-runs every eligible variant on each across-shuffle and compares the
selected candidate's Spearman with the null distribution of the best of them, which pays for having
picked the best of n_variants. Writes shuffle_test.json.
"""

import json
import sys
import time

import numpy as np

from candidates import CANDIDATES, ELIGIBLE
from common import HERE, load, make_folds, metrics, pair_masks, run

STATS = ["spearman", "skill", "MAE", "conc_between", "conc_matched", "first_MAE"]


def shuffled(y, groups, rng):
    out = y.copy()
    for g in np.unique(groups):
        m = np.flatnonzero(groups == g)
        out[m] = y[rng.permutation(m)]
    return out


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    chosen = json.loads((HERE / "selected.json").read_text())["selected"]
    panel, _ = load()
    folds, masks, y = make_folds(panel), pair_masks(panel), panel.kss.to_numpy(float)
    fn = CANDIDATES[chosen][0]

    def score(fn_, yy):
        return metrics(panel, yy, *run(fn_, folds, yy, len(panel)), masks)

    observed = score(fn, y)
    observed_all = {k: score(CANDIDATES[k][0], y)["spearman"] for k in ELIGIBLE}
    schemes = {"across": np.zeros(len(y)), "same_checkin": panel.checkin_n.to_numpy(),
               "within_person": panel.pid.to_numpy()}
    out = {"selected": chosen, "n": n, "observed": observed, "tests": {}}
    t0 = time.time()
    for s_i, (scheme, groups) in enumerate(schemes.items()):
        rng = np.random.default_rng(s_i)
        null = {k: [] for k in STATS}
        best_null = []
        for _ in range(n):
            yy = shuffled(y, groups, rng)
            res = score(fn, yy)
            for k in STATS:
                null[k].append(res[k])
            if scheme == "across":
                rhos = [score(CANDIDATES[k][0], yy)["spearman"] for k in ELIGIBLE]
                best_null.append(np.nanmax(np.where(np.isnan(rhos), -np.inf, rhos)))
        test = {}
        for k in STATS:
            v = np.asarray(null[k], float)
            v = v[~np.isnan(v)]
            lower_better = k in ("MAE", "first_MAE")
            hits = np.sum(v <= observed[k]) if lower_better else np.sum(v >= observed[k])
            test[k] = {"observed": observed[k], "null_median": float(np.median(v)),
                       "null_95": float(np.quantile(v, 0.05 if lower_better else 0.95)),
                       "p": float((hits + 1) / (len(v) + 1)), "n_valid": int(len(v))}
        if best_null:
            b = np.asarray(best_null)
            test["selection_aware_spearman"] = {
                "observed_best_eligible": float(np.nanmax(list(observed_all.values()))),
                "null_median_of_best": float(np.median(b)), "null_95_of_best": float(np.quantile(b, 0.95)),
                "p": float((np.sum(b >= observed["spearman"]) + 1) / (len(b) + 1))}
        out["tests"][scheme] = test
        print(f"{scheme}: done in {time.time() - t0:.0f}s")
        for k, v in test.items():
            print(f"  {k}: " + ", ".join(f"{a}={b:.3f}" if isinstance(b, float) else f"{a}={b}" for a, b in v.items()))
    (HERE / "shuffle_test.json").write_text(json.dumps(out, indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
