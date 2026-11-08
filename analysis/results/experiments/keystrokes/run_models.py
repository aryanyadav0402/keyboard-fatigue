"""Do keystroke-level features add anything to v1? Four candidates, fixed before any label was looked at.

    uv run --project ../../.. python run_models.py

The candidates were chosen from theory and from reliability.csv only (no KSS involved):
  C1 v1, backspaces -> errors made  backspaces are errors made x share fixed + other deletions; nearly every
                                     error is fixed here and spurious deletions are ~0, so "errors made"
                                     (from the buffer replay) is the cleaner count of the same thing
  C2 v1 + rhythm distance            Wasserstein distance between this check-in's log ordinary gaps and the
                                     person's baseline ones: the most reliable new feature (split-half 0.66)
  C3 v1 reliable only                drop the v1 features whose within-person split-half reliability is
                                     below 0.3 on either split or without the one outlier check-in:
                                     rule cost, accuracy, gap CV
  C4 C3 + rhythm distance
Rule for keeping a v1 feature (C3): Spearman-Brown within-person split-half >= 0.3 for first/second half,
for odd/even words, and for first/second half without P1's first baseline check-in.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import harness  # noqa: E402
from harness import TYPING, anchored, ridge  # noqa: E402

TRACK = "keystrokes"
UNRELIABLE_V1 = ["vt_rule_cost_dev", "vt_accuracy_dev", "vt_gap_cv_dev"]
V1_RELIABLE = [c for c in TYPING if c not in UNRELIABLE_V1]
CANDIDATES = {
    "C1 v1 bs->errors made": [c if c != "backspace_per100_adj" else "ks_err_per100_dev" for c in TYPING],
    "C2 v1 + rhythm distance": TYPING + ["ks_w1_base_dev"],
    "C3 v1 reliable only": V1_RELIABLE,
    "C4 v1 reliable + rhythm distance": V1_RELIABLE + ["ks_w1_base_dev"],
}
N_VARIANTS = len(CANDIDATES)
N_PERM = 200


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    t = harness.load_table()
    ks = pd.read_csv(HERE / "keystroke_features.csv")
    new = [c for c in ks if c.startswith("ks_")]
    t = t.merge(ks[["checkin_id"] + new], on="checkin_id", how="left")
    return t, harness.scored_panel(t)


def coefficients(d: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Standardised ridge weights on all 24 scored check-ins (for reading the signs only)."""
    m = ridge().fit(d[cols].to_numpy(float), d.kss_delta.to_numpy(float))
    return pd.Series(m[-1].coef_, index=cols).round(3)


def main() -> None:
    t0 = time.time()
    t, d = load()
    pd.set_option("display.width", 250)
    print(f"{len(d)} scored check-ins; missing values in candidate columns:")
    cols = sorted({c for v in CANDIDATES.values() for c in v})
    print(d[cols].isna().sum().to_string(), "\n")

    fns = {name: anchored(ridge, c) for name, c in CANDIDATES.items()}
    table = harness.compare(d, fns)
    print(table.to_string(), "\n")

    results = {}
    for name, fn in fns.items():
        res = harness.evaluate(d, fn)
        obs_l, p_l, _ = harness.perm_test(d, fn, "lopo", "skill", N_PERM)
        obs_r, p_r, _ = harness.perm_test(d, fn, "rolling", "skill", N_PERM)
        res["perm"] = {"lopo_skill": obs_l, "lopo_p": p_l, "rolling_skill": obs_r, "rolling_p": p_r, "n": N_PERM}
        results[name] = res
        print(f"{name:34s} perm (within-person KSS shuffle, n={N_PERM}): lopo skill {obs_l:.3f} p={p_l:.3f} | "
              f"rolling skill {obs_r:.3f} p={p_r:.3f}")
    # v1 under the same perm test, for reference (not a variant)
    v1 = anchored(ridge, TYPING)
    v1_res = harness.evaluate(d, v1)
    o1, p1, _ = harness.perm_test(d, v1, "lopo", "skill", N_PERM)
    o2, p2, _ = harness.perm_test(d, v1, "rolling", "skill", N_PERM)
    v1_res["perm"] = {"lopo_skill": o1, "lopo_p": p1, "rolling_skill": o2, "rolling_p": p2, "n": N_PERM}
    print(f"{'v1 (reference)':34s} perm (within-person KSS shuffle, n={N_PERM}): lopo skill {o1:.3f} p={p1:.3f} | "
          f"rolling skill {o2:.3f} p={p2:.3f}\n")

    print("Standardised weights on all 24 (signs only; + means predicts more tired):")
    for name, c in {"v1": TYPING, **CANDIDATES}.items():
        print(f"  {name}: {coefficients(d, c).to_dict()}")

    for name, res in results.items():
        harness.log_result(TRACK, name, N_VARIANTS, res,
                           notes=f"features {CANDIDATES[name]}; candidates fixed from theory + split-half "
                                 f"reliability before labels; anchored ridge as v1; perm n={N_PERM}")
    out = {"table": table.reset_index().rename(columns={"index": "model"}).to_dict("records"),
           "results": results, "v1": v1_res, "candidates": CANDIDATES, "n_variants": N_VARIANTS}
    (HERE / "model_results.json").write_text(json.dumps(out, indent=1, default=float))
    table.to_csv(HERE / "model_table.csv")
    print(f"\n{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
