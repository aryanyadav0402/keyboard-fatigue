"""Run every registered first-guess variant under the stranger evaluation and write the results.

    uv run --project <analysis> --with tabulate python run_candidates.py [--log]

--log appends one record per variant to the shared results.jsonl (do it once, on the final run).
Writes results.csv, results.md, predictions.csv (pid, check-in number, KSS and each variant's
prediction; no names, no times) and selected.json (the candidate the pre-set rule picks).
"""

import json
import sys

import numpy as np
import pandas as pd

from candidates import CANDIDATES, ELIGIBLE, N_VARIANTS, flat
from common import HERE, TRACK, load, log_result, make_folds, metrics, pair_masks, run, walkup_check

ORDER = ["MAE", "MAE_flat", "skill", "spearman", "conc_between", "conc_matched", "conc_within",
         "idx_spearman", "idx_conc_between", "idx_conc_matched", "idx_conc_within",
         "first_MAE", "first_MAE_flat", "first_bias", "rho_pred_checkin", "walk_MAE", "walk_MAE_flat", "walk_pred"]


def evaluate_all(panel, walk):
    folds, folds_full = make_folds(panel), make_folds(panel, full_refs=True)
    masks, y = pair_masks(panel), panel.kss.to_numpy(float)
    rows, preds = {}, {"pid": panel.pid, "checkin_n": panel.checkin_n, "kss": y}
    for name, (fn, honesty, desc) in ({"flat (reference)": (flat, "reference", "training mean")} | CANDIDATES).items():
        pred, idx, fl = run(fn, folds_full if name.endswith("fullrefs") else folds, y, len(panel))
        res = metrics(panel, y, pred, idx, fl, masks)
        walk_res = walkup_check(fn, panel, walk)
        rows[name] = {"honesty": honesty, **res, **walk_res, "description": desc}
        preds[name] = np.round(pred, 3)
    return pd.DataFrame(rows).T, pd.DataFrame(preds)


def select(table: pd.DataFrame) -> str:
    """Eligible candidate with the highest pooled Spearman; ties broken by skill."""
    t = table.loc[ELIGIBLE].copy()
    t = t[t.spearman.astype(float).notna()]
    return t.sort_values(["spearman", "skill"], ascending=False, key=lambda s: s.astype(float)).index[0]


def main():
    panel, walk = load()
    table, preds = evaluate_all(panel, walk)
    table.to_csv(HERE / "results.csv")
    preds.to_csv(HERE / "predictions.csv", index=False)
    show = table[["honesty"] + ORDER].copy()
    for c in ORDER[:-1]:
        show[c] = show[c].astype(float).round(3)
    (HERE / "results.md").write_text(show.to_markdown() + "\n")
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(show.drop(columns=["walk_pred"]).to_string())
    chosen = select(table)
    (HERE / "selected.json").write_text(json.dumps({"selected": chosen, "rule": select.__doc__,
                                                    "n_variants": N_VARIANTS}, indent=2) + "\n")
    print(f"\nn_variants = {N_VARIANTS}; selected by the pre-set rule: {chosen}")
    if "--log" in sys.argv:
        shuffle = HERE / "shuffle_test.json"
        shuffle = json.loads(shuffle.read_text()) if shuffle.exists() else None
        for name in CANDIDATES:
            r = table.loc[name]
            res = {k: (float(r[k]) if k != "walk_pred" else r[k]) for k in ORDER if k == "walk_pred" or pd.notna(r[k])}
            results = {"stranger_lopo": res}
            if shuffle and shuffle["selected"] == name:
                results["shuffle_p"] = {scheme: {k: v["p"] for k, v in tests.items()}
                                        for scheme, tests in shuffle["tests"].items()}
            log_result(TRACK, name, N_VARIANTS, results,
                       notes=f"{r['honesty']}: {r['description']}. Every panel check-in incl. baselines predicted "
                             f"leave-one-person-out from single-attempt features; rule references refitted in fold."
                             + (" Selected by the pre-set rule; shuffle tests n=200." if name == chosen else ""))
        print(f"logged {len(CANDIDATES)} records to results.jsonl")


if __name__ == "__main__":
    main()
