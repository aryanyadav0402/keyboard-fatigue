"""Descriptive only (post hoc, not used to pick anything): how each single-attempt feature moves
with KSS pooled, within a person, across people at the same check-in number, and with check-in
number itself (practice). Full-panel rule references. Writes directions.json.

    uv run --project <analysis> python describe_directions.py
"""

import json

import numpy as np
import pandas as pd

from common import FG5, HERE, PRIOR, load, matched_deviations, rule_adjust, spearman

COLS = FG5 + ["log_wpm_adj", "sqrt_bsp_adj", "vt_hold_med_adj", "log_rt"]


def main():
    panel, _ = load()
    d = rule_adjust(panel, panel)
    y, n = d.kss.to_numpy(float), d.checkin_n.to_numpy()
    multi = d.groupby("pid").kss.transform("size").ge(2).to_numpy()
    yc = y - d.groupby("pid").kss.transform("mean").to_numpy()
    dy, _ = matched_deviations(y, n)
    rows = {}
    for c in COLS:
        x = d[c].to_numpy(float)
        xc = x - d.groupby("pid")[c].transform("mean").to_numpy()
        rows[c] = {"prior_direction": PRIOR.get(c, 0) if c in PRIOR else None,
                   "pooled": spearman(x, y),
                   "within_person": spearman(xc[multi], yc[multi]),
                   "same_checkin_between": spearman(matched_deviations(x, n)[0], dy),
                   "with_checkin_n_within_person": spearman(xc[multi], (n - d.groupby("pid").checkin_n.transform("mean").to_numpy())[multi])}
    t = pd.DataFrame(rows).T
    print(t.round(2).to_string())
    (HERE / "directions.json").write_text(json.dumps(rows, indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
