"""Does any model order a person's check-ins better than 'later = more tired'?

Splits the within-person pairs (different KSS) into those where the later check-in was more tired
(time-concordant) and those where it was less tired (time-discordant). A model that only tracks the
night's trend (practice, tiredness, or both) gets the first kind right and the second kind wrong.
Also reports the median within-person Spearman between each model's predictions and the clock.
Diagnostic only: no new models, uses the out-of-fold predictions of existing ones.

    uv run --project /Users/aryan/Projects/hacktheburgh/keyboard-fatigue-nobaseline/analysis python beyond_clock.py
"""

import numpy as np
import pandas as pd
from scipy import stats

import common as C
from common import lin, mean_of, ordinal, pairwise, tuned, within_ridge
from run_round2 import ordinal_aug
from run_variants import ORD_A


def split_conc(d, pred):
    out = {"time_concordant": [0.0, 0], "time_discordant": [0.0, 0]}
    m = ~np.isnan(pred)
    sub = d[m].assign(pred=pred[m])
    for _, g in sub.groupby("pid"):
        k, q, h = g.kss.to_numpy(float), g.pred.to_numpy(float), g.hours_rel.to_numpy(float)
        for i in range(len(k)):
            for j in range(i + 1, len(k)):
                if k[i] == k[j]:
                    continue
                kind = "time_concordant" if np.sign(h[i] - h[j]) == np.sign(k[i] - k[j]) else "time_discordant"
                s = np.sign(q[i] - q[j]) * np.sign(k[i] - k[j])
                out[kind][0] += 1.0 if s > 0 else 0.5 if s == 0 else 0.0
                out[kind][1] += 1
    rho = np.nanmedian([stats.spearmanr(g.pred, g.hours_rel).statistic for _, g in sub.groupby("pid")
                        if len(g) >= 3 and np.ptp(g.pred) > 0])
    return {f"{k} ({v[1]} pairs)": v[0] / v[1] if v[1] else np.nan for k, v in out.items()} | {"rho_with_clock": rho}


def main():
    t, d, B = C.load()
    models = {"clock": C.H.BASELINES["clock"], "v1": C.H.BASELINES["v1"], "v3": C.H.v3,
              "H1 v1 + baseline rows": lin(a=1.0, aug=True),
              "C2 ordinal kss_delta": tuned(lambda al: ordinal(alpha=al, a=1.0), ORD_A),
              "G1 mean(v1, v3)": mean_of(C.H.BASELINES["v1"], C.H.v3),
              "R2a ordinal + baseline rows": tuned(lambda al: ordinal_aug(al), ORD_A),
              "R2e pairwise + baseline rows, unanch": pairwise(a=0.0, aug=True),
              "B7 within-person ridge, unanch": within_ridge(a=0.0),
              "F2 composite label, unanch": C.composite(a=0.0)}
    pd.set_option("display.width", 250)
    for scheme in ("lopo", "rolling"):
        rows = {name: split_conc(d, C.H.predict(d, fn, scheme)[0]) for name, fn in models.items()}
        print(f"== {scheme} ==")
        print(pd.DataFrame(rows).T.round(3).to_string())


if __name__ == "__main__":
    main()
