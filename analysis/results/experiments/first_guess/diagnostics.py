"""Post-hoc checks on the selected candidate (ridge_fg5). Nothing here is eligible to be frozen; the
sensitivity runs are counted in the README (POST_HOC).

    uv run --project <analysis> python diagnostics.py

1. per-person error against the flat guess;
2. the fitted weights in every fold and on the whole panel (what would be frozen);
3. the panel without P5 (one check-in, the lowest KSS) for training and testing;
4. fixed alphas instead of inner-LOPO tuning, and leaving each feature out;
5. practice: do predictions climb with check-in number inside a person, and does the ranking
   survive once check-in number is partialled out of both sides (nuisance covariate only);
6. first attempts only (5 panel first check-ins out of fold + 3 walk-ups from the whole-panel model):
   the closest thing to a stranger's single try.
Writes diagnostics.json.
"""

import json

import numpy as np
import pandas as pd
from scipy import stats

from candidates import CANDIDATES, flat, inner_lopo, ridge_predict
from common import FG5, HERE, load, make_folds, metrics, pair_masks, rule_adjust, run, spearman

POST_HOC = ["ridge_fg5 alpha=1", "ridge_fg5 alpha=10", "ridge_fg5 alpha=100"] + [f"ridge_fg5 without {c}" for c in FG5]


def fixed_ridge(cols, alpha):
    return lambda train, y, test: (ridge_predict(train[cols].to_numpy(float), y, test[cols].to_numpy(float), alpha), None)


def weights(train, y, cols, alpha):
    X = train[cols].to_numpy(float)
    X = np.where(np.isnan(X), np.nanmedian(X, 0), X)
    A = (X - X.mean(0)) / np.where(X.std(0) > 1e-12, X.std(0), 1.0)
    return np.linalg.solve(A.T @ A + alpha * np.eye(len(cols)), A.T @ (y - y.mean()))


def chosen_alpha(train, y, cols):
    grid = list(np.logspace(-2, 3, 11))
    X, pid = train[cols].to_numpy(float), train.pid.to_numpy()
    errs = [sum(np.sum((ridge_predict(X[pid != p], y[pid != p], X[pid == p], g) - y[pid == p]) ** 2)
                for p in np.unique(pid)) for g in grid]
    return grid[int(np.argmin(errs))]


def partial_spearman(a, b, z):
    ra, rb, rz = (stats.rankdata(v) for v in (a, b, z))
    res = [v - np.polyval(np.polyfit(rz, v, 1), rz) for v in (ra, rb)]
    return float(stats.pearsonr(*res).statistic)


def main():
    panel, walk = load()
    y = panel.kss.to_numpy(float)
    folds, masks = make_folds(panel), pair_masks(panel)
    fn = CANDIDATES["ridge_fg5"][0]
    pred, idx, fl = run(fn, folds, y, len(panel))
    out = {}

    out["per_person_MAE"] = {p: {"ridge_fg5": round(float(np.mean(np.abs(pred[m] - y[m]))), 2),
                                 "flat": round(float(np.mean(np.abs(fl[m] - y[m]))), 2), "n": int(m.sum())}
                             for p in sorted(panel.pid.unique()) for m in [panel.pid.eq(p).to_numpy()]}

    fold_w = {}
    for te, train, test in folds:
        a = chosen_alpha(train, y[~te], FG5)
        fold_w[panel.pid[te].iloc[0]] = {"alpha": round(a, 3), **{c: round(float(w), 3)
                                         for c, w in zip(FG5, weights(train, y[~te], FG5, a))}}
    full = rule_adjust(panel, panel)
    a = chosen_alpha(full, y, FG5)
    fold_w["whole panel (would be frozen)"] = {"alpha": round(a, 3), **{c: round(float(w), 3)
                                                for c, w in zip(FG5, weights(full, y, FG5, a))}}
    out["weights_per_standardised_feature"] = fold_w

    no5 = panel[panel.pid.ne("P5")].reset_index(drop=True)
    y5, f5, m5 = no5.kss.to_numpy(float), make_folds(no5), pair_masks(no5)
    out["without_P5"] = {name: metrics(no5, y5, *run(f, f5, y5, len(no5)), m5)
                         for name, f in {"flat": flat, "ridge_fg5": fn}.items()}

    sens = {f"ridge_fg5 alpha={a_}": fixed_ridge(FG5, a_) for a_ in (1, 10, 100)}
    sens |= {f"ridge_fg5 without {c}": inner_lopo(ridge_predict, list(np.logspace(-2, 3, 11)))([x for x in FG5 if x != c])
             for c in FG5}
    out["sensitivity"] = {k: {s: round(v, 3) for s, v in metrics(panel, y, *run(f, folds, y, len(panel)), masks).items()
                              if s in ("MAE", "MAE_flat", "skill", "spearman", "conc_matched")}
                          for k, f in sens.items()}

    n = panel.checkin_n.to_numpy(float)
    within = [spearman(pred[m], n[m]) for p in panel.pid.unique() for m in [panel.pid.eq(p).to_numpy()] if m.sum() >= 4]
    within_kss = [spearman(y[m], n[m]) for p in panel.pid.unique() for m in [panel.pid.eq(p).to_numpy()] if m.sum() >= 4]
    out["practice"] = {"median_within_person_rho(pred, checkin_n)": float(np.median(within)),
                       "median_within_person_rho(kss, checkin_n)": float(np.median(within_kss)),
                       "spearman(pred, kss)": spearman(pred, y),
                       "partial_spearman(pred, kss | checkin_n)": partial_spearman(pred, y, n),
                       "prediction_range": [round(float(pred.min()), 2), round(float(pred.max()), 2)]}

    def by_phase(frame, yy, p_, f_):
        """Unpractised check-ins (the first two, like a stranger's try) against practised ones."""
        out_ = {}
        for label, m in {"checkins_1_2": frame.checkin_n.le(2).to_numpy(), "checkins_3_plus": frame.checkin_n.ge(3).to_numpy()}.items():
            out_[label] = {"n": int(m.sum()), "MAE_ridge_fg5": float(np.mean(np.abs(p_[m] - yy[m]))),
                           "MAE_flat": float(np.mean(np.abs(f_[m] - yy[m]))), "spearman_ridge_fg5": spearman(p_[m], yy[m])}
        return out_
    p5, _, fl5 = run(fn, f5, y5, len(no5))
    out["by_phase"] = {"all_panel": by_phase(panel, y, pred, fl), "without_P5": by_phase(no5, y5, p5, fl5)}
    w5, _ = fn(rule_adjust(no5, no5), y5, rule_adjust(no5, walk))
    out["walkups_without_P5_in_training"] = {"pred": np.round(np.clip(w5, 1, 9), 2).tolist(), "kss": walk.kss.tolist(),
                                             "MAE": float(np.mean(np.abs(np.clip(w5, 1, 9) - walk.kss))),
                                             "MAE_flat": float(np.mean(np.abs(y5.mean() - walk.kss)))}

    first = panel.checkin_n.eq(1).to_numpy()
    w_pred, _ = fn(rule_adjust(panel, panel), y, rule_adjust(panel, walk))
    fa = pd.DataFrame({"who": list(panel.pid[first]) + list(walk.pid), "kss": np.r_[y[first], walk.kss],
                       "ridge_fg5": np.r_[pred[first], np.clip(w_pred, 1, 9)],
                       "flat": np.r_[fl[first], np.full(len(walk), y.mean())]})
    out["first_attempts"] = {"rows": fa.round(2).to_dict("records"),
                             "spearman": spearman(fa.ridge_fg5, fa.kss),
                             "MAE": float(np.mean(np.abs(fa.ridge_fg5 - fa.kss))),
                             "MAE_flat": float(np.mean(np.abs(fa.flat - fa.kss)))}
    (HERE / "diagnostics.json").write_text(json.dumps(out, indent=2, default=float) + "\n")
    for k, v in out.items():
        print(k, json.dumps(v, default=lambda x: round(float(x), 3)))


if __name__ == "__main__":
    main()
