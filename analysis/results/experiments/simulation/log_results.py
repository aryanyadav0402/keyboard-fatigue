"""Append the headline numbers to ../results.jsonl via harness.log_result(track='simulation', ...)."""
import sys; sys.path.insert(0, "/Users/aryan/Projects/hacktheburgh/keyboard-fatigue-nobaseline/analysis/results/experiments")
import pandas as pd, harness as H
rd = lambda f: pd.read_csv(f, keep_default_na=False)
g = rd("power_grid.csv"); r = g[(g.design == "real(7,5,6,6)")].set_index("rho")
H.log_result("simulation", "power_real_design", 1, {
    "null_v1_skill_q95": float(r.null_thr95.iloc[0]), "P_v1_skill_gt0_null": float(r.loc[0.0, "P_skill_gt0"]),
    "P_v1_clears_null95_rho0.4": float(r.loc[0.4, "P_skill_gt_thr"]), "P_v1_beats_guess_MAE_rho0.4": float(r.loc[0.4, "P_MAE_beats_guess"]),
    "dMAE_v1_v3_sd_rho0.4": float(r.loc[0.4, "dMAE_v1_v3_sd"])},
    "feature-level sim, 4 people 24 scored, LOPO; rho=0.4 reproduces real v1 skill ~0.22; 1500 worlds/cell")
x = g[(g.rho == 0.4)].set_index("design")
H.log_result("simulation", "power_80pct_designs", 1, {d: float(x.loc[d, "P_skill_gt_thr"]) for d in ["real(7,5,6,6)", "4x15", "6x10", "6x15", "8x10", "12x6", "20x4"]},
             "P(v1 LOPO skill exceeds the design's null 95th percentile) at rho=0.4")
p = rd("practice.csv"); p = p[p.design == "real(7,5,6,6)"].set_index("world")
H.log_result("simulation", "practice_confound", 1, {w: {"P_skill_gt0": float(p.loc[w, "P_skill_gt0"]), "P_conc_ge_0.65": float(p.loc[w, "P_conc_ge_0p65"]),
             "P_perm_p05": float(p.loc[w, "P_perm_p05"]), "P_looks_real": float(p.loc[w, "P_looks_real"])} for w in ["null", "practice .5", "practice .7", "fatigue .4"]},
             "practice-only world, real design; looks_real = skill>0 & conc>=0.65 & perm p<0.05 (200 shuffles)")
m = rd("morning_by_sleep_share.csv")
H.log_result("simulation", "morning_frozen_by_sleep_share", 1, {f"{a}_{b}": float(c) for a, b, c in zip(m.share_slept, m.world, m.P_v1_beats_guess)},
             "P(frozen v1 beats guess on 15 morning check-ins); depends on whether people sleep")
s = rd("selection_same_family.csv").set_index("world").loc["null"]
H.log_result("simulation", "selection_bias_threshold", 50, {"best_skill_q95_K5": float(s.K5_best_skill_q95), "K20": float(s.K20_best_skill_q95), "K50": float(s.K50_best_skill_q95),
             "MAEgain_over_v1_q95_K5": float(s.K5_MAEgain_over_v1_q95), "K20": float(s.K20_MAEgain_over_v1_q95), "K50": float(s.K50_MAEgain_over_v1_q95)},
             "null world, 4 people 24 scored; best-of-K anchored ridge variants by LOPO skill; a win needs to beat these")
