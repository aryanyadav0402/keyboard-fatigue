# First guess for strangers

Track `first_guess`, 8 Nov, 02:00–02:30. Data: `harness.load_table()`, the 36 check-ins up to 00:53:51. That's 33 panel check-ins from P1–P5 (P5 has only one, a baseline check-in rated KSS 2) and 3 walk-ups (W1–W3). No names appear anywhere in this folder.

**Question.** Can any honest single-attempt model rank tiredness across people better than the flat guess? The shipped no-baseline model (`analysis/first_guess.py`) fits a slope of 0, so every stranger gets the team average.

**Short answer.** One model does: `ridge_fg5`, which picks its own direction for each feature inside the training folds. It beats flat on the panel, and it passes every shuffle test. But nearly all of its gain comes from practised check-ins. On unpractised attempts, which are the only kind a stranger makes, it is no better than flat, and that result changes when one person's single check-in is dropped. It also learns that smoother, more accurate typing with fewer corrections means more tired. That reverses the expected direction for 4 of the 5 features, and it is also what practice looks like. Recommendation: on a stranger's first try, show no tiredness number. Freeze `ridge_fg5` only so the morning batch can test it.

## How the evaluation works

`common.py` runs each fold the same way:

- **Leave one person out.** Every panel check-in is predicted, including baseline check-ins (a stranger has no baseline). The model is trained on the other panel people only.
- **Single-attempt inputs only.** No `_dev` features, no `kss_base`, no clock and no check-in number. Check-in number appears only as a nuisance covariate, in evaluation and in two training-only contrasts that are labelled as such.
- **Rule references refitted in each fold.** The per-rule typical values behind `_adj` come from the training people only. The shipped references use every panel check-in, including the person being predicted. The row `fg5_prior_fullrefs` shows what that leak changes.
- **Tuning stays in the training folds.** Directions, feature drops, the ridge alpha and the kNN k are all tuned by inner leave-one-person-out over the training people.

Metrics, out of fold:

- **MAE** against the flat guess (the training mean), which scores 1.51.
- **skill** = 1 − SSE / SSE of the flat guess.
- **Pooled Spearman** over all 33 predictions. A constant guess has no ranking. The flat row shows −0.36 only because leaving a person out shifts the training mean against them.
- **conc_matched**: the share of pairs from different people at the same check-in number (same time of night, same practice) that the model orders correctly; 0.5 is chance. This is the closest test of the stranger question: who is more tired at that moment?
- **conc_within**: the same share for pairs from the same person.
- **idx_\***: the same rankings for the raw index, which is what a percentile display would show.
- **first_MAE**: MAE on the 5 first check-ins (flat: 1.84).
- **walk_MAE**: MAE on the 3 walk-ups, predicted by a model trained on the whole panel (flat: 1.12). Three rows are a sanity check, not a test.

The candidates were registered in `candidates.py` before any run, each with an honesty label:

| Label | Meaning |
|---|---|
| prior | Directions fixed before this data |
| fold | Learned inside the training folds |
| evidence | Direction picked from correlations already seen on tonight's full data; optimistic and not eligible |
| check | Not eligible |

Selection rule, set in advance: the eligible candidate with the highest pooled Spearman.

## Variance split: KSS is almost all within-person

From `variance_split.json`:

| Method | Between people | Within people |
|---|---|---|
| One-way ANOVA, 5 people, 33 check-ins | 18% (F test p = 0.07) | 82% |
| Same, 4 people with repeat check-ins (32) | 0.5% (p = 0.39) | 99.5% |
| REML random intercept, 33 check-ins | 2.5% | 97.5% |
| REML, allowing for clock (nuisance only) | 0.2% | 99.8% |
| REML, allowing for check-in number | 5.9% | 94.1% |

The 18% comes entirely from P5's single early check-in (KSS 2). The four people who checked in all night have nearly the same average: 5.6, 5.9, 6.75 and 6.75.

At the same check-in number, people do differ: the variance across people is 1.9, against 2.7 within a person. For example, at check-in 3 the ratings were 3, 3, 7, 3, and at check-in 7 they were 8, 8, 4, 8. But those differences don't persist, so they are momentary state or rating noise, not people using the scale differently.

The features are mostly within-person too: the between-person share is 1.5–16% for the five first-guess features. Speed (29%) and key-hold time (74%) are mostly who you are.

So scale use is not what blocks a first guess in our data. What blocks it is that typing changes over the night are tangled up with practice.

## Results: all 28 variants plus the flat reference

From `results.csv`. 22 variants were eligible. MAE is in KSS points.

| variant | label | MAE | skill | Spearman | conc_matched | conc_within | idx Spearman | idx conc_matched | first_MAE | walk_MAE |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| flat (reference) | reference | 1.51 | 0 | −0.36 | 0.31 | 0.50 | | | 1.84 | 1.12 |
| fg5_prior (shipped spec) | prior | 1.51 | 0 | −0.36 | 0.31 | 0.50 | **−0.41** | **0.28** | 1.84 | 1.12 |
| fg5_flip_bsp | evidence | 1.52 | −0.01 | −0.31 | 0.34 | 0.54 | −0.10 | 0.41 | 1.80 | 1.12 |
| fg5_flip_bsp_cv | evidence | 1.50 | −0.02 | 0.01 | 0.47 | 0.68 | 0.20 | 0.56 | 1.59 | 1.70 |
| fg4_no_bsp | evidence | 1.51 | 0 | −0.36 | 0.31 | 0.50 | −0.27 | 0.38 | 1.84 | 1.12 |
| fg5_drop_contra | fold | 1.50 | 0.01 | −0.23 | 0.44 | 0.55 | −0.09 | 0.50 | 1.75 | 1.12 |
| fg5_sign_pooled | fold | 1.46 | −0.08 | 0.36 | 0.69 | 0.66 | 0.32 | 0.69 | 2.02 | 0.71 |
| fg5_sign_within | fold | 1.43 | −0.02 | 0.13 | 0.56 | 0.63 | 0.20 | 0.62 | 1.76 | 2.01 |
| fg5_sign_between | fold | 1.52 | −0.09 | 0.20 | 0.59 | 0.57 | 0.19 | 0.59 | 1.61 | 0.97 |
| fg5_sign_matched | fold | 1.63 | −0.18 | 0.19 | 0.53 | 0.58 | 0.28 | 0.66 | 1.73 | 0.68 |
| fg5_prior_free_slope | fold | 1.46 | −0.04 | 0.34 | 0.69 | 0.59 | −0.41 | 0.28 | 2.11 | 0.71 |
| fg5_prior_matched_slope | fold | 1.51 | 0 | −0.36 | 0.31 | 0.50 | −0.41 | 0.28 | 1.84 | 1.12 |
| fg5_rank_prior | prior | 1.51 | 0 | −0.36 | 0.31 | 0.50 | −0.31 | 0.34 | 1.84 | 1.12 |
| fg5_rank_flip_bsp | evidence | 1.53 | −0.01 | −0.31 | 0.34 | 0.55 | 0.00 | 0.47 | 1.81 | 1.12 |
| fg5_rank_sign_pooled | fold | 1.42 | −0.09 | 0.28 | 0.69 | 0.63 | 0.25 | 0.69 | 1.98 | 1.07 |
| fg5_prior_quantile | prior | 2.51 | −2.07 | −0.34 | 0.30 | 0.44 | −0.41 | 0.28 | 2.45 | 3.35 |
| fg5_flip_bsp_quantile | evidence | 2.40 | −1.66 | −0.12 | 0.38 | 0.55 | −0.10 | 0.41 | 2.02 | 2.68 |
| ratio3_prior | prior | 1.51 | 0 | −0.36 | 0.31 | 0.50 | −0.14 | 0.34 | 1.84 | 1.12 |
| ratio3_sign_pooled | fold | 1.55 | −0.14 | −0.16 | 0.47 | 0.35 | −0.13 | 0.56 | 2.12 | 0.77 |
| swell_raw | prior | 1.56 | −0.12 | 0.20 | 0.56 | 0.74 | 0.37 | 0.62 | 1.56 | 1.18 |
| swell_raw_free | fold | 1.56 | −0.12 | 0.20 | 0.56 | 0.74 | 0.37 | 0.62 | 1.56 | 1.18 |
| swell_adj | prior | 1.63 | −0.28 | 0.15 | 0.56 | 0.69 | 0.34 | 0.62 | 1.56 | 1.11 |
| rt_prior | prior | 1.55 | −0.11 | −0.33 | 0.31 | 0.52 | 0.16 | 0.55 | 2.05 | 1.15 |
| fg5_rt_prior | prior | 1.51 | 0 | −0.36 | 0.31 | 0.50 | −0.33 | 0.41 | 1.84 | 1.12 |
| **ridge_fg5** | fold | **1.13** | **0.36** | **0.52** | **0.84** | 0.71 | | | 1.16 | 1.47 |
| ridge_wide | fold | 1.79 | −0.31 | 0.12 | 0.62 | 0.70 | | | 2.09 | 1.02 |
| knn_fg5 | fold | 1.29 | 0.15 | 0.42 | 0.59 | 0.72 | | | 2.18 | 1.42 |
| knn_wide | fold | 1.42 | 0.04 | 0.22 | 0.55 | 0.75 | | | 1.91 | 1.41 |
| fg5_prior_fullrefs | check | 1.51 | 0 | −0.36 | 0.31 | 0.50 | −0.54 | 0.28 | 1.84 | 1.12 |

What the table shows:

- **The shipped index ranks tiredness backwards.** Its slope is 0 in every fold, so it is exactly flat. The index behind the percentile has Spearman −0.41 (−0.54 with the shipped references). At the same check-in number it orders only 28% of between-person pairs correctly. The current wording, "your typing looks more tired than X% of check-ins", is therefore anti-informative on our data.
- **Flipping backspaces alone doesn't fix it** (index Spearman −0.10). Rhythm unevenness and accuracy also run the other way. In `directions.json` (descriptive, post hoc), rhythm unevenness goes with KSS at −0.46 within a person and −0.50 across people at the same check-in number. Backspaces go at −0.51 and −0.28, and accuracy at −0.04 and +0.31.
- **Directions learned inside the folds rank but don't calibrate.** `fg5_sign_pooled` reaches Spearman 0.36, but its MAE (1.46) only just beats flat, its skill is −0.08, and its first-check-in MAE is 2.02.
- **The other families don't beat flat.** Rank scoring, ratio-only features, the reaction test and the percentile read as a quantile (MAE 2.4–2.5) all fail.
- **SWELL stress on raw speed and backspaces** ranks within people (0.74) but calibrates badly across them (MAE 1.56).
- **Adding speed, holds and reaction time makes ridge worse** (`ridge_wide`, MAE 1.79): it learns who is who. kNN comes second (MAE 1.29), but its first-check-in MAE is 2.18.

## The selected candidate: ridge_fg5

**Spec.** Ridge regression on absolute KSS, using the five first-guess features (rule-adjusted, median-imputed, standardised). Alpha is chosen by inner leave-one-person-out from 10^-2 to 10^3.

**Shuffle tests.** From `shuffle_test.py`, n = 200, re-running the whole procedure each time, inner tuning included:

| Shuffle | Spearman p | skill p | conc_matched p | first_MAE p |
|---|---|---|---|---|
| KSS across all check-ins (required test) | **0.005** | 0.005 | 0.005 | 0.22 |
| KSS only within the same check-in number (keeps the night's drift and practice) | 0.005 | 0.005 | 0.005 | 0.005 |
| KSS within person | 0.010 | 0.010 | 0.005 | 0.05 |
| Selection-aware: best of all 22 eligible variants on each shuffle | **0.010** | | | |

In the selection-aware test, the null's 95th percentile for the best Spearman is 0.39; ridge_fg5 observed 0.52.

**Why it still shouldn't score a judge's first try.** From `diagnostics.py`, post hoc:

- **The gain is in practised check-ins.**

  | Check-ins | n | ridge_fg5 MAE | flat MAE |
  |---|---|---|---|
  | 3 onward | 24 | 1.14 | 1.64 |
  | 1–2 (unpractised, like a stranger) | 9 | 1.10 | 1.14 |
  | 1–2, P5 left out of the data | 8 | 1.41 | 0.78 |

  With P5 left out, P1's first check-in is predicted at 9.0 against a rating of 4. On all 8 first attempts (5 panel first check-ins plus 3 walk-ups) the Spearman is 0.80, but that also depends on P5's one row. Walk-up MAE is 1.47 against flat's 1.12 with P5 in training, and 0.92 against 1.25 without.
- **The weights reverse the fatigue story.** Weights per SD on the whole panel: accuracy +0.82, rhythm unevenness −0.73, backspaces −0.30, relative pauses −0.26, rule cost +0.21. The four reversed weights keep their sign in every fold, while rule cost is about 0 in one fold. So "smooth, accurate and few corrections" means tired. That is the practice signature: within a person, predictions rise with check-in number (median ρ 0.40). The ranking does survive partialling out check-in number (0.48), and the same-check-in shuffle passes. But every judge is unpractised, and in our panel "unpractised" coincided with "early evening, alert". A careful typist would be told they are sleepy, and the drivers would have to say so.
- **The rest of the robustness checks hold.**
  - Every left-out person improves on flat.
  - Every alpha from 1 to 100 beats flat (MAE 1.07–1.33).
  - Most of the signal is rhythm and accuracy. Dropping backspaces even helps (1.07). That's post hoc, so it isn't adopted.
  - Without P5, overall MAE is 1.16 against 1.39, but skill is only 0.06.

## The spec I'd freeze, for confirmation only

`ridge_fg5`, fitted on all 33 panel check-ins. `frozen_spec.json` has every number: rule references, imputation medians, means, SDs and weights. In short:

```
z_f  = (x_f − mean_f) / sd_f                           x = rule-adjusted single-attempt feature, missing → training median
KSS  = clip(6.121 + 0.205 z_rule_cost − 0.731 z_gap_cv − 0.258 z_pause_rel
                  − 0.304 z_backspace + 0.816 z_accuracy, 1, 9)        alpha = 3.16
```

Score it, unchanged, on the morning batch in stranger mode: every check-in, no baseline. Set these checks in advance:

1. MAE below the flat guess of 6.12.
2. Spearman above 0.
3. If anyone slept, it must predict lower KSS for the people whose rating dropped. Practice stays high after sleep, so this is the clean test of practice against tiredness.

Only new walk-ups' first attempts can test the stranger case itself.

## Recommendation for the demo's first try

1. **Drop the "more tired than X%" line now.** The index behind it runs backwards on our data.
2. **Show no tiredness number on a stranger's first try.** Show a neutral typing snapshot against the team's check-ins: speed, slowdown from the rule, corrections and rhythm, as plain percentiles with no tired or alert words. Add one line: "One try can't tell your typing style from tiredness. Try again later and we'll compare you with yourself." The team's overnight average (about 6/9) can appear as context, never as their score. The pitch's tiredness claim belongs to teammates' personal, baseline-anchored scores.
3. **Add `ridge_fg5` as a labelled "experimental guess (±1.1 KSS)" only if it passes the morning checks above.** Even then, its drivers will tell careful typists they look sleepy.

## n_variants and files

**n_variants: 36.** That is 28 registered (`candidates.py`; 8 prior, 14 fold, 5 evidence, 1 check) plus 8 post-hoc sensitivity variants of ridge_fg5 (three fixed alphas, five leave-one-feature-out). All are logged to `../results.jsonl` with track `first_guess`: 28 records plus one sensitivity record.

Run from this folder with `uv run --project ../../.. [--with tabulate] python <script>`:

- `run_candidates.py [--log]` writes `results.csv`, `results.md`, `predictions.csv` and `selected.json`. It needs `--with tabulate`.
- `variance_split.py` writes `variance_split.json`.
- `shuffle_test.py 200` writes `shuffle_test.json` and takes about 6 minutes.
- `diagnostics.py` writes `diagnostics.json`.
- `describe_directions.py` writes `directions.json` (descriptive only).
- `frozen_spec.py` writes `frozen_spec.json`.
- `log_posthoc.py` was run once.
- `common.py` holds the data, folds and metrics; `candidates.py` holds every variant.
