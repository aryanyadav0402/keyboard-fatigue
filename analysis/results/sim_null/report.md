# Tiredness from typing: results

Data from `../data/sim/null`, analysed Sat 07 Nov 19:27. 122 check-ins: 95 from 7 panel people and 27 from 20 one-off people. 81 panel check-ins are scored; each person's first 2 are their baseline.

- 4 vowel task timings dropped because the page lost focus.
- 10 free text timings dropped because the page lost focus.
- 4 check-ins were stopped early; their finished tasks are used.

## Findings

- Typing alone predicted tiredness with an average error of 2.24 KSS points, no better than hours awake alone (1.40). Always guessing the average gives 1.93.
- Clock check: typing didn't clearly add anything to hours awake (error 1.40 → 1.57, shuffle test p = 0.727), so part of what it picks up may just be how late it is.
- The original test, which shuffles ratings within each person, gives p = 0.473.
- After allowing for practice, the rule cost changed by -0.002 per hour awake (95% CI -0.008 to 0.004), which isn't clearly different from zero.
- Gaps next to a capital took 1.32× as long as ordinary gaps (Wilcoxon p < 0.001, 118 check-ins), so the vowel rule does cost time.
- Across 3 sleeps, KSS changed by -6.0 on average and the rule cost by -0.16.
- For 20 one-off people, who have no baseline, the model was no better than always guessing the panel average (error 1.55 vs 1.53 KSS points).

## How well does typing predict tiredness?

Each person is predicted by a model trained on everyone else. Anchored models predict the change from the person's own baseline rating, then add it back. Predictions use typing only; hours awake is a benchmark, because the demo can't ask for it.

| Model | Error (MAE, KSS points) | Spearman | Spearman within person (median) |
|---|---|---|---|
| Training mean | 1.93 | – | – |
| Own baseline rating | 3.15 | – | – |
| Hours awake (benchmark) | 1.40 | 0.60 | 0.88 |
| Typing only (main) | 2.24 | -0.05 | -0.17 |
| Typing + reaction test | 2.26 | -0.08 | -0.23 |
| Typing without rule cost | 2.23 | -0.02 | -0.19 |
| Typing, not anchored | 1.95 | -0.48 | -0.26 |
| Typing + hours awake (check only) | 1.57 | 0.52 | 0.82 |

**Clock check.** Adding typing to hours awake changes the error from 1.40 to 1.57 KSS points. Shuffling the typing features within each person 500 times gives p = 0.727. A small p-value means typing knows something the clock doesn't.

**Original test.** Shuffling ratings within each person, as in the plan, gives a typing-only error of 1.95 and p = 0.473. This test can't tell typing apart from anything else that drifts over the night.

**Tired vs alert** (above each person's median KSS or not): AUC 0.51 with typing alone, 0.79 with hours awake alone.

![Permutation tests](figures/permutation_tests.png)

![Tiredness over the night](figures/tiredness_over_night.png)

## Mixed models

### Tiredness

`kss ~ hours_awake + log_checkin + vt_gap_cv_dev + vt_rule_cost_dev`, random intercept per person.

91 check-ins from 7 people.

| Term | Coefficient | 95% CI | p |
|---|---|---|---|
| Intercept | 1.183 | 0.284 to 2.082 | 0.010 |
| hours_awake | 0.249 | 0.195 to 0.303 | < 0.001 |
| log_checkin | 1.104 | 0.668 to 1.540 | < 0.001 |
| vt_gap_cv_dev | 1.194 | -1.059 to 3.448 | 0.299 |
| vt_rule_cost_dev | 2.327 | 0.446 to 4.208 | 0.015 |

### Rule cost

`vt_rule_cost ~ hours_awake + log_checkin`, random intercept per person.

91 check-ins from 7 people.

| Term | Coefficient | 95% CI | p |
|---|---|---|---|
| Intercept | 1.470 | 1.378 to 1.561 | < 0.001 |
| hours_awake | -0.002 | -0.008 to 0.004 | 0.502 |
| log_checkin | -0.081 | -0.127 to -0.034 | < 0.001 |

![Rule cost against hours awake](figures/rule_cost_vs_hours.png)

## Is the vowel rule really harder?

Median gap next to a capital ÷ median ordinary gap: 1.32 across 118 check-ins (Wilcoxon signed-rank p < 0.001).

## Before and after sleep

| Person | Slept (min) | KSS | Rule cost | Median RT (ms) |
|---|---|---|---|---|
| P02-XETJ | 162 | 9 → 1 | 1.33 → 1.25 | 273 → 276 |
| P05-FBRG | 179 | 9 → 3 | 1.47 → 1.15 | 261 → 263 |
| P07-WXGF | 171 | 7 → 3 | 1.23 → 1.14 | 270 → 264 |

![Before and after sleep](figures/before_after_sleep.png)

## One-off people

A model trained on all panel check-ins, using only features that need no baseline (vt_rule_cost, vt_gap_cv, vt_accuracy, backspace_per100), predicted 27 check-ins from 20 one-off people with an error of 1.55 KSS points. Always guessing the panel average gives 1.53.

![Reaction time over the night](figures/reaction_time_over_night.png)

## Notes

- Typing features in the models: vt_rule_cost_dev, backspace_per100, vt_accuracy_dev, vt_gap_cv_dev, vt_pause1s_share_dev, vt_hold_med_dev, ft_pause2s_share_dev.
- Fixed features with no data here: none.
- Reaction-test features: rt_lapse_rate_dev.
- `_dev` features are each person's change from the average of their first two check-ins. Backspaces per 100 keys are used as they are. Hours awake reset after a sleep of an hour or more.
- Lapses are reaction times of 355 ms or more, the cut-off for the 3-minute test.
