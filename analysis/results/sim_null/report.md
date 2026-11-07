# Tiredness from typing: results

Data from `../data/sim/null`, analysed Sat 07 Nov 20:12. 129 check-ins: 96 from 7 panel people and 33 from 20 one-off people. 81 panel check-ins are scored; each person's first 2 are their baseline. 12 in-progress checkpoint files were skipped.

- Task 1 rules shown: dashes_for_spaces 30, dots_and_vowels 29, capitalize_vowels 25, dashes_and_vowels 23, capitalize_first_and_last 22.
- The clock is hours since first check-in.
- 1 Task 1 timings dropped because the page lost focus.
- 6 check-ins were stopped early; their finished tasks are used.

## Findings

- Typing alone predicted tiredness with an average error of 1.79 KSS points, no better than the clock alone (hours since first check-in: 1.62). Always guessing the average gives 2.10.
- Clock check: typing didn't clearly add anything to the clock (error 1.62 → 1.78, shuffle test p = 0.431), so part of what it picks up may just be how late it is.
- The original test, which shuffles ratings within each person, gives p = 0.154.
- After allowing for practice and the rule type, the rule cost changed by 0.003 per hour on the clock (95% CI -0.012 to 0.017), which isn't clearly different from zero.
- Gaps next to a rule key (a capital, dash or dot) took 1.33× as long as ordinary gaps (Wilcoxon p < 0.001, 128 check-ins), so the rules do cost time.
- For 20 one-off people, who have no baseline, the model was no better than always guessing the panel average (error 1.92 vs 1.89 KSS points).

## How well does typing predict tiredness?

Each person is predicted by a model trained on everyone else. Anchored models predict the change from the person's own baseline rating, then add it back. Predictions use typing only; the clock (hours since first check-in) is a benchmark, because the demo can't ask for it.

| Model | Error (MAE, KSS points) | Spearman | Spearman within person (median) |
|---|---|---|---|
| Training mean | 2.10 | – | – |
| Own baseline rating | 2.77 | – | – |
| Clock (benchmark) | 1.62 | 0.41 | 0.73 |
| Typing only (main) | 1.79 | 0.34 | 0.01 |
| Typing + reaction test | 1.79 | 0.34 | -0.16 |
| Typing without rule cost | 1.80 | 0.34 | 0.02 |
| Typing, not anchored | 2.25 | -0.60 | -0.05 |
| Typing + clock (check only) | 1.78 | 0.27 | 0.51 |

**Clock check.** Adding typing to the clock changes the error from 1.62 to 1.78 KSS points. Shuffling the typing features within each person 500 times gives p = 0.431. A small p-value means typing knows something the clock doesn't.

**Original test.** Shuffling ratings within each person, as in the plan, gives a typing-only error of 2.24 and p = 0.154. This test can't tell typing apart from anything else that drifts over the night.

**Tired vs alert** (above each person's median KSS or not): AUC 0.47 with typing alone, 0.74 with the clock alone.

![Permutation tests](figures/permutation_tests.png)

![Tiredness over the night](figures/tiredness_over_night.png)

## Mixed models

### Tiredness

`kss ~ hours_clock + log_checkin + vt_gap_cv_dev + vt_rule_cost_dev`, random intercept per person.

92 check-ins from 7 people.

| Term | Coefficient | 95% CI | p |
|---|---|---|---|
| Intercept | 2.406 | 0.957 to 3.855 | 0.001 |
| hours_clock | 0.015 | -0.146 to 0.177 | 0.853 |
| log_checkin | 1.424 | 0.254 to 2.594 | 0.017 |
| vt_gap_cv_dev | 1.043 | -2.074 to 4.161 | 0.512 |
| vt_rule_cost_dev | -0.167 | -2.592 to 2.259 | 0.893 |

### Rule cost

`vt_rule_cost_adj ~ hours_clock + log_checkin`, random intercept per person.

95 check-ins from 7 people.

| Term | Coefficient | 95% CI | p |
|---|---|---|---|
| Intercept | 1.087 | 0.999 to 1.175 | < 0.001 |
| hours_clock | 0.003 | -0.012 to 0.017 | 0.729 |
| log_checkin | -0.064 | -0.166 to 0.039 | 0.225 |

![Rule cost over the night](figures/rule_cost_vs_hours.png)

## Do the rules really cost time?

Median gap next to a rule key ÷ median ordinary gap: 1.33 across 128 check-ins (Wilcoxon signed-rank p < 0.001).

## One-off people

A model trained on all panel check-ins, using only features that need no baseline (vt_rule_cost_adj, vt_gap_cv_adj, vt_accuracy_adj, backspace_per100_adj), predicted 33 check-ins from 20 one-off people with an error of 1.92 KSS points. Always guessing the panel average gives 1.89.

![Reaction time over the night](figures/reaction_time_over_night.png)

## Notes

- Typing features in the models: vt_rule_cost_dev, backspace_per100_adj, vt_accuracy_dev, vt_gap_cv_dev, vt_pause1s_share_dev, vt_hold_med_dev.
- Fixed features with no data here: none.
- Reaction-test features: rt_speed_dev.
- Task 1's rule changes every visit, so its features are first measured against the panel's typical value for that rule. Rule cost and accuracy are divided by it (1.0 = normal for that rule); the others have it subtracted.
- `_dev` features are each person's change from the average of their first two check-ins. Backspaces per 100 keys aren't compared with the baseline.
- The clock is hours awake when the app records a wake-up time (reset after a sleep of an hour or more), and otherwise hours since the person's first check-in.
- Lapses are reaction times of 355 ms or more. The app's 20-second test gives only about 6-9 trials, so the reaction-test comparison uses response speed (the mean of 1000 ÷ reaction time) instead.
