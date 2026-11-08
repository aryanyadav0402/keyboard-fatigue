# Simulation track: power, practice and selection bias

Everything here is simulated. No real check-in was read except through `harness.load_table()` (which stops at 00:53:51 on 8 Nov), and only for
spreads, correlations and the real baselines. No names are printed. LOPO only (rolling was not simulated).

## Setup

- `simlib.py`: feature-level simulator plus fast copies of the harness evaluation (LOPO, rolling, perm test). `check_fast_ridge.py` shows the fast
  ridge reproduces the harness exactly (guess/clock/v1/v3, LOPO and rolling, all four to 3 decimals).
- Why feature-level: `simulate.py` writes raw keystrokes (about 1 s per world); these questions need thousands of worlds. The simulator draws the processed table
  directly: the 6 TYPING features as changes from each person's first two check-ins (+ the 2 SWELL inputs), `hours_rel`, `kss`, `kss_base`.
- Calibration to the real scored table (24 rows, 4 people): feature spreads and their correlations are the real ones (noise correlation = real matrix shrunk halfway to
  identity; practice and fatigue move each feature by `pi` / `rho` standard deviations of its real spread, noise is shrunk so the total spread stays real). KSS: mean 6.1 (real 6.4), sd 1.6 (1.8),
  baseline 5.8 (5.7), rises about 0.6 per hour, within-person KSS-vs-check-in-number correlation 0.6 (real 0.75), report noise included.
- Fatigue direction per feature: rule cost up, backspaces up, accuracy down, gap CV up, pauses up, hold up. Practice goes the opposite way (faster, cleaner). `rho` = correlation of every
  feature with true tiredness, `pi` = correlation with log check-in number.
- Effect size: the real v1 (LOPO skill 0.217, concordance 0.837) is matched by `rho` about 0.4 (v1 skill 0.22, concordance 0.74). The project's own `simulate.py` "moderate" effect world at our size
  (`calib_pipeline.py`, 40 worlds through the full pipeline) gives v1 skill 0.15 (sd 0.47), about `rho` 0.35. So `rho` = 0.4 is the central "realistic" effect; 0.25 and 0.55 bracket it. The real feature
  spreads are 1.5 to 3.3 times those in `simulate.py`'s effect world, so real data is noisier than that simulator.
- Null skill is very noisy: with 4 people / 24 rows, a world with no effect at all has v1 LOPO skill mean -0.25, sd 0.80, and P(skill > 0) = 44%. Real v1 (0.217) is exceeded by 29% of pure-null worlds.
- Reps: A 1500 worlds per cell, B 600, C 1000 (+200 shuffles each), D 2000, E 2000. Runtime about 15 minutes in total. Seeds fixed.

Files: `power_grid.csv`, `morning.csv`, `morning_by_sleep_share.csv`, `practice.csv`, `practice_sleep.csv`, `real_match_and_signs.csv`, `selection.csv`, `selection_same_family.csv`,
`selection_whole_procedure_perm.csv` (raw per-world results in `*_raw.pkl`); scripts `run_experiments.py A|B|C|D`, `exp_B2.py`, `exp_D2.py`, `exp_E.py`, `log_results.py`.
Headline numbers are logged to `../results.jsonl` with `track="simulation"` (the first selection record has a key bug; `selection_bias_threshold_corrected` supersedes it).

## 1. Power

"v1 beats guessing" has to mean more than skill > 0, because under no effect skill > 0 happens 44% of the time and v1's MAE beats guessing 46% of the time. The test used is: v1's LOPO skill exceeds the 95th percentile
of its own null distribution for that design (false positive rate 5%). Power at `rho` = 0.4:

| design (people x scored rows) | rows | null 95th pct of skill | P(skill > 0) | P(skill > null 95th) |
|---|---|---|---|---|
| ours: 4 people (7,5,6,6) | 24 | 0.58 | 0.76 | **0.25** (0.11 at rho 0.25, 0.41 at 0.55) |
| 4 x 10 | 40 | 0.49 | 0.91 | 0.54 |
| 4 x 15 | 60 | 0.43 | 0.92 | 0.64 |
| 6 x 6 | 36 | 0.50 | 0.87 | 0.39 |
| 6 x 10 | 60 | 0.41 | 0.98 | 0.74 |
| 6 x 15 | 90 | 0.34 | 0.97 | 0.80 |
| 8 x 6 | 48 | 0.45 | 0.92 | 0.48 |
| 8 x 10 | 80 | 0.35 | 0.99 | 0.86 |
| 12 x 6 | 72 | 0.37 | 0.98 | 0.72 |
| 12 x 10 | 120 | 0.28 | 1.00 | 0.98 |
| 20 x 4 | 80 | 0.37 | 0.98 | 0.59 |

Reaching 80% power at `rho` 0.4 takes about 8 people with 10 scored check-ins each (80 rows), or 6 people with 15 each (90 rows). Four people do not get there even with 15 rows each (4 x 15 is 64%), because with 4 folds
one odd person dominates. At `rho` 0.25 it takes about 12 x 15 (12 x 10 gives 72%). People matter more than check-ins per person. (Full grid, including 4 to 20 people, 4 to 15 check-ins and four effect sizes: `power_grid.csv`.)

**v1 vs v3.** The paired difference in LOPO MAE (v1 minus v3) has a sampling sd of 0.32 to 0.39 in our design (0.2 under no effect), falling to about 0.11 at 20 people x 15 rows (300 rows). The real difference, 0.016, is
about 0.05 sd: P(|difference| < 0.02) is 4.5% at our size even when the models truly differ by 0.3, and a coin flip on the sign when they don't. Shrinking the sd to 0.01 (so that 0.02 is 2 sd) would take roughly 30,000 scored check-ins.
It cannot be detected with this design, and no feasible extension of it fixes that. (In the simulation v1 beats v3 by a lot when there is an effect, 79% at rho 0.4, but that is by construction: v1 uses six features that carry signal and v3 two.
A real 0.02 gap is noise.)

**Morning batch** (our 24 night rows, then 10 / 15 / 25 more check-ins from the 4 people, half of whom sleep first; models frozen on the night and tested on the morning rows; `morning.csv`):

| true world | M | P(v1 beats guess, MAE) | P(frozen skill > null 95th) | pooled LOPO shuffle test p < 0.05 | P(v1 beats v3) |
|---|---|---|---|---|---|
| no effect | 10 / 15 / 25 | 0.52 / 0.50 / 0.49 | 0.05 | 0.04 / 0.06 / 0.07 | 0.51 / 0.52 / 0.56 |
| rho 0.25 | 10 / 15 / 25 | 0.82 / 0.83 / 0.83 | 0.34 / 0.37 / 0.45 | 0.70 / 0.80 / 0.89 | 0.91 to 0.96 (by construction) |
| rho 0.4 | 10 / 15 / 25 | 0.94 / 0.94 / 0.95 | 0.68 / 0.75 / 0.79 | 0.94 / 0.96 / 0.99 | 0.98 to 0.99 (by construction) |
| practice only (pi 0.6) | 10 / 15 / 25 | 0.29 / 0.34 / 0.39 | 0.01 / 0.01 / 0.02 | 0.03 | 0.23 to 0.40 |

If the true effect is `rho` 0.4, the morning batch settles "v1 beats guessing" about 94% of the time (by MAE), and clears the null 95th percentile 68 to 79% of the time. It does not settle v1 vs v3 (the sd of their
MAE difference is still 0.2 to 0.5). The frozen test is cleaner than re-fitting because the model cannot be tuned on the morning rows. These numbers need the sleep contrast (next section).

## 2. Practice (typing improves with each check-in, no fatigue effect, KSS still rises)

Real-size design (4 people, 24 rows), 1000 worlds per row, perm test = harness shuffle (200 shuffles), "looks real" = skill > 0 and within-person concordance >= 0.65 and shuffle p < 0.05 (`practice.csv`):

| world | P(skill > 0) | P(conc >= 0.65) | P(conc >= 0.8) | P(shuffle p < .05) | P(looks real) | P(skill >= .2 and conc >= .8, like ours) |
|---|---|---|---|---|---|---|
| null | 0.45 | 0.11 | 0.00 | 0.05 | 0.02 | 0.000 |
| practice 0.3 | 0.48 | 0.47 | 0.07 | 0.17 | 0.12 | 0.04 |
| practice 0.5 | 0.53 | 0.70 | 0.24 | 0.30 | 0.23 | 0.14 |
| practice 0.7 | 0.56 | 0.79 | 0.33 | 0.38 | 0.29 | 0.20 |
| practice 0.85 | 0.58 | 0.83 | 0.37 | 0.40 | 0.32 | 0.22 |
| fatigue 0.4 | 0.77 | 0.84 | 0.32 | 0.62 | 0.54 | 0.26 |
| fatigue 0.55 | 0.86 | 0.96 | 0.58 | 0.83 | 0.76 | 0.48 |
| both (fatigue 0.4 + practice 0.5) | 0.45 | 0.11 | 0.01 | 0.03 | 0.02 | 0.00 |

With practice alone and no fatigue effect, v1 passes all three checks about 23 to 32% of the time, and looks like our real result (skill >= 0.2 and concordance >= 0.8) 14 to 22% of the time. Mean skill is still slightly
negative: practice makes concordance and the shuffle test pass far more often than it makes skill positive. Skill and concordance together beat either alone, but a fatigue effect passes 54% of the time against 23 to 32%, a likelihood
ratio of only about 2. The shuffle test cannot separate them (it passes anything that drifts over the night). Real v1 numbers (skill 0.217, concordance 0.837 both at once, `real_match_and_signs.csv`): P = 0.2% under null,
8 to 14% under practice only, 16 to 33% under fatigue 0.4 to 0.55. The null is rejected; practice vs fatigue is not. A mix of both with opposite signs cancels (last row): practice can also hide a real effect.

**Sign diagnostic (free, uses existing data).** Check whether v1's standardised coefficients point in the fatigue direction (rule cost +, backspaces +, accuracy -, gap CV +, pauses +, hold +). With fatigue 0.4, 5.3 of 6 do
(P(>= 5 of 6) = 85%); with practice only, 1.4 to 1.5 of 6 (P(<= 1 of 6) is 50%); with null, 3.0 of 6. Real v1: 3 of 6 (rule cost, backspaces, gap CV in the practice direction; accuracy, pauses, hold in the fatigue direction).
That looks like the null, not a clean fatigue effect. Six coefficients fitted on 24 rows is weak evidence, but no simulated world in which v1 matched our numbers and had 5+ fatigue-direction signs came from practice (0%) or null (0%); 14 to 30% of fatigue worlds did.

**What separates practice from fatigue: check-ins after sleep** (`practice_sleep.csv`, 4 people, half sleep, 3 or 5 morning check-ins each, true KSS drop after sleep about 3 points). Practice keeps accumulating while tiredness resets.

| world | P(frozen v1 predicts a lower KSS after sleep than at last night check-in) | frozen skill on after-sleep rows | v1 minus practice-only model, pooled LOPO (night + morning), P(> 0.1) |
|---|---|---|---|
| null | 0.63 to 0.65 | -0.13 | 0.19 to 0.24 |
| practice 0.5 | 0.11 to 0.19 | -1.0 to -1.1 | 0.13 |
| practice 0.7 | 0.07 to 0.10 | -1.2 to -1.4 | 0.10 |
| fatigue 0.4 | 1.00 | +0.54 | 0.97 to 0.98 |
| both 0.4 / 0.5 | 0.67 | -0.02 to -0.06 | 0.66 to 0.75 |

Night data alone cannot do it (v1 minus a practice-only model on the night rows: P(> 0.1) is 12% under practice, 53% under fatigue 0.4, since check-in number and hours awake are collinear). After-sleep rows separate them almost completely:
the sign of v1's predicted change across the sleep, and v1's frozen skill on the after-sleep rows (>0 in 93% of fatigue worlds, 6 to 25% under practice). Caution: this only works if people really slept. If nobody sleeps and morning check-ins
simply continue the night, practice looks even more like fatigue (`morning_by_sleep_share.csv`, 15 morning check-ins, frozen v1 beats guess: no sleep: practice-only 90% vs fatigue 99% vs null 37%; half slept: 22% / 95% / 45%; all slept: 4% / 99% / 49%).
Use the sleepers' check-ins (and the same person's check-in right before and right after sleep) as the test.

## 3. Selection bias (null world, 4 people, 24 scored rows)

Variants: anchored ridge on a random subset of 2 to 8 of the 8 typing/SWELL features, random alpha (30% RidgeCV), pick the best LOPO skill (`selection_same_family.csv`, 2000 worlds). The first version of this (`selection.csv`) also mixed in
unanchored models, the clock and a 20-feature pool; it gives about the same best-of-K skill (K = 50: 0.78 at the 95th percentile) and shows the same pattern.

| K variants tried | mean best skill by chance | 95th pct of best skill (**threshold**) | 95th pct of MAE improvement over v1 (**threshold**) | mean MAE improvement over v1 by chance |
|---|---|---|---|---|
| 1 (a single new variant) | -0.25 | 0.60 | 0.35 | -0.02 |
| 5 | -0.02 | 0.66 | 0.49 | 0.15 |
| 20 | 0.06 | 0.69 | 0.56 | 0.21 |
| 50 | 0.09 | 0.70 | 0.63 | 0.24 |

(v1 itself: null skill mean -0.22. Practice-only world, 4 people: thresholds are 0.74 / 0.77 / 0.79 for K = 5 / 20 / 50, so practice makes a win easier but not believable.)

**Threshold for a believable win over v1 in our design: a variant chosen from K <= 50 tried needs LOPO skill above about 0.70 (and an LOPO MAE at least 0.5 to 0.6 below v1's 1.383, so about 0.8 to 0.9).** At K = 20 chance alone
gives a 0.2 MAE gain on average, so a variant that beats v1 by 0.2 is what luck delivers. In the mixed-family version (`selection.csv`) the best of 20 variants exceeds skill 0.2 in 72% of pure-null worlds. With a single
pre-specified variant (K = 1) the bar is the null 95th percentile, 0.60, still above v1's 0.217. Count every variant tried, including different alphas, feature subsets and
the ones dropped.

The harness `perm_test` on the winning variant does not fix this: after picking the best of 50, it declares p < 0.05 in 59 to 68% of null worlds (`selection.csv`: 68% for our design; 77% for practice-only). The fix that works is
to repeat the whole selection inside each shuffle: with K = 20 it gives 4.7% false positives under null and 66% power at `rho` 0.4 (300 worlds, 100 shuffles; `selection_whole_procedure_perm.csv`).

## What the morning data can and can't settle

Can:
- Whether there is a fatigue effect of roughly `rho` 0.4 or more, versus none or practice only, if at least some people sleep and their post-sleep check-ins (3 or more, ideally the one just before and just after sleep) are kept
  as a frozen-model test. Frozen v1 beats guessing 94% of the time under `rho` 0.4, clears the null 95th percentile 68 to 79% of the time (10 to 25 check-ins), and loses badly to guessing under practice only.
- Rule out "v1 predicts tiredness" cleanly if the after-sleep predictions go the wrong way.
- Check the sign pattern of the coefficients; a clean 5+/6 fatigue-direction pattern is evidence against practice.

Can't:
- v1 vs v3 (0.02 MAE gap against a sampling sd of 0.2 to 0.4). Treat them as tied; choose by simplicity or demo reasons.
- A small effect (`rho` 0.25 or less): power stays at 34 to 45% at the null 95th percentile even with 25 check-ins.
- Practice vs fatigue if nobody sleeps, or if morning check-ins are only from the same awake stretch.
- Any new variant tried on the night data: nothing under 0.70 skill (or 0.5 to 0.6 MAE gain over v1) is distinguishable from selection luck at K up to 50.

Limitations: the simulator is feature-level and Gaussian; real typing features are non-linear and heavy-tailed. The practice and fatigue effects are modelled as equal-size loadings on all features; real effects will be
uneven (which helps a model that picks the right features and hurts one that doesn't). KSS noise and person offsets are set so spreads match 24 real rows, which is a thin calibration target. The after-sleep drop (about 3 KSS) is assumed, not measured;
a smaller drop weakens the separation. 4 people were used for the morning batch, not 5.
