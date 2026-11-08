# Keystroke-level features (track: keystrokes)

**Bottom line: nothing here beats v1, and I would not freeze any keystroke variant.** The useful
results are about how people handle mistakes and about reliability:

- **Tired people make fewer errors. They don't fix fewer.** Nearly every error gets fixed whether
  people are alert or tired, so "backspacing less" means fewer typos to fix.
- **v1's leave-one-person-out (LOPO) skill rests on `vt_gap_cv_dev`.** That feature's within-person
  split-half reliability is about zero. Expect v1 to shrink on new data.

Data: the 36 check-ins up to 00:53:51 (harness), 24 scored panel check-ins from 4 people. All
outputs use pid only.

## What I built

**Buffer replay** (`ks.py`). Each rule sentence uses only a few kinds of key:

| Key | Key-downs | Notes |
|---|---|---|
| Character keys | | |
| Shift | 503 | |
| Backspace | 193 | 32 of them are auto-repeats from holding the key |
| Enter | 36 | One per check-in, to submit |

There are no Delete, arrow, Tab, Caps Lock or paste events. The app's input stream has only
`insertText` and `deleteContentBackward`, and the caret is always at the end. So the buffer works
as a stack: a character key-down pushes, and a Backspace key-down pops (repeats too). On all 36
check-ins the replay reproduces three things exactly:

- the typed text
- the app's input stream (event types and lengths)
- the final Levenshtein distance

My recomputed v1 raw features also match `features_raw.csv` exactly. Note that `features.py`
counts a held Backspace as one press, so `backspace_per100` slightly undercounts deletions.

**Errors.** After every key the replay keeps d = min over j of Levenshtein(buffer,
expected[:j]), the distance to the best-matching start of the answer. It stores one DP row per
buffer character, so a Backspace restores the previous row exactly. A key that raises d is an
error keystroke. This handles substitutions, extra characters and skipped characters. Final d
always equals the number of error keystrokes still in the text.

Each error is classified as one of two kinds:
- **Rule error:** the rule wasn't applied, was applied where it shouldn't be, or the wrong
  separator was used.
- **Typo:** any other wrong character.

**Features** (`keystroke_features.csv`, one row per check-in, keyed by `checkin_id` with `pid`):

| Group | Features |
|---|---|
| Errors made | `ks_err_per100`, `ks_err_rule_per100`, `ks_err_typo_per100` (per 100 expected characters) |
| Fixed vs left | `ks_fix_rate`, `ks_left_per100`, `ks_spurious_del_per100` (deletions in runs that remove no error) |
| Detection | `ks_detect_keys` (keys typed after an error before the Backspace run that removes it), `ks_detect_ms`, `ks_immediate_share` |
| Post-error slowing | `ks_post_err_ratio` (gap right after an error key), `ks_resume_ratio` (end of a correction run to the next key), `ks_pes_ratio` (the next 2 gaps); each divided by the check-in's median clean gap |
| Timing shape | `ks_iki_q10/q50/q90`, `ks_iki_logsd`, `ks_tail_share` (gaps > 2x own median), all on clean gaps between ordinary keys (both keys untouched by the rule, no Backspace or error key involved) |
| Baseline-relative | `ks_tail_base90` (share of gaps above the person's baseline 90th percentile), `ks_w1_base` (Wasserstein distance between log gaps and the person's pooled baseline log gaps), `ks_ks_base` (KS statistic). A baseline check-in is compared with the other baseline check-in. |

**`_dev` versions** follow v1's convention:
1. Rule adjustment: subtract the panel median for that rule type and add back the overall median.
   This is label-free and works the same way as `features.rule_references`.
2. Subtract the mean of the person's two baseline check-ins.

The three baseline-relative features are used as they are.

## Errors are sparse, and nearly all get fixed

On P1–P4's 32 check-ins, the median sentence (about 50 characters) has:
- 5.8 error keystrokes per 100 characters (IQR 3.9–12.1), about 3 errors in total
- fix rate 1.00 (IQR 0.86–1.00)
- 0 errors left in the text (IQR 0–1.9)
- spurious deletions about 0
- immediately caught share 1.0: most errors are fixed before another character is typed
- detection latency 741 ms (IQR 514–881)

Across all 36 check-ins there are 184 error keystrokes.

## Split-half reliability (`reliability.csv`)

**How it's computed.** Each feature is measured on two parts of the sentence, using two ways of
splitting it:
- first half vs second half (split at the word boundary nearest the middle)
- odd vs even words, as a robustness check

The table reports the **within-person** correlation between the two parts:
- each part is rule-adjusted and centred on the person's mean
- the sample is P1–P4, up to 32 check-ins
- each value is a Spearman correlation, Spearman-Brown corrected

Within-person reliability matters here, not between-person: tracking tiredness depends on changes
inside a person. Between-person split-half values are much higher for speed features (median gap
0.86, hold 0.94) because people differ a lot from each other.

Column key:
- **noP1b1:** first/second half again, but without P1's first baseline check-in, where the rule
  was misread and 8 errors were left in the text.
- **90% CI:** a bootstrap interval on the raw first/second-half r.

| feature | first/second | odd/even | noP1b1 | 90% CI | verdict |
|---|---|---|---|---|---|
| **v1** `vt_hold_med_dev` | 0.79 | 0.74 | 0.80 | 0.40–0.84 | reliable |
| **v1** `vt_pause1s_share_dev` | 0.58 | 0.39 | 0.44 | 0.10–0.68 | moderate |
| **v1** `backspace_per100_adj` | 0.51 | 0.44 | 0.44 | 0.02–0.63 | moderate |
| **v1** `vt_rule_cost_dev` | 0.36 | 0.26 | 0.27 | −0.12–0.53 | weak |
| **v1** `vt_accuracy_dev` | 0.61 | 0.61 | **0.09** | 0.02–0.80 | unreliable (all from one check-in) |
| **v1** `vt_gap_cv_dev` | **−0.28** | 0.27 | −0.28 | −0.65–0.11 | unreliable |
| `ks_w1_base` | 0.71 | 0.60 | 0.71 | 0.19–0.80 | best new feature |
| `ks_iki_q50` / `q10` / `q90` | 0.54 / 0.49 / 0.60 | 0.52 / 0.59 / 0.31 | same | | moderate |
| `ks_err_per100` (errors made) | 0.48 | 0.46 | 0.43 | 0.02–0.58 | moderate (about the same as backspaces) |
| `ks_err_typo_per100` | 0.46 | 0.53 | 0.41 | | moderate |
| `ks_err_rule_per100` | −0.17 | −0.12 | −0.19 | | unreliable |
| `ks_fix_rate`, `ks_left_per100` | 0.54, 0.54 | 0.61, 0.38 | **−0.04, 0.00** | | unreliable (all from one check-in) |
| detection (`keys`, `ms`, `immediate`) | 0.16, −0.06, 0.21 | 0.31, 0.25, −0.13 | about 0 | | unreliable |
| post-error slowing (`post_err`, `resume`, `pes`) | 0.47, −0.31, −0.02 | 0.07, −0.11, −0.16 | | | unreliable |
| `tail_share`, `tail_base90`, `ks_base`, `iki_logsd` | 0.33, 0.13, 0.42, 0.23 | −0.02, 0.07, 0.15, 0.64 | | | weak or inconsistent |

Timing features need at least 5 ordinary gaps in each part, so they rest on only 17–24 check-ins.
Every CI is wide.

**How reliable is the v1 score itself?** I combined half-sentence features with v1's weights
(post hoc):
- using all check-ins, the score's within-person split-half is 0.31 (first/second half) and 0.15
  (odd/even)
- complete-case subsets (n = 14–19) swing from −0.22 to 0.65

The score is not stable within one sentence.

## Candidates

These were fixed before any label was looked at, from theory and reliability only.
`run_models.py` states the rule.

| | change to v1 | why |
|---|---|---|
| C1 | backspaces → errors made | backspaces = errors × fix rate + other deletions. Fix rate is about 1 and spurious deletions about 0, so errors made is the cleaner count. |
| C2 | + rhythm distance (`ks_w1_base_dev`) | the most reliable new feature |
| C3 | drop unreliable v1 features (rule cost, accuracy, gap CV) | keep rule: split-half ≥ 0.3 on both splits and without the outlier check-in |
| C4 | C3 + rhythm distance | |

All use the anchored ridge, exactly like v1 (RidgeCV picks alpha inside each training fold).
**n_variants = 4.** All four are logged to `results.jsonl` with track "keystrokes".

## Results vs baselines (`model_table.csv`, `model_results.json`)

The permutation test shuffles KSS within each person (n = 200). Statistic: skill.

| model | LOPO MAE | LOPO skill | LOPO within-conc | rolling MAE | rolling skill | rolling within-conc | perm p (LOPO / rolling) |
|---|---|---|---|---|---|---|---|
| guess | 1.595 | 0.000 | 0.500 | 1.710 | 0.000 | 0.615 | |
| clock | 1.791 | −0.325 | 0.878 | 1.559 | −0.144 | 0.423 | |
| **v1** | **1.383** | **0.217** | 0.837 | **1.107** | 0.403 | **0.923** | 0.020 / 0.010 |
| v3 | 1.399 | 0.138 | 0.653 | 1.150 | 0.422 | 0.692 | |
| C1 bs → errors made | 1.663 | −0.190 | 0.796 | 1.125 | 0.440 | 0.769 | 0.070 / 0.010 |
| C2 v1 + rhythm distance | 1.658 | −0.122 | 0.857 | 1.266 | 0.345 | 0.769 | 0.045 / 0.010 |
| C3 v1 reliable only | 2.295 | −1.068 | 0.704 | 1.885 | −0.217 | 0.846 | 0.920 / 0.100 |
| C4 C3 + rhythm distance | 2.686 | −1.743 | 0.684 | 2.010 | −0.365 | 0.846 | 1.000 / 0.194 |

- **C1** has the best rolling skill of any model (0.440). But its MAE is worse than v1 and its LOPO
  skill turns negative. Not a win.
- **C2** gets a negative weight on rhythm distance. Within person, rhythm distance falls with check-in
  number (ρ = −0.30 on scored check-ins), so it mostly tracks time, not tiredness.
- **C3 and C4 collapse.** The ablation below explains why.

## Post-hoc diagnostics (`diagnose.py`)

These were run after scoring the candidates. They are not candidates and are not eligible for
freezing.

**Anchor-only reference** (own baseline rating + average training change, no typing): LOPO MAE 2.116,
skill −0.72; rolling MAE 2.115. This is much worse than guessing. Much of v1's skill comes from
typing features correcting the anchor for P2 (baseline rated 7, later 3–7) and P3.

**v1 leave-one-feature-out** (`ablation_table.csv`):

| v1 minus | LOPO MAE | LOPO skill | rolling MAE | rolling skill |
|---|---|---|---|---|
| rule cost | 1.355 | 0.285 | 0.999 | 0.492 |
| backspaces | 1.336 | 0.229 | 1.110 | 0.426 |
| accuracy | 1.326 | 0.231 | 1.102 | 0.457 |
| **gap CV** | **2.174** | **−0.760** | 1.596 | −0.097 |
| **pause 1 s share** | 1.603 | −0.132 | 1.385 | 0.068 |
| hold | 1.327 | 0.284 | 1.240 | 0.367 |

v1's LOPO skill comes almost entirely from gap CV, which has about zero split-half reliability, and
pause share. Each of the other four can be dropped with no loss. The best ablation (minus rule cost)
was picked from 6 after seeing labels, so its gain is optimistic. Don't freeze it on this evidence.

**How tired people handle mistakes** (`associations.csv`). These are exploratory and involve many
comparisons. They are within-person Spearman correlations on P1–P4's 32 check-ins, including
baselines, with features rule-adjusted.

Errors made, by feature:

| feature | ρ with KSS | after removing each person's linear trend over check-in number | ρ with check-in number |
|---|---|---|---|
| `ks_err_per100` (errors made) | −0.53 | −0.36 | −0.17 |
| `ks_err_typo_per100` (typos) | −0.60 | −0.35 | |
| `ks_err_rule_per100` (rule errors) | −0.35 | | |
| `backspace_per100_adj` | −0.59 | | |

Errors made are negative for every person: P1 −0.16, P2 −0.93, P3 −0.60, P4 −0.47. They are
unchanged without P1's outlier check-in (−0.50).

Fixing doesn't change:
- fix rate ρ = −0.13
- errors left ρ = +0.04
- detection latency ρ = +0.23 (+0.31 after removing the trend), but this feature is unreliable

Timing:
- More 1 s+ pauses when tired: ρ = +0.45, and +0.41 after removing the trend.
- Fewer typos go with more long pauses: ρ = −0.35 (label-free).
- Ordinary-key speed doesn't change with KSS: q50 ρ = 0.01.

A cautious-typing account fits this pattern: tired people pause more, slip less, and still fix
everything. Practice is the main alternative: KSS rises with check-in number within person
(ρ = 0.57). But errors barely trend with check-in number, and the association survives removing
that trend.

## What I'd freeze

**No model.** Keep v1 as frozen. If the team wants a cheap confirmation on the morning check-ins,
pre-register this descriptive check rather than a model:

- **Feature:** `ks_err_per100` (or `ks_err_typo_per100`), computed by `ks.py` on the rule sentence and
  taken as a change from baseline.
- **Prediction:** its within-person correlation with KSS is negative, and the median fix rate stays at
  0.9 or above.

Also watch `vt_gap_cv_dev`. If v1 fails in the morning, its unreliability is the first suspect.

## Caveats

- 24 labels from 4 people. LOPO is 4 folds, and every number above has wide uncertainty.
- I chose the reliability threshold after seeing the reliability table. That's label-free, but it's
  still a choice. The 0.3 cut-off and the CI-based alternative give different keep-lists for rule
  cost.
- Rule-type medians use all 36 check-ins. That's label-free and the same as the pipeline.
- Split-half on IQR-based statistics with about 10 gaps per half may understate reliability. Both
  splits agree, though.
- P1's first baseline check-in misread the rule, which shifts P1's baseline for accuracy and
  errors-left.
- Total label-touching evaluations: 4 candidates (logged), plus 6 post-hoc ablations and 1 reference
  (not candidates).

## Files and rerun

From this folder, using `uv run --project ../../.. python <script>`:

1. `run_features.py` writes:
   - `keystroke_features.csv`
   - `halves.csv`
   - `reliability.csv`

   It is label-free and takes about 25 s.
2. `run_models.py` runs the 4 candidates, permutation tests and logging, and writes `model_table.csv`
   and `model_results.json`. It takes about 2 min. **Rerunning it appends duplicate entries to
   `results.jsonl`.**
3. `diagnose.py` runs the post-hoc diagnostics and writes `associations.csv` and
   `ablation_table.csv`.

`ks.py` is the replay and feature library.
