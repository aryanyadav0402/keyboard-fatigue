# Track: complex and sequence models (keyboard-fatigue)

Run from `experiments/` with `uv run --project ../.. [--with torch] python sequence/<script>.py`
(scripts write their json output to the current directory; the files here were moved into this folder afterwards).
Files: `common.py` (IKI/hold extraction), `exp1.py` (distance/kernel), `exp2.py` (windows), `exp3.py` (variants), `final.py` (perm test), `log_all.py`, `table.md`, `*_results.json`.
All results are also in `../results.jsonl` (track="sequence", n_variants=28).

## Data handling
- Rule sentence (task 0) only. Non-modifier key presses paired down/up; IKI = down-to-down gap, hold = up minus down, both log; gaps over 5 s dropped. About 55 intervals per check-in (min 47).
- Each person's baseline = their own first two check-ins (the harness's `is_baseline`): median log-IKI/hold from those are subtracted from every check-in, and distances to "the person's baseline" use the pooled baseline intervals. Baselines are always earlier than anything scored, so this is available at demo time in both evaluations. Clock and check-in number are never used.
- Hyperparameters of the ridge/kernel models were tuned by inner leave-one-person-out inside the training rows. The CNN/GRU/MLP use fixed, pre-chosen hyperparameters (hidden 8, dropout 0.3, AdamW weight decay 0.1, 30 epochs, 3 seeds averaged; MLP alpha 10) because inner tuning of a net on 18-24 labels is noise.
- Windows: length W, stride S, each window inherits the check-in's label (delta or absolute), window predictions are averaged per check-in. Windows are built inside fit_predict from the harness's train/test split, so a check-in's windows and the test person's windows are never split.

## Results (all 28 evaluated variants; full table in `table.md`)
Baselines from the harness: guess lopo MAE 1.595 / skill 0 / conc 0.50 | rolling 1.710 / 0 / 0.615; v1 1.383 / +0.217 / 0.837 | 1.107 / +0.403 / 0.923. Control: v1 features with an inner-tuned alpha, 1.504 / +0.149 / 0.816 | 1.206 / +0.361 / 0.769 (so the tuning itself costs a bit).

| group | lopo skill range | rolling skill range | verdict |
|---|---|---|---|
| anchored (predict change from baseline rating), every model, 15 variants | -0.2 to -1.2 | -0.25 to -0.8 | worse than guessing. A constant "base + mean change" rule alone gets MAE 2.12, skill -0.72, so anchoring on a single baseline rating hurts here and v1 only escapes it through its features. |
| absolute, distance/kernel (ridge on distance-to-baseline features, KRR with W1 kernel) | -0.03, -0.27 | +0.03, +0.04 | no better than guess |
| absolute, windows (ridge, summary stats, CNN, GRU, MLP; W 10/16/24) | -0.06 to +0.09 | -0.13 to +0.08 | about guess level, tiny positive for ridge and CNN |
| v1 + sequence distance features (anchored) | -0.20, -0.30 | +0.34, +0.22 | adding them to v1 made it worse |

Best sequence candidate, `win24 ridge (abs)` (24-interval windows, stride 4, ridge alpha 1000 on baseline-centred log-IKI, log-hold, backspace flag; absolute KSS target):

| | lopo MAE | lopo skill | lopo conc | rolling MAE | rolling skill | rolling conc |
|---|---|---|---|---|---|---|
| win24 ridge (abs) | 1.566 | +0.086 | 0.694 | 1.668 | +0.075 | 0.692 |
| win24 cnn (abs) | 1.576 | +0.057 | 0.755 | 1.587 | +0.064 | 0.615 |
| v1 | 1.383 | +0.217 | 0.837 | 1.107 | +0.403 | 0.923 |
| guess | 1.595 | 0 | 0.500 | 1.710 | 0 | 0.615 |

Perm test (`final.py`, shuffle KSS within person, n=100 rather than 200; fast model, but kept at 100): lopo skill observed 0.086, p = 0.050 (null mean -0.041, sd 0.064); rolling skill 0.075, p = 0.050. This p is for one pre-chosen model; it was the best of 28 evaluated, so treat it as roughly unadjusted-p 0.05 times a large multiple-comparison penalty, i.e. no evidence. Passing would also not exclude practice drift.

## What happened, including failures
- Anchoring failed for every sequence model. The baseline rating is one number per person, between-person differences in the change (P2 about -2, P3 about +2.3) dominate, and distribution features cannot predict them.
- First run of the sklearn MLP (absolute) predicted near zero (MAE 4+) because I forgot to centre the target; fixed (mean-subtracted) and rerun, counted as a variant.
- Ridge on 12 baseline-distance features (W1, median/quantile shifts, spread, for IKI and hold) and KRR on W1 distance (kernel ridge = GP posterior mean; I did not run a separate GP) found nothing. Raw uncentred kernels were worse than baseline-centred ones.
- Within-person concordance of nets (0.74 to 0.80 in lopo, anchored CNN/GRU 0.80) looks decent but it did not hold in rolling (0.54) and goes with huge MAE, so it is likely ordering by practice/time-correlated drift rather than tiredness. Not trusted.
- v1's six features include rule cost, accuracy, backspaces and pauses: errors and rule handling, not just timing. IKI/hold distributions on ~55 intervals carry little beyond that.

## Freeze?
Nothing from this track. No sequence model beats v1 under either evaluation, none beats guess by more than 0.09 skill, and adding sequence features to v1 hurt. Keep v1.

## Caveats
24 labels from 4 people; every difference of ~0.05 skill here is inside noise (perm null sd 0.064). The lopo/rolling numbers for the "best" are selected from 28 tries. Torch was run with `--with torch`, seeds fixed (0-2 per model, perm-test shuffles seed 0).
