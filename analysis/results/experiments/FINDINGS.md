# Overnight modelling: what we found

8 Nov 2026, 01:55–02:35. Five parallel experiment tracks on the 36 check-ins collected by 00:53 (24 scored, from the 4 teammates). Everything here is in `analysis/results/experiments/`, which git ignores. Nothing was pushed, deployed or committed.

## The short version

- **No model beat v1.** About 117 variants across four tracks didn't improve on v1 in a way that survives honest tests: sequence and neural models, keystroke-level features, small-data training strategies, and first-guess models. Keep v1.
- **Much of what the models learn looks like practice.** Typing gets smoother and cleaner with every check-in, while check-ins get later and more tired, so "smooth typing" ends up meaning "tired".
- **That makes the first-try percentile backwards.** The percentile shown to strangers comes from the shipped first-guess index, which correlates −0.50 with tiredness across our check-ins (p = 0.003). We recommend dropping it from the demo.
- **But v1 isn't just the clock.** In 6 cases where someone was less tired at a later check-in, v1 put all 6 in the right order; the clock got none. Four of the six come from one person.
- **Check-ins after sleep settle it,** because people are still practised but rested. A frozen test for the morning batch is ready (`frozen/`).

## What we tried

| Track | Variants | Best result vs v1 | Verdict |
|---|---|---|---|
| Sequence and complex models (kernels on timing distributions, CNN, GRU, sliding windows) | 28 | Leave-one-person-out skill +0.09, against v1's +0.22 | Nothing helps; 24 labels can't train them |
| Keystroke-level features (errors made and fixed, detection delay, post-error slowing, rhythm distance) | 4, fixed before looking at any labels | None beat v1 | Keep v1; interesting finding on mistakes below |
| Training strategies (pairwise ranking, ordinal, anchoring, lasso, PLS, practice removal, composite label, ensembles, baseline rows) | 49 | Lower errors, but on shuffled labels they beat v1 by just as much (p 0.12–0.89) | Gains come from setting a person's overall level, not from tracking tiredness |
| First guess for strangers | 36 | One model (ridge_fg5) beats flat, but only on practised check-ins | Frozen for testing only |
| Simulation, power and selection bias | — | — | Gives the thresholds below |

Every track used the same harness (`harness.py`) with the same two evaluations:
- **Each person left out:** every teammate is predicted by a model that never saw them.
- **Rolling:** every check-in is predicted by a model trained only on earlier check-ins.

Each track also used the same baselines and a shuffle test.

## The practice problem

**Evidence that the models pick up practice:**
- **The shipped first-guess index is backwards.** It correlates −0.50 with KSS (−0.47 within a person). Its two strongest features, rhythm unevenness and backspaces, run opposite to the directions we assumed.
- **v1's biggest weight points the practice way.** It reads smoother rhythm as more tired (−1.26 per SD). Rule cost and backspaces also point the opposite way to tiredness; only pauses, hold time and accuracy point the expected way.
- **An unconstrained first-guess model reverses 4 of the 5 assumed directions.** It reads smooth, accurate typing as tired, and it only beats flat on practised check-ins (the third onwards).
- **Practice alone can produce our numbers.** In simulated worlds with practice and no tiredness signal, our real numbers (skill ≥ 0.2 and concordance ≥ 0.8) appear 14–22% of the time.
- **v1's skill rests on one unstable feature.** Its leave-one-person-out skill depends on rhythm unevenness, and that feature doesn't even agree between the two halves of a single sentence.

**Evidence that typing tracks something beyond the night's trend:**
- **The out-of-order cases.** In 6 cases where the later check-in was less tired, v1 got all 6 right and the clock got none. Within a person, v1's predictions correlate only 0.31 with the clock.
- **Two features point the tiredness way:** more long pauses (+0.91 per SD in v1; +0.45 within a person) and longer key holds.

## How sleepy people type (exploratory)

People fix almost every error: the median share fixed is 100%, and the median number left in the text is 0. When sleepier, they make *fewer* errors, mostly typos:
- within a person, −0.53 for errors and −0.60 for typos, negative for all four teammates
- still −0.36 after removing the trend across check-ins

They also take more pauses longer than a second. So "backspacing less" means slipping less, not fixing less, which looks like more careful typing. Practice could still explain part of it.

## Data notes

- **Four teammates.** The fifth "teammate" in the data is one aborted test session. It's never scored, but it still counts in the team averages; drop it before the next re-export.
- **The baseline isn't alert.** Each person's first two check-ins were at 5–6pm, and three of the four rated themselves 6–7 then. All three dropped to 3 at about 18:45. A person's baseline rating correlates −0.70 with their later average, so it's a poor anchor.
- **Practice is huge early on.** Typing speed roughly doubled over the first three check-ins for some people, and backspaces fell from about 17 to 4 per 100 keys.
- **The app records no naps or wake times.** "After sleep" has to come from gaps of 2+ hours, or from asking the team.
- **Tiredness varies within people, not between them.** Almost all the variation in KSS is within a person: only 0.5–2.5% is between the four teammates.

## How much the data can tell us (simulation, rough)

- **v1 vs v3 is a tie.** Their difference in error is 0.02, against a sampling SD of about 0.3.
- **A believable win needs a big margin.** With 4 people and picking the best of up to 50 variants, a winner has to reach a leave-one-person-out skill of about 0.70 (an error around 0.8–0.9) to beat chance.
- **The morning batch can only separate tiredness from practice if people sleep.** After sleep, frozen v1 predicts lower KSS in 99–100% of simulated fatigue worlds, against 7–19% of practice-only worlds. It can't settle small differences.

## What this means for the product

1. **First try:** drop "your typing looks more tired than X% of our team's check-ins". Instead, show a neutral typing snapshot against the team (speed, how much the rule slows you, corrections, rhythm) and one line: "One try can't tell your typing style from tiredness; try again later and we'll compare you with yourself."
2. **Second try:** add a warm-up sentence that isn't scored. Otherwise the practice gained between attempts 1 and 2 reads as "more tired".
3. **Pitch:**
   - Correct the claims: fewer slips, not fewer fixes; the ordering numbers above; 4 teammates.
   - Lead with the practice discovery. It's a strong story for both G-Research ("what's hidden, and how we found it") and HRT ("we caught our own model learning the wrong thing").

## Frozen for the morning

`frozen/morning_test.py`, with `ks.py` and `ridge_fg5.json`, was frozen at 02:33:45 GMT. The SHA-256 sums are in `frozen/SHA256SUMS`. The test fits on check-ins up to 00:53:51 and scores only later ones:

1. v1, v3, H1 (v1 with the baseline check-ins as extra training rows), the clock and guessing.
2. New out-of-order pairs (the later check-in less tired): how many each model gets right.
3. After 2+ hour gaps: does each model's prediction move with the rating?
4. First try: the shipped index's sign, and ridge_fg5's pass criteria.
5. The nap rule: caught, missed and false alarms.
6. Mistakes: errors made fall with tiredness, and nearly all errors get fixed.

To run it on a new batch:

```bash
cd analysis
uv run python export_features.py --data <new data folder> --out results/<name>
uv run python results/experiments/frozen/morning_test.py results/<name> <new data folder>
```

Then check the files are unchanged: `cd results/experiments/frozen && shasum -a 256 -c SHA256SUMS` (the last line, the freeze time, will report as a bad line; that's expected).

## Files

- `harness.py`: shared data loading, evaluations, baselines, shuffle test and results log
- `results.jsonl`: every logged candidate (73 records)
- `sequence/`, `keystrokes/`, `strategies/`, `first_guess/`, `simulation/`: each track's code, results and README
- `frozen/`: the morning test

## First results on new check-ins (03:40)

Three check-ins arrived after the freeze: P1 at 01:58 (rated 6) and P2 at 01:48 and 03:32 (both rated 9). The fingerprints were checked first (all OK), and fair_test.py is unchanged since db625ee.

**Frozen morning test** (fitted up to 00:53:51, scored on these 3):

| Model | P2 01:48 (9) | P1 01:58 (6) | P2 03:32 (9) | Error |
|---|---|---|---|---|
| v1 | 8.9 | 6.5 | 8.2 | **0.46** |
| H1 | 8.7 | 6.6 | 8.1 | 0.61 |
| v3 (SWELL stress) | 7.0 | 8.7 | 9.0 | 1.56 |
| Clock (benchmark) | 9.0 | 7.8 | 9.0 | 0.60 |
| Guessing | | | | 1.86 |

- **Nap rule at 7+:** v1 caught both 9s, missed none and raised no false alarms. The clock raised one false alarm, on P1's 6.
- **Order beyond the clock:** there are 4 new pairs where a later check-in was less tired. v1 got 3 right and the clock got 0.
- **v3 tied v1 overnight, but not here:** on new data it rated P1's 6 at 8.7.
- **First try for strangers:** ridge_fg5 failed, with an error of 1.97 against 1.96 for flat. The shipped index's sign over only 3 check-ins means nothing.
- **After sleep:** there were no gaps of 2+ hours, so this question is still open.

**Fair test** (fitted up to 00:18:48, scored on the 5 later check-ins): v1 0.96, clock 0.75, v3 1.57, guessing 2.06.

These check-ins were late in the night, and two were rated 9, so the clock does well here too. v1's edge over the clock is the check-ins where someone was *less* tired later. It's still only 3 to 5 check-ins from a few people, so it's encouraging but not proof.

## Correction (04:15): all 7 unseen check-ins

Ananya's full zip added 4 check-ins from 23:03–23:10 on 7 Nov, one from each teammate. They had been missing from the batches we received, so the frozen models never saw them either. Data: keyboard-fatigue/data/real-2026-11-08-0410 (43 check-ins). Features: results/real-2026-11-08-0410. Fingerprints OK.

**As written** (morning_test.py, which by its own rule now trains on 28 check-ins including the 4 recovered ones): on the 3 check-ins after the cut-off, v1 0.57, H1 0.52, clock 0.57, guessing 1.82. The nap rule caught both 9s with no false alarm.

**Strict** (trained on exactly the 24 scored check-ins we had at the freeze, scored on all 7 the models never saw):

| Unseen check-in | KSS | v1 | H1 | v3 | Clock |
|---|---|---|---|---|---|
| P1 23:03 | 8 | 4.2 | 5.3 | 4.0 | 6.4 |
| P4 23:05 | 8 | 7.0 | 6.9 | 7.5 | 7.4 |
| P2 23:08 | 5 | 6.8 | 6.6 | 6.0 | 8.5 |
| P3 23:10 | 8 | 5.3 | 5.1 | 6.8 | 6.5 |
| P2 01:48 | 9 | 8.7 | 8.7 | 7.0 | 9.0 |
| P1 01:58 | 6 | 6.7 | 6.6 | 8.7 | 7.8 |
| P2 03:32 | 9 | 8.3 | 8.1 | 9.0 | 9.0 |

| Model | Error | Skill | Nap rule at 7+ |
|---|---|---|---|
| Guessing | 1.68 | 0 | caught 0, missed 5 |
| Clock (benchmark) | 1.28 | +0.11 | caught 3, missed 2, 2 false alarms |
| v1 | 1.56 | −0.18 | caught 3, missed 2, no false alarms |
| H1 | 1.47 | +0.08 | caught 2, missed 3, no false alarms |
| v3 | 1.62 | −0.30 | caught 3, missed 2, 1 false alarm |

On everything it hadn't seen, v1 performs at about guessing level. It was right about the late-night 9s and badly wrong about two of the 23:00 8s. As overnight warned, its score rests on a shaky feature, and 4 people is too few. The pitch notes now say "promising on the late-night 9s, unproven overall" instead of the earlier 0.46.
