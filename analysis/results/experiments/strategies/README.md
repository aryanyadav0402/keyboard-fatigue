# Training strategies for tiny data (strategies track, 8 Nov)

**Bottom line: nothing beats v1 in a way that survives the permutation test.**

- **Ordering** (within-person concordance, what a nap rule needs): none of the 49 variants beat v1's
  LOPO concordance without losing rolling concordance. The largest LOPO gain is 2.5 of 49 pairs (F2),
  and it loses 2 of 13 rolling pairs.
- **The KSS number (MAE)**: one change helped consistently. Adding each panel person's two baseline
  check-ins as extra training rows improved MAE in 13 of 14 base-versus-augmented comparisons.
- **Choosing by CV**: picking a strategy by cross-validation (nested selection over the whole menu) did
  *worse* than v1, so the best rows below are optimistic.
- **Permutation test**: every candidate passes on its own (LOPO skill p = 0.005, the smallest possible
  with 200 shuffles; v1's is 0.020). But no gain over v1 survives: with KSS shuffled within each
  person, the candidates beat v1 by about as much as on the real labels (gain-over-v1 p 0.12 to 0.89).
  The gains come from how each model sets a person's overall level, which here mostly means P2, not
  from tracking tiredness better within a person. I wouldn't replace v1.

## Setup

- Data, schemes and metrics: the shared harness (`../harness.py`), unchanged. 24 scored check-ins from
  4 panel people; LOPO (4 folds) and rolling (14 predictions); nothing after 00:53:51.
- Tuning: every hyperparameter is chosen by leave-one-*training*-person-out CV inside the training
  rows (`common.tuned`), never on the test person or time.
- Inputs: v1's 6 typing features (`model.TYPING`) and the baseline rating, as v1 uses them. The
  check-in number appears only as a training-time nuisance covariate (E1, E2) or in rows marked
  NOT deployable.
- **n_variants = 49** (the 4 baselines aren't counted): 39 planned before any result, 5 combinations
  declared after round 1 (R2a–R2e, so data-informed), 2 nested selections, 3 level-only references.
  Every one is in the table.

## What I tried

| group | variants | idea |
|---|---|---|
| A anchoring | A1–A4 | unanchored (a=0); a ∈ {0, .25, .5, .75, 1} tuned in folds; baseline rating as a ridge feature; fixed a=0.5. pred = a·kss_base + f(x), f fitted on kss − a·kss_base |
| B pairwise | B1–B5 | logistic on within-person feature differences (label: which check-in had higher KSS), symmetric pairs, no intercept. Score standardized; KSS = a·kss_base + c + b·score with b ≥ 0 and c fitted on inner leave-one-person-out scores. C=1, C tuned on inner concordance, unanchored mapping, pairs weighted by abs(ΔKSS), baseline rows added to the pairs |
| B within-person | B6–B7 | the linear cousin of pairwise: ridge slopes from person-demeaned features and targets (fixed effects), with a pooled intercept |
| C ordinal | C1–C3 | penalized cumulative-logit model (own implementation, analytic gradient, alpha tuned) on KSS and on kss_delta; statsmodels `OrderedModel` on the out-of-fold ridge score |
| D regularization | D1–D7 × {anchored, unanchored} | lasso and elastic net (tuned), PLS with 1 and 2 components, Bayesian ridge, ridge with alpha by inner leave-one-person-out instead of LOO, top-k features (k ∈ {1,2,3,6}) tuned |
| E practice | E1–E4 | features residualized within person on log(check-in number), with a pooled slope estimated on training rows plus those people's baseline check-ins. E1/E2 predict from raw test features (deployable); E3/E4 also residualize the test features (NOT deployable, upper bound) |
| F composite | F1–F2 | train on mean(z(KSS), z(mental fatigue)), as changes from baseline (F1) or absolute (F2); map back to KSS with the training regression of KSS on the composite; evaluate on KSS |
| G ensembles | G1–G2 | mean(v1, v3); mean(v1, unanchored v1) |
| H augmentation | H1–H2 | add each panel person's two baseline check-ins (training people only in LOPO; past ones in rolling) as training rows: anchored and unanchored ridge |
| R reference | R1 | NOT deployable: unanchored ridge plus a shrunk offset from the person's own earlier KSS ratings |
| round 2 | R2a–R2e | baseline rows added to the ordinal, within-person and pairwise models; mean(H1, v3) |
| nested | N1–N2 | in each outer fold, score the 36 deployable round-1 strategies plus v1 and v3 (38) by inner leave-one-person-out MSE (N1) or concordance (N2), use the winner |
| level only | Z1–Z3 | no typing: kss_base + mean change; the same with baseline rows in the mean; the group mean with baseline rows |

## Full results

Bold = better than v1 on that metric. Concordance = share of same-person pairs with different KSS put
in the right order (LOPO: 49 pairs, so one pair = 0.020; rolling: 13 pairs, one pair = 0.077).

| variant | LOPO MAE | LOPO skill | LOPO conc | rolling MAE | rolling skill | rolling conc |
|---|---:|---:|---:|---:|---:|---:|
| guess | 1.595 | 0.000 | 0.500 | 1.710 | 0.000 | 0.615 |
| clock | 1.791 | -0.325 | 0.878 | 1.559 | -0.144 | 0.423 |
| v1 | 1.383 | 0.217 | 0.837 | 1.107 | 0.403 | 0.923 |
| v3 | 1.399 | 0.138 | 0.653 | 1.150 | 0.422 | 0.692 |
| A1 unanchored v1 (a=0) | 1.454 | 0.148 | 0.837 | 1.302 | 0.298 | 0.846 |
| A2 partial anchor, a tuned | 1.568 | 0.060 | 0.837 | 1.248 | 0.356 | 0.923 |
| A3 kss_base as ridge feature | 1.561 | 0.037 | 0.837 | 1.379 | 0.262 | 0.692 |
| A4 half anchor (a=0.5) | **1.336** | **0.257** | 0.837 | 1.205 | 0.365 | 0.923 |
| B1 pairwise C=1, anchored | **1.378** | 0.217 | 0.837 | 1.522 | 0.103 | 0.769 |
| B2 pairwise C tuned (conc), anchored | **1.377** | **0.233** | 0.837 | 1.515 | 0.129 | 0.846 |
| B3 pairwise C=1, unanchored | **1.173** | **0.422** | 0.837 | 1.221 | 0.397 | 0.769 |
| B4 pairwise ¦dKSS¦-weighted, anchored | 1.465 | 0.141 | 0.837 | 1.725 | -0.093 | 0.769 |
| B5 pairwise + baseline rows, anchored | **1.250** | **0.277** | 0.816 | **0.998** | **0.575** | 0.846 |
| B6 within-person ridge, anchored | **1.368** | **0.324** | 0.816 | 1.536 | 0.133 | 0.769 |
| B7 within-person ridge, unanchored | **1.022** | **0.517** | 0.816 | 1.242 | 0.305 | 0.769 |
| C1 ordinal on KSS, alpha tuned | **1.372** | **0.261** | 0.837 | **1.055** | **0.497** | 0.846 |
| C2 ordinal on kss_delta, alpha tuned | **1.278** | **0.327** | **0.857** | **0.947** | **0.589** | 0.769 |
| C3 OrderedModel on OOF ridge score | **1.300** | **0.264** | 0.837 | 1.622 | 0.041 | 0.846 |
| D1 lasso tuned, anch | **1.349** | **0.276** | **0.857** | 1.110 | **0.422** | 0.846 |
| D2 elastic net tuned, anch | **1.344** | **0.266** | 0.816 | 1.167 | 0.374 | 0.846 |
| D3 PLS 1 comp, anch | **1.379** | **0.232** | 0.816 | 1.253 | 0.394 | 0.846 |
| D4 PLS 2 comp, anch | **1.311** | **0.292** | 0.816 | 1.119 | **0.441** | 0.846 |
| D5 Bayesian ridge, anch | **1.361** | **0.219** | 0.816 | 1.129 | **0.414** | 0.769 |
| D6 ridge, alpha by inner LOPO, anch | 1.517 | 0.163 | 0.796 | 1.145 | **0.410** | 0.769 |
| D7 top-k features tuned, anch | **1.379** | 0.156 | 0.796 | 1.195 | 0.341 | 0.769 |
| D1 lasso tuned, unanch | 1.469 | -0.057 | 0.724 | 1.211 | 0.329 | 0.692 |
| D2 elastic net tuned, unanch | **1.382** | 0.061 | 0.816 | 1.307 | 0.293 | 0.692 |
| D3 PLS 1 comp, unanch | **1.377** | 0.151 | 0.816 | 1.264 | 0.315 | 0.923 |
| D4 PLS 2 comp, unanch | **1.358** | **0.289** | 0.837 | 1.134 | 0.373 | 0.769 |
| D5 Bayesian ridge, unanch | 1.396 | 0.184 | 0.837 | 1.248 | 0.337 | 0.846 |
| D6 ridge, alpha by inner LOPO, unanch | 1.580 | 0.057 | 0.796 | 1.258 | 0.343 | 0.846 |
| D7 top-k features tuned, unanch | 1.572 | 0.027 | 0.796 | 1.327 | 0.261 | 0.923 |
| E1 practice-residualized, anch | 1.634 | -0.211 | 0.796 | 1.220 | 0.298 | 0.923 |
| E2 practice-residualized, unanch | 1.911 | -0.418 | 0.806 | 1.341 | 0.230 | 0.615 |
| E3 practice-resid. test too, anch (NOT deployable) | 1.680 | -0.279 | 0.776 | 1.297 | 0.196 | 0.923 |
| E4 practice-resid. test too, unanch (NOT deployable) | 1.907 | -0.438 | 0.786 | 1.340 | 0.228 | 0.692 |
| F1 composite KSS+MF label, anch | 1.492 | 0.099 | 0.837 | 1.240 | 0.334 | 0.846 |
| F2 composite KSS+MF label, unanch | **1.349** | 0.100 | **0.888** | 1.287 | 0.301 | 0.769 |
| G1 mean(v1, v3) | **1.224** | **0.315** | 0.776 | **1.042** | **0.549** | 0.923 |
| G2 mean(v1, unanchored v1) | **1.325** | **0.259** | 0.837 | 1.202 | 0.372 | 0.923 |
| H1 v1 + baseline rows, anch | **1.119** | **0.473** | 0.816 | **1.036** | **0.502** | 0.769 |
| H2 unanchored + baseline rows | **1.127** | **0.334** | 0.816 | 1.291 | 0.355 | 0.923 |
| R1 person offset from own past KSS (NOT deployable) | 1.454 | 0.148 | 0.837 | 1.326 | 0.295 | 0.923 |
| R2a ordinal kss_delta + baseline rows, alpha tuned | **1.025** | **0.556** | 0.837 | **0.923** | **0.607** | 0.769 |
| R2b mean(H1 v1+baseline rows, v3) | **1.138** | **0.428** | 0.776 | **1.020** | **0.571** | 0.846 |
| R2c within-person ridge + baseline rows, anch | **1.131** | **0.477** | 0.816 | **1.075** | **0.495** | 0.769 |
| R2d within-person ridge + baseline rows, unanch | **1.180** | **0.406** | 0.816 | **0.944** | **0.533** | 0.769 |
| R2e pairwise + baseline rows, unanch | **1.012** | **0.518** | 0.816 | **0.867** | **0.643** | 0.846 |
| N1 nested selection by inner MSE | 1.795 | -0.234 | 0.745 | 1.146 | 0.385 | 0.846 |
| N2 nested selection by inner concordance | **1.217** | **0.329** | 0.837 | 1.229 | 0.383 | 0.846 |
| Z1 anchor only: kss_base + mean change | 2.116 | -0.719 | 0.500 | 2.115 | -0.537 | 0.538 |
| Z2 anchor only, mean change incl. baseline rows | 2.090 | -0.623 | 0.500 | 2.132 | -0.555 | 0.615 |
| Z3 guess incl. baseline rows | 1.587 | 0.053 | 0.500 | 1.719 | -0.008 | 0.615 |

49 rows besides the baselines. 10 beat v1 on MAE in both schemes; none beats v1 on LOPO concordance without losing rolling concordance.

## Permutation tests (n = 200, within-person KSS shuffles)

Observed value (permutation p). All seven are the same within-person shuffles (seed 0, the harness's scheme). The LOPO skill column equals `harness.perm_test` and is asserted in `run_perm.py`.

| model | LOPO MAE | LOPO skill | LOPO conc | rolling MAE | rolling skill | rolling conc |
|---|---:|---:|---:|---:|---:|---:|
| v1 | 1.383 (0.030) | 0.217 (0.020) | 0.837 (0.005) | 1.107 (0.015) | 0.403 (0.010) | 0.923 (0.005) |
| H1 v1 + baseline rows (anchored ridge) | 1.119 (0.005) | 0.473 (0.005) | 0.816 (0.005) | 1.036 (0.010) | 0.502 (0.005) | 0.769 (0.040) |
| C2 ordinal on kss_delta, alpha tuned | 1.278 (0.010) | 0.327 (0.005) | 0.857 (0.005) | 0.947 (0.010) | 0.589 (0.005) | 0.769 (0.045) |
| G1 mean(v1, v3) | 1.224 (0.010) | 0.315 (0.005) | 0.776 (0.005) | 1.042 (0.015) | 0.549 (0.005) | 0.923 (0.005) |
| R2a ordinal on kss_delta + baseline rows | 1.025 (0.005) | 0.556 (0.005) | 0.837 (0.005) | 0.923 (0.005) | 0.607 (0.005) | 0.769 (0.035) |
| R2e pairwise + baseline rows, unanchored | 1.012 (0.005) | 0.518 (0.005) | 0.816 (0.005) | 0.867 (0.010) | 0.643 (0.005) | 0.846 (0.015) |
| B7 within-person ridge, unanchored | 1.022 (0.005) | 0.517 (0.005) | 0.816 (0.005) | 1.242 (0.030) | 0.305 (0.005) | 0.769 (0.025) |

Gain over v1 on the real labels, mean gain over v1 on shuffled labels, and the share of shuffles with a gain at least as large (p). MAE gains are negative when better:

| model | LOPO MAE | LOPO skill | LOPO conc | rolling MAE | rolling skill | rolling conc |
|---|---:|---:|---:|---:|---:|---:|
| H1 v1 + baseline rows (anchored ridge) | -0.264 / -0.143 (p 0.19) | +0.256 / +0.288 (p 0.51) | -0.020 / -0.001 (p 0.69) | -0.071 / -0.040 (p 0.28) | +0.099 / +0.132 (p 0.51) | -0.154 / +0.013 (p 0.95) |
| C2 ordinal on kss_delta, alpha tuned | -0.105 / +0.046 (p 0.32) | +0.110 / -0.138 (p 0.37) | +0.020 / +0.002 (p 0.45) | -0.160 / -0.043 (p 0.21) | +0.186 / +0.085 (p 0.23) | -0.154 / -0.014 (p 0.91) |
| G1 mean(v1, v3) | -0.159 / -0.157 (p 0.42) | +0.098 / +0.230 (p 0.61) | -0.061 / +0.008 (p 0.81) | -0.065 / -0.118 (p 0.67) | +0.146 / +0.146 (p 0.47) | +0.000 / +0.013 (p 0.68) |
| R2a ordinal on kss_delta + baseline rows | -0.358 / -0.078 (p 0.12) | +0.339 / +0.087 (p 0.43) | +0.000 / -0.002 (p 0.56) | -0.184 / -0.085 (p 0.25) | +0.205 / +0.208 (p 0.44) | -0.154 / +0.015 (p 0.92) |
| R2e pairwise + baseline rows, unanchored | -0.371 / -0.479 (p 0.72) | +0.301 / +0.724 (p 0.89) | -0.020 / +0.009 (p 0.67) | -0.240 / -0.229 (p 0.50) | +0.240 / +0.367 (p 0.68) | -0.077 / +0.005 (p 0.73) |
| B7 within-person ridge, unanchored | -0.361 / -0.259 (p 0.45) | +0.300 / +0.377 (p 0.68) | -0.020 / +0.001 (p 0.62) | +0.135 / +0.223 (p 0.45) | -0.097 / -0.495 (p 0.41) | -0.154 / +0.031 (p 0.86) |

How to read the gain table: within-person shuffles keep each person's average KSS and scramble the order of their check-ins. A gain over v1 that also appears on shuffled labels therefore comes from person-level calibration (mostly P2's level), not from better within-person signal. For every candidate the mean gain on shuffled labels is similar to or larger than the real gain, and no gain-over-v1 p is below 0.12. On concordance no candidate gains more than 1 pair, and no difference is distinguishable from shuffling (p 0.45 to 0.95). In the baseline-row variants (H1, R2a, R2e) the baseline rows keep their real ratings in the shuffled runs, which makes their gain test conservative. But C2, G1 and B7 have no such rows and show the same pattern.

## Why unanchored beat anchored, and why it no longer does

- **It was never stable.** The earlier report's result reproduces on the first 22 scored check-ins
  (LOPO MAE: unanchored 1.215, anchored 1.280, with today's feature files), but reverses on all 24
  (1.454 against 1.383). Two late check-ins (P3 at KSS 8, P4 at KSS 9) flipped it.
- **The baseline rating is a poor anchor in this sample.** Across the 4 people, the baseline rating
  correlates −0.70 with the person's later mean KSS (P2: baseline 7, later mean 5.0; P3: baseline 5,
  later mean 7.3). The anchor alone (kss_base plus the average change, no typing: Z1) has LOPO MAE
  2.12, worse than guessing the group mean (1.60).
- **Subtracting it triples the person-level variance of the target**: between-person variance 0.74 for
  KSS but 2.26 for kss_delta, while within-person variance is 2.26 for both. The anchored model has to
  explain that extra person-level variance with typing.
- **The typing features carry person-level offsets too.** P3's gap CV sat far below their baseline
  on every check-in (mean −0.76) and pushes P3's predicted change up by about 2.3 in both versions.
  `vt_accuracy_dev` is nonzero almost only for P1, so it acts as a person indicator, and
  `backspace_per100_adj` isn't baseline-relative at all. Whether anchoring helps then depends on how the
  person-level parts of the features and of the baseline rating line up for the held-out person:
  anchored wins for P3 (0.40 against 1.02) and loses for P2 (2.38 against 1.61). That is a
  between-person question estimated from 3 people per fold.
- **How much to trust the anchor can't be learned here.** Tuning a inside the folds (A2) picks a = 0
  whenever P2 is among the training people and a = 0.75 when P2 is held out. That is the wrong choice
  for each held-out person, and gives LOPO MAE 1.568, worse than either extreme. A fixed a = 0.5 (A4)
  lands between them (LOPO 1.336, rolling 1.205).

## Other findings

- **Baseline check-ins as training rows** (H1, H2, B5, R2a–R2e) improved MAE in 13 of 14 paired
  comparisons. The exception is within-person ridge unanchored in LOPO (1.022 → 1.180). It's not just
  a smaller intercept: level-only Z2 is no better than Z1. For P2, where most of H1's LOPO gain lies,
  H1's predicted average change is −0.66 against v1's +0.41 (actual −2.0). Roughly a third comes from
  the smaller intercept, a third from a stronger gap-CV slope (RidgeCV picks less shrinkage with 8 more
  rows), and a third from the accuracy coefficient changing sign. There's no single clean mechanism.
  Concordance doesn't improve: LOPO 41 → 40 of 49 pairs, rolling 12 → 10 of 13 for H1. The permutation
  test says the gain is mostly person-level calibration: it largely appears on shuffled labels too (LOPO
  skill gain over v1 +0.26 on real labels, +0.29 on average when shuffled).
- **Nested selection does worse than v1.** By inner MSE (N1): LOPO skill −0.234, rolling 0.385, with
  a different strategy chosen in every LOPO fold. By inner concordance (N2): LOPO 0.329, rolling 0.383.
  With 3 training people, CV can't tell these strategies apart.
- **v1's ordering is not just the night's trend.** In LOPO there are 6 same-person pairs where the
  later check-in was *less* tired. v1 and every candidate I checked except v3 get all 6 right; the
  clock gets 0.
  v1's predictions correlate only 0.31 with the clock within person, and in rolling it gets 5 of 5
  such pairs. Caveat: 4 of the 6 LOPO pairs come from P2 (`beyond_clock.txt`).
- **Practice removal hurts** (E1/E2 LOPO skill −0.21 and −0.42), even in the NOT deployable version
  that also corrects the test features. No naps were recorded, so nothing in the data separates
  practice from tiredness: removing the check-in trend from the features removes the signal along with
  any practice.
- **Pairwise ranking** gets the same LOPO concordance as v1 (41 of 49 pairs) and adds nothing to it.
  Mapped without the anchor (B3), it helps LOPO MAE (1.173) but loses rolling pairs.
- **Ordinal regression on kss_delta** (C2) beats v1 on MAE in both schemes (1.278, 0.947). LOPO
  ordering gains 1 pair and rolling loses 2.
- **Composite label**: no MAE gain. Unanchored F2 has the best LOPO concordance of any deployable row
  (0.888, +2.5 pairs) but loses 2 rolling pairs.
- **Regularization** (lasso, elastic net, PLS, Bayesian ridge, top-k) lands within noise of v1.
  Choosing the ridge alpha by grouped CV instead of LOO is worse (D6: LOPO 1.517).

## What I'd freeze

**Nothing as a replacement for v1.** Each candidate has signal on its own, but none of its gains over
v1 needs the real within-person order of KSS, and none orders a person's check-ins better. Keep v1 as the
main model and for the nap rule.

If you want one pre-registered challenger scored *next to* v1 on the morning check-ins (offline, not in
the app), the only one I'd name is **H1: v1 with the baseline check-ins as extra training rows**. It was
planned before any result, it changes one thing, and the augmentation pattern held in 13 of 14 model
families. Those 13 aren't independent, though: they share the same 8 rows and the same P2 effect.
Expect at best a level (MAE) gain, not better nap decisions. Exact spec:

- Training rows: every scored panel check-in at freeze time (`harness.scored_panel`: panel, not a
  baseline, KSS rated, typing present), **plus** both baseline check-ins of every panel person who has
  two. Their kss_delta = kss − kss_base (± half the gap between their two baseline ratings), with their
  usual `_dev` features.
- Features: `model.TYPING`, unchanged. Target: kss_delta. Estimator: `model.ridge()`, unchanged (median
  imputer, StandardScaler, RidgeCV over logspace(−2, 3, 30) with its default leave-one-out alpha).
- Prediction: kss_base + ridge prediction, clipped to 1–9. Code: `common.lin(a=1.0, aug=True)`.
- Fitted on all 32 rows today (24 scored + 8 baseline; standardized coefficients): gap CV −1.218,
  pause share +0.859, hold +0.368, accuracy +0.143, backspace −0.106, rule cost −0.003; intercept
  +0.562; alpha 2.59. v1 on the same 24 scored rows: −1.257, +0.906, +0.381, −0.064, −0.184, −0.078;
  intercept +0.750; alpha 2.59.
- Morning comparison, decided now: H1 against v1 on MAE and skill, plus within-person concordance, on
  check-ins after 00:53:51 only. Call H1 better only if it wins on MAE without losing concordance.

## Caveats

- 24 scored check-ins from 4 people. LOPO has 4 folds and rolling has 14 predictions from the same 4
  people. P2 dominates LOPO MAE, so most MAE gains are person-level calibration for P2.
- 49 variants were tried; the best of them is optimistic by construction. R2a–R2e were declared after
  seeing round 1. The nested selection (N1, N2) is the honest estimate of choosing among them, and it
  loses to v1.
- The permutation test shuffles KSS among each person's scored check-ins. It can't rule out practice;
  only the time-discordant pairs above speak to that, and there are 6 of them. For the
  baseline-row variants, the baseline rows keep their real ratings in the null runs.
- The paired comparisons against v1 (`paired_vs_v1.csv`) flip signs per row, but rows aren't
  independent, so treat those p-values as rough. Per person, nothing beats v1 for all 4 people.
- Not run: a permutation test of the composite label (it would also need mental fatigue shuffled);
  extra features (another track).

## Files and how to rerun

Run from this folder with `uv run --project /Users/aryan/Projects/hacktheburgh/keyboard-fatigue-nobaseline/analysis python <script>`.

- `common.py`: every model builder. `run_variants.py`: round 1, writes `results_round1.csv` and
  `tuned_choices_round1.json` (20 s). `run_round2.py`: round 2, nested selection and paired tests, writes
  `results_round2.csv`, `nested_choices.json` and `paired_vs_v1.csv` (3 min). `diagnose.py`: anchoring,
  per-person errors and coefficients, writes `level_only_refs.csv`. `beyond_clock.py`: time-discordant
  pairs, writes `beyond_clock.txt`. `run_perm.py 49`: permutation tests, writes `perm_results.json` and
  `perm_log.txt`, and logs each candidate to `../results.jsonl` (about 4 min wall time with 3 worker
  processes; the two ordinal models take about 3.5 min each).
  `make_table.py` prints the results table above.
