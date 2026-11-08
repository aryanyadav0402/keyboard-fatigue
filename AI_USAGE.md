# How we used AI

Keep this current while building, and copy it into the submission. The event's Code of Conduct and MLH's recommended rules ask teams to say how they used AI and to be clear about what they made and what was generated.

## Tools

- Claude Code: Aryan uses it for the repo setup and the analysis.
- Codex:

## What AI generated

| Part of the project | Tool | Roughly how much | Who reviewed it |
|---|---|---|---|
| README and repo setup | Claude Code | all of it | not reviewed yet |
| Sentence pool, vowel rule and answer scoring (`analysis/tasks.py`, `items.json`) | Claude Code | all of it | not reviewed yet |
| Reading the check-in app's session files and computing typing, reaction-time and context features (`analysis/load_data.py`, `analysis/features.py`) | Claude Code | all of it | not reviewed yet |
| Simulated check-ins in the app's file format, with and without a fatigue effect (`analysis/simulate.py`) | Claude Code | all of it | not reviewed yet |
| Models, evaluation, report, figures and latest predictions (`analysis/model.py`, `report.py`, `run_all.py`, `predict_latest.py`), tests, and the simulated results | Claude Code | all of it | not reviewed yet |
| Raw and processed feature files measured from each person's first two check-ins (`analysis/export_features.py`, with changes to `features.py`, `model.py` and `report.py`) | Claude Code | all of it | not reviewed yet |
| Typing-only predictions from 6 chosen features, with hours awake as a benchmark (changes to `analysis/model.py`, `report.py`, `run_all.py`) | Claude Code | all of it | not reviewed yet |
| Support for the check-in app's version 1.2 files: five rotating rules scored by their own rule, rule-type adjustment, 20-second reaction tests, a clock without wake-up times, skipped checkpoints; the simulator now writes this format (changes across `analysis/`) | Claude Code | all of it | not reviewed yet |
| Live prediction service for the dashboard: watches the check-in folder and serves JSON (`analysis/serve.py`, with shared prediction code in `model.py` and `predict_latest.py`) | Claude Code | all of it | not reviewed yet |
| Demo scoring: POST /api/score keeps each user's best attempt in SQLite and compares later attempts with it, plus GET /api/best (`analysis/demo.py`, `analysis/serve.py`) | Claude Code | all of it | not reviewed yet |
| First-guess model for people with no baseline: the fatigue index, pauses relative to the person's own pace, and a comparison that predicts each panel person as a stranger (`analysis/first_guess.py`, `analysis/compare_first_guess.py`, with changes to `features.py` and `model.py`) | Claude Code | all of it | not reviewed yet |
| The fatigue index as the demo's headline for people with no baseline: a percentile against the team's check-ins and what drove it (`fatigue_index` in `POST /api/score`, `describe` in `analysis/first_guess.py`) | Claude Code | all of it | not reviewed yet |
| Hosted scoring API on Vercel: trained models exported as plain numbers, a scorer that needs neither the team's data nor scikit-learn, and the page keeping each user's best (`service/`, `analysis/frozen_model.py`, `analysis/export_model.py`, `analysis/attempt.py`, with changes to `features.py`, `first_guess.py` and `demo.py`) | Claude Code | all of it | not reviewed yet |
| A fair test of our model against a single SWELL stress score, fixed before we looked at later check-ins (`analysis/fair_test.py`) | Claude Code | all of it | not reviewed yet |

## What we wrote or decided ourselves

-

## Frameworks and public code

- pandas, NumPy, SciPy, scikit-learn and statsmodels (BSD 3-Clause licence), Matplotlib (Matplotlib licence) and pytest (MIT licence), for the analysis
- uv (MIT or Apache 2.0 licence), to manage the Python environment
- The Karolinska Sleepiness Scale and the 3-minute reaction-time test (PVT-B), published research methods used by the check-in app
- The SWELL knowledge work dataset (Koldijk et al., 2014), used by Ananya's SWELL stress model
