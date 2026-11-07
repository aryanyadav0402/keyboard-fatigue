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

## What we wrote or decided ourselves

-

## Frameworks and public code

- pandas, NumPy, SciPy, scikit-learn and statsmodels (BSD 3-Clause licence), Matplotlib (Matplotlib licence) and pytest (MIT licence), for the analysis
- uv (MIT or Apache 2.0 licence), to manage the Python environment
- The Karolinska Sleepiness Scale and the 3-minute reaction-time test (PVT-B), published research methods used by the check-in app
