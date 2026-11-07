# keyboard-fatigue

Our HackTheBurgh XIII project: we predict how tired someone is from how they type.

## How it works

People on our overnight panel do a check-in every hour or so while they're awake, plus one just before and one just after sleeping. Everyone uses one shared laptop, so the keyboard is always the same. Each check-in has:

1. A self-report, including sleepiness on the Karolinska Sleepiness Scale (KSS: 1 is extremely alert, 9 is fighting sleep) and when they last woke up.
2. A sentence to type with every vowel (a, e, i, o, u, not y) capitalised, using Shift.
3. A short free-text answer.
4. A 3-minute reaction-time test.

The vowel rule adds mental load. Within the same sentence, we compare the gaps next to a capital with the gaps between ordinary letters. That cancels out each person's own typing speed, and we expect tired people to pay more for the rule.

A few one-off people, such as judges, do one or two check-ins. They're our test set, and we never train on them.

## Running the analysis

The analysis is in `analysis/` and needs [uv](https://docs.astral.sh/uv/).

```bash
cd analysis
uv sync
```

Put the check-in app's session files (one JSON file per check-in) in `data/sessions/` at the top of the repo, then run:

```bash
uv run python run_all.py --data ../data/sessions --out results/latest
```

That writes `report.md`, figures, `predictions.csv` and `features.csv` to `results/latest/` in under a minute. To print each person's latest predicted tiredness, run `predict_latest.py`; add `--json` for the dashboard, sleep rota and push guard:

```bash
uv run python predict_latest.py --data ../data/sessions
```

To try everything without real data:

```bash
uv run python simulate.py
uv run python run_all.py --data ../data/sim/effect --out results/sim_effect
uv run python run_all.py --data ../data/sim/null --out results/sim_null
uv run pytest
```

The simulator writes check-ins in the app's format for two worlds: one where fatigue changes typing and one where it doesn't. Their reports are in `analysis/results/`.

## What the analysis does

- **Features:** typing speed, rhythm, pauses, key hold times, errors, the vowel rule's cost and Shift timing, free-text pauses, and the reaction test's median, speed and lapses. Timings leave out Shift presses and corrections. Each feature is also measured relative to the person's first two check-ins.
- **Models:** ridge regression predicts each person's change in KSS from their own first two check-ins. Every person is predicted by a model trained on everyone else. A logistic regression separates tired check-ins from alert ones.
- **The key test:** over one night, practice, tiredness and the clock all drift together, so we test whether typing adds anything to hours awake. The null comes from shuffling the typing features within each person.

| File | What it does |
|---|---|
| `tasks.py` | 40 checked sentences, the vowel rule and answer scoring; writes `items.json` |
| `load_data.py` | Reads the app's session files |
| `features.py` | One row of features per check-in |
| `model.py` | Models and their evaluation |
| `report.py` | The report, figures and predictions |
| `run_all.py`, `predict_latest.py`, `simulate.py` | The commands above |

`data/` is kept out of git because it holds people's keystrokes. Results from real check-ins stay out of git too.

## Built with

- [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [SciPy](https://scipy.org/), [scikit-learn](https://scikit-learn.org/) and [statsmodels](https://www.statsmodels.org/), under the BSD 3-Clause licence
- [Matplotlib](https://matplotlib.org/), under the Matplotlib licence
- [pytest](https://pytest.org/), under the MIT licence
- [uv](https://docs.astral.sh/uv/), under the MIT or Apache 2.0 licence
- The KSS (Åkerstedt and Gillberg, 1990), and the 3-minute reaction-time test with a 355 ms lapse cut-off (Basner, Mollicone and Dinges, 2011)

## Credits

Claude Code helped build this project. `AI_USAGE.md` records what AI generated and what we wrote ourselves.
