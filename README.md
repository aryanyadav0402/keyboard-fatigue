# keyboard-fatigue

Our HackTheBurgh XIII project: we predict how tired someone is from how they type.

## How it works

People on our overnight panel do a check-in every hour or so while they're awake. Everyone uses one shared laptop, so the keyboard is always the same. Each check-in in the check-in app (`typing_station.zip`) has:

1. Their name and five quick questions, including sleepiness on the Karolinska Sleepiness Scale (KSS: 1 is extremely alert, 9 is fighting sleep).
2. A sentence to retype with a small rule, such as capitalising every vowel or replacing spaces with dashes. The rule rotates between five types, so nobody can learn it.
3. A 20-second reaction-time test.

The rule adds mental load. Within the same sentence, we compare the gaps next to the keys the rule changes (capitals, dashes, dots) with the gaps between ordinary letters. That cancels out each person's own typing speed, and we expect tired people to pay more for the rule.

A few one-off people, such as judges, do one or two check-ins. They're our test set, and we never train on them.

## Running the analysis

The analysis is in `analysis/` and needs [uv](https://docs.astral.sh/uv/).

```bash
cd analysis
uv sync
```

The check-in app saves each check-in as a JSON file under `data/sessions/<NAME>/` on the laptop running it. Copy that `sessions` folder to `data/sessions/` at the top of this repo, then run:

```bash
uv run python run_all.py --data ../data/sessions --out results/latest
```

That writes `report.md`, figures, `predictions.csv`, `features_raw.csv` and `features_processed.csv` to `results/latest/` in under a minute. To print each person's latest predicted tiredness, run `predict_latest.py`; add `--json` for the dashboard, sleep rota and push guard:

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

## Live predictions for the dashboard

`serve.py` runs on the laptop where the check-in app saves its files. It watches the app's `data/sessions` folder, rescores within a few seconds of each new check-in, and serves the results as JSON:

```bash
uv run python serve.py --data /path/to/typing_station/data/sessions
```

| Endpoint | What it returns |
|---|---|
| `GET /api/latest` | Each person's latest check-in: `person`, `participant_type`, `timestamp`, `kss_reported`, `tiredness_pred` (1–9), `tired_prob`, `model`, `nap_suggested` and `push_warning`, most tired first, plus `nap_at` and `push_warning_at` |
| `GET /api/history` | Every check-in with `checkin_id`, `person`, `timestamp`, `kss_reported`, `tiredness_pred`, `tired_prob`, `model`, `is_baseline` and `question_type`, for charts over the night |
| `GET /api/checkin/<checkin_id>` | One check-in, for showing someone their score right after they finish. It returns 404 for a few seconds until the check-in is scored. |
| `GET /api/status` | How many check-ins and people there are, whether the model is ready, and what's missing if it isn't |

It answers on `http://localhost:8765` by default, with CORS headers so a dashboard on another address can read it. `--host 0.0.0.0` shares it with other laptops on the same Wi-Fi, and `cloudflared tunnel --url http://localhost:8765` gives a public HTTPS link, for example for a dashboard on Vercel. After each update it also writes `status.json`, `latest.json` and `history.json` to `results/live/`.

`model` says which model made the prediction:
- `panel model`: a teammate, compared with their own first two check-ins
- `no-baseline model`: someone without a baseline, such as a judge, so it's a rough first guess
- `not enough data yet`: the model needs 3 teammates with 3 check-ins each beyond their first two, and 15 in total, so `tiredness_pred` is empty

These live predictions come from a model trained on everyone, so they look better than the honest numbers in the report, where each person is predicted by a model that never saw them.

## What the analysis does

- **Features:** typing speed, rhythm, pauses, key hold times, errors, the vowel rule's cost and Shift timing, and the reaction test's median, speed and lapses. Timings leave out Shift presses and corrections. `features_raw.csv` has each feature as measured. `features_processed.csv` has each one as a change from the person's first two check-ins, with backspaces per 100 keys left as they are, and the models use its columns.
- **Models:** ridge regression predicts each person's change in KSS from their own first two check-ins, using 6 typing features: the vowel rule's cost, backspaces, accuracy, rhythm unevenness, long pauses and key hold time. Predictions use typing only, because the demo can't ask how long someone has been awake. Every person is predicted by a model trained on everyone else, and a logistic regression separates tired check-ins from alert ones.
- **The clock as a benchmark:** we compare typing alone with the clock alone. The clock is hours awake when the app records a wake-up time, and otherwise hours since the person's first check-in. Over one night, practice, tiredness and the clock all drift together, so a clock check also tests whether typing adds anything to the clock, by shuffling the typing features within each person.
- **Rule types:** some of the app's five rules are harder than others, so each Task 1 feature is first measured against the panel's typical value for that rule. Rule cost and accuracy are divided by it, so 1.0 means normal for that rule.

| File | What it does |
|---|---|
| `tasks.py` | 40 checked sentences, the vowel rule and answer scoring; writes `items.json` |
| `load_data.py` | Reads the app's session files |
| `features.py` | One row of features per check-in |
| `export_features.py` | Writes just the raw and processed feature files |
| `model.py` | Models and their evaluation |
| `report.py` | The report, figures and predictions |
| `serve.py` | Live predictions as JSON for the dashboard |
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
