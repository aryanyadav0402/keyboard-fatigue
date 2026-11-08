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
| `POST /api/score` | Scores one attempt from the demo page, compared with that user's best attempt (see below) |
| `GET /api/best/<username>` | A demo user's best attempt so far |

It answers on `http://localhost:8765` by default, with CORS headers so a dashboard on another address can read it. `--host 0.0.0.0` shares it with other laptops on the same Wi-Fi, and `cloudflared tunnel --url http://localhost:8765` gives a public HTTPS link, for example for a dashboard on Vercel. After each update it also writes `status.json`, `latest.json` and `history.json` to `results/live/`.

`model` says which model made the prediction:
- `panel model`: a teammate, compared with their own first two check-ins
- `no-baseline model`: someone without a baseline, such as a judge, so it's a rough first guess
- `not enough data yet`: the model needs 3 teammates with 3 check-ins each beyond their first two, and 15 in total, so `tiredness_pred` is empty

These live predictions come from a model trained on everyone, so they look better than the honest numbers in the report, where each person is predicted by a model that never saw them.

### Scoring the demo page: `POST /api/score`

On the demo page, someone types one sentence with a rule and is told how tired they are. The page posts that typing to `POST /api/score` with a username and gets a score straight back:

- **First attempt:** a first guess from the no-baseline model. That attempt is kept as the user's best.
- **Later attempts:** compared with their best. The model scores how much more tired the typing looks than at their best, and adds that to their best score. Typing just like their best gives the same score, and a more alert attempt becomes the new best.

Only each user's best attempt is stored (its score and features, never keystrokes), in a small SQLite file: `results/demo_users.sqlite`, or wherever `--db` points. Demo attempts never become training data; only the check-in app's files are used for training. `GET /api/best/<username>` returns a user's best so far, or 404 before their first attempt.

Send:

```json
{
  "username": "Zoe O'Neil",
  "sentence": "a curious raccoon found a shiny golden key",
  "rule": "capitalize_vowels",
  "typed": "A cUrIOUs rAccOOn fOUnd A shIny gOldEn kEy",
  "shown_at": 1532.4,
  "keys": [
    {"type": "down", "key": "Shift", "code": "ShiftLeft", "t": 1840.2},
    {"type": "down", "key": "A", "code": "KeyA", "t": 2011.7},
    {"type": "up", "key": "A", "code": "KeyA", "t": 2090.3}
  ]
}
```

- `username` is matched however it's typed, so `Zoe` and `zoe` are the same user. Without it, every attempt gets a first guess and nothing is stored.
- `rule` defaults to `capitalize_vowels`, and any of the app's five rules works.
- `shown_at` is `performance.now()` when the sentence appeared, and each key's `t` is the browser's `event.timeStamp`.
- `reaction_times_ms`, a list of numbers, is optional.

Recording the keys on the page:

```js
const keys = [];
const shownAt = performance.now();
box.addEventListener("keydown", e => { if (!e.repeat) keys.push({ type: "down", key: e.key, code: e.code, t: e.timeStamp }); });
box.addEventListener("keyup", e => keys.push({ type: "up", key: e.key, code: e.code, t: e.timeStamp }));

// when they finish:
const response = await fetch("http://localhost:8765/api/score", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ username, sentence, rule: "capitalize_vowels", typed: box.value, shown_at: shownAt, keys }),
});
const result = await response.json();
```

The reply has:
- `score`: tiredness on the KSS from 1 to 9, or `null` if the team data isn't enough yet
- `label`: the KSS wording for that score, such as "Some signs of sleepiness"
- `level`: `alert` (KSS 1–5), `getting tired` (6) or `tired` (7–9), matching the label
- `model` and `note`: `best-attempt model` (compared with their best), `no-baseline model` (a first guess) or `not enough data yet`, with a sentence the page can show
- `fatigue_index`: how this attempt compares with the team's check-ins. It's only filled in for `no-baseline model` results, where it should be the headline rather than `score`; anyone with a baseline is compared only with themselves, so it's `null` for them. It holds `percentile` (their typing looks more tired than this percentage of the team's check-ins), a ready-made `text` sentence, `value` (0 = typical, higher = more tired-looking) and `drivers`, the five features, biggest effect first. Each driver has `label` (such as "pauses" or "uneven rhythm"), `pushes` ("more tired" or "less tired") and `strength` ("clearly", "a bit" or "about typical"). `z`, `you` and `typical` are the numbers behind them, adjusted for the rule type. The field is `null` until the team data is enough.
- `features`: what was measured from this typing, such as wpm, rule cost, rhythm unevenness, accuracy and backspaces
- `scaled_features`: each feature divided by the same feature in their best attempt, so 1.2 means 20% higher than at their best; `null` on a first attempt
- `best`: their best so far after this attempt: `score`, `attempts`, `new_best` (true if this attempt just became the best) and `best_at`
- `typical_for_this_rule`: the team's typical value for the same features and rule, for "you vs typical"
- `model_inputs`: the exact numbers the model used
- `saved_as_training_data`: always `false`

Errors come back with an `error` message: 400 if the body isn't JSON, and 422 if there isn't enough typing to score (fewer than 5 key presses in the sentence).

To try the endpoint without the UI, start `serve.py` on the simulated night and post the example request from the `analysis` folder. It's a simulated person's attempt at the vowel rule, with the username `test-user`: the first post is a first guess with a `fatigue_index`, and repeating it compares with that best attempt.

```bash
curl -X POST -H "Content-Type: application/json" --data @examples/score_request.json http://localhost:8765/api/score
```

## Hosted scoring for the demo page

The demo page calls a hosted copy of `POST /api/score` at **https://keyboard-fatigue-api.vercel.app**, so it doesn't need this laptop. The API serves frozen models: plain numbers exported from a training run, with no names, timestamps or keystrokes. It retrains nothing, stores nothing and never logs requests.

| Endpoint | What it does |
|---|---|
| `POST /api/score` | Scores one attempt. The request and reply match the local `POST /api/score` above, and the reply adds `best_reference` and `model_version` |
| `GET /api/models` | Lists the models it serves, which one is the default, what each was trained on and what it can do |

How it differs from the local server:
- **The page keeps each user's best.** A reply for someone with a `username` includes `best_reference`, their best attempt so far, as numbers. Save it for that username, for example in `localStorage`, and send it back as `best_reference` with their next attempt. Without it, every attempt counts as a first attempt. A `best_reference` from a different model is ignored, so everyone starts again after a model update.
- **Choose a model** with `?model=<name>` or `"model"` in the body. Otherwise the API uses the default, the latest export from our real check-ins. `sim-test` was trained on a simulated night and its scores vary, so use it to try the UI. `model_version` in the reply says which model answered.
- There's no `GET /api/best/<username>`, because the page already holds the best in `best_reference`.
- Errors: 400 if the body isn't JSON, 404 for an unknown model, 413 if the body is over 2 MB, 422 if there isn't enough typing to score, and 500 for anything else.

```js
const API = "https://keyboard-fatigue-api.vercel.app";
const saved = "best:" + username.trim().toLowerCase();
const best_reference = JSON.parse(localStorage.getItem(saved) || "null");
const response = await fetch(`${API}/api/score`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ username, sentence, rule, typed: box.value, shown_at: shownAt, keys, best_reference }),
});
const result = await response.json();
if (response.ok && result.best_reference) localStorage.setItem(saved, JSON.stringify(result.best_reference));
```

To update the models, export one from the latest check-ins, copy the code into `service/lib/` and deploy from the Vercel CLI:

```bash
cd analysis
uv run python export_model.py --data ../data/sessions --name team-2026-11-08-0900 --default
uv run python ../service/build.py
cd ../service && vercel deploy --prod
```

`export_model.py` saves the model to `service/models/` and lists it in `service/models/index.json`, and `--default` makes it the default. To try the API on this laptop before deploying, run `uv run python ../service/dev_server.py` from `analysis/`; it serves the API at `http://localhost:8766`.

## What the analysis does

- **Features:** typing speed, rhythm, pauses, key hold times, errors, the vowel rule's cost and Shift timing, and the reaction test's median, speed and lapses. Timings leave out Shift presses and corrections. `features_raw.csv` has each feature as measured. `features_processed.csv` has each one as a change from the person's first two check-ins, with backspaces per 100 keys left as they are, and the models use its columns.
- **Models:** ridge regression predicts each person's change in KSS from their own first two check-ins, using 6 typing features: the vowel rule's cost, backspaces, accuracy, rhythm unevenness, long pauses and key hold time. Predictions use typing only, because the demo can't ask how long someone has been awake. Every person is predicted by a model trained on everyone else, and a logistic regression separates tired check-ins from alert ones.
- **The clock as a benchmark:** we compare typing alone with the clock alone. The clock is hours awake when the app records a wake-up time, and otherwise hours since the person's first check-in. Over one night, practice, tiredness and the clock all drift together, so a clock check also tests whether typing adds anything to the clock, by shuffling the typing features within each person.
- **Rule types:** some of the app's five rules are harder than others, so each Task 1 feature is first measured against the panel's typical value for that rule. Rule cost and accuracy are divided by it, so 1.0 means normal for that rule.
- **First guess for strangers:** someone with no baseline, such as a judge, gets the fatigue index (`first_guess.py`). It takes five features from their one attempt (the rule's cost, rhythm unevenness, pauses relative to their own pace, backspaces and accuracy), scores each against the panel, averages them with directions fixed in advance, and turns the result into KSS with two numbers fitted on the panel. `compare_first_guess.py` predicts each panel person as if they were a stranger, to compare it with the alternatives. On six simulated nights it was the most accurate for walk-ups (an error of 1.63 KSS points, against 2.26 for guessing the average), and the only option that did no worse than guessing when there was nothing to find.

| File | What it does |
|---|---|
| `tasks.py` | 40 checked sentences, the vowel rule and answer scoring; writes `items.json` |
| `load_data.py` | Reads the app's session files |
| `features.py` | One row of features per check-in |
| `export_features.py` | Writes just the raw and processed feature files |
| `model.py` | Models and their evaluation |
| `first_guess.py` | The fatigue index, the first guess for people with no baseline |
| `compare_first_guess.py` | Compares first-guess models, predicting each panel person as a stranger |
| `report.py` | The report, figures and predictions |
| `serve.py` | Live predictions as JSON for the dashboard |
| `demo.py`, `attempt.py` | Scores one demo attempt for `POST /api/score` |
| `frozen_model.py`, `export_model.py` | Exports a trained model as plain numbers and scores with it, for the hosted API in `service/` |
| `fair_test.py` | Tests our model against a single SWELL stress score on check-ins after 00:18 on 8 Nov, fixed before we looked at any of them |
| `results/experiments/` | The overnight experiments of 8 Nov: a shared test harness, five experiment tracks and the frozen morning test. Start with `FINDINGS.md`. Only the code and write-ups are in git, not results from real check-ins |
| `run_all.py`, `predict_latest.py`, `simulate.py` | The commands above |

`data/` is kept out of git because it holds people's keystrokes. Results from real check-ins stay out of git too, except the frozen models in `service/models/`, which hold only numbers.

## Built with

- [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [SciPy](https://scipy.org/), [scikit-learn](https://scikit-learn.org/) and [statsmodels](https://www.statsmodels.org/), under the BSD 3-Clause licence
- [Matplotlib](https://matplotlib.org/), under the Matplotlib licence
- [pytest](https://pytest.org/), under the MIT licence
- [uv](https://docs.astral.sh/uv/), under the MIT or Apache 2.0 licence
- [Vercel](https://vercel.com/), which hosts the scoring API
- The SWELL knowledge work dataset (Koldijk et al., 2014), which Ananya's SWELL stress recipe was trained on
- The KSS (Åkerstedt and Gillberg, 1990), and the 3-minute reaction-time test with a 355 ms lapse cut-off (Basner, Mollicone and Dinges, 2011)

## Credits

Claude Code helped build this project. `AI_USAGE.md` records what AI generated and what we wrote ourselves.
