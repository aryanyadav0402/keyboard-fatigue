# keyboard-fatigue

Our HackTheBurgh XIII project: we predict how tired and how motivated someone is from how they type.

## How it works

Our panel of 6–7 people does a check-in every hour they're awake, plus one just before and one just after sleeping. Everyone uses one shared laptop, so the keyboard is always the same. Each check-in has four steps, in this order:

1. Rate tiredness (0 = fully alert, 10 = falling asleep) and motivation (0 = want to go home, 10 = fully locked in). Then answer a few tags: caffeine in the last hour, ate in the last hour, slept since the last check-in, what they were just doing, and when they last woke up.
2. Type a sentence exactly as shown.
3. Type a different sentence with every vowel (a, e, i, o, u, not y) capitalised, using Shift rather than Caps Lock.
4. Do about 20 reaction-time trials: press Space when a box turns green after a random wait of 1–4 seconds.

The check-in page must not show accuracy feedback.

The vowel rule adds mental load. Comparing the two typing tasks within the same check-in cancels out each person's own typing speed, and we expect the rule to slow people down more as they get tired.

A few one-off people, such as judges, do one or two check-ins. They're our test set, and we never train on them.

## Data format

The analysis reads two CSV files from `data/`. If the check-in page or the logger saves something different, tell Aryan and the analysis will adapt to it.

`checkins.csv` has one row per check-in:

```text
checkin_id, person, participant_type (panel or oneoff), timestamp (ISO, local time), epoch_ms,
tiredness, motivation, caffeine, ate, napped, activity, woke_at (HH:MM),
plain_item_id, plain_typed, rule_item_id, rule_typed,
rt_trials (reaction times in ms, separated by semicolons), rt_false_starts
```

`keystrokes.csv` has one row per key press, leaving out auto-repeats:

```text
checkin_id, person, task (plain or rule), seq, key, code, down_ms, up_ms
```

`key` and `code` are the browser's `KeyboardEvent.key` and `KeyboardEvent.code`. `down_ms` and `up_ms` are milliseconds since the task appeared on screen, taken from `event.timeStamp` or `performance.now()`.

`data/` is kept out of git because it holds people's keystrokes.

## Credits

Claude Code helped build this project. `AI_USAGE.md` records what AI generated and what we wrote ourselves.
