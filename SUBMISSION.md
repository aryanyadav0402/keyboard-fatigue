# keyboard-fatigue: submission draft

Paste each section into the form and edit freely.

**Tracks:** G-Research (make the invisible visible), Hudson River Trading (best use of data for predictions and decisions), Metaview (best overall)

**Links**
- Demo: https://tiredtype.vercel.app/
- Scoring API: https://keyboard-fatigue-api.vercel.app
- Repo: https://github.com/aryanyadav0402/keyboard-fatigue

## Tagline

We stayed up all night measuring ourselves getting tired, so one typed sentence can tell how tired you are.

## Inspiration

Tiredness is invisible. You can't see it, and people are bad at judging their own. At a 24-hour hackathon, all four of us were about to find out what sleep loss does to us, so we turned the night into an experiment.

## What it does

You type one short sentence with a small rule, such as capitalising every vowel. We estimate how sleepy you are on the Karolinska Sleepiness Scale (KSS: 1 is extremely alert, 9 is fighting sleep).

The demo runs Ananya's per-person model. It compares your typing with your own usual typing, and adds the change to your usual sleepiness for the time of day.

- **Teammates** are compared with their own check-ins from the night.
- **Everyone else** is compared with the average of our 4 testers on their first try, then with their own earlier tries. The page only sends the session; every attempt is saved, and the baseline is built on the server.
- **A second model,** Aryan's, runs on the same API. It's the one we tested most thoroughly overnight (see Results).

## How we built it

- **Data:** Ananya built a check-in app, and the four of us used it about once an hour all night. Each check-in has a sentence with a rule, a 20-second reaction test, and a sleepiness rating. That gave us 42 check-ins from the 4 of us plus 3 walk-ups.
- **Features:** the timing of every key press. The strongest idea is *rule cost*: the gaps next to keys the rule changes, compared with ordinary gaps in the same sentence. That cancels out how fast someone types anyway. We also use pauses, rhythm, how long keys are held, backspaces and accuracy.
- **Models:** both compare you with yourself, using a ridge regression on how your typing changed to predict how your sleepiness changed.
  - Ananya's, which runs the demo, uses 8 timing features from each half of the sentence (speed, gaps, rhythm, pauses, key holds, backspaces), measured against your usual typing.
  - Aryan's uses 6 features, including rule cost, measured against your first check-ins.
- **Testing:**
  - Each person is predicted by a model that never saw them.
  - We also test forward in time.
  - We compare against guessing and against the clock, and run shuffle tests.
  - Before new data arrived, we froze a test with file fingerprints.
- **Product:** a Python API on Vercel serves both trained models as plain numbers. A Neon Postgres database stores each attempt's typing numbers under a scrambled username, never the keystrokes or text. The front end is React with Vite.

## Results (small data, so treat them as promising)

- **Aryan's model:** typing predicted sleepiness better than guessing and better than the clock, with an average error of 1.38 KSS points against 1.60 for guessing and 1.79 for hours into the night (shuffle test p = 0.02).
- **Ananya's model:** holding out one session at a time, it called whether someone was more or less tired than usual right two times in three (AUC 0.71).
- The rule really does load the brain: gaps next to rule keys took 1.86 times as long (p < 0.001).
- When sleepier, we made fewer typos and paused more, and still fixed almost every mistake. Tired typing looked careful.
- On 7 check-ins neither model had seen, both were about as accurate as guessing on average (Aryan's: 1.56 against 1.68). Aryan's break rule (suggest a break at 7 or more) caught 3 of the 5 tired ratings with no false alarms. So the result is promising but not proven.

## Challenges

- **Practice looks like tiredness.** Typing gets smoother with every check-in while the night gets later, so a model can learn that smooth typing means tired. We found this, checked it with simulations, and designed the test that separates the two: check-ins after sleep, when people are still practised but rested.
- **Tiny data.** Four people over one night. We counted every model we tried (about 117 overnight, from neural networks to ranking models) and checked winners against luck. None beat the simple model, so we kept the simple model.
- **Transfer learning.** We tried pretraining on SWELL, a public dataset of office workers under stress. It did no better than two simple typing numbers, and we worked out why.

## Accomplishments we're proud of

- Real data from a real all-nighter, collected with our own tool.
- Testing we'd defend in front of a quant: models frozen before new data, every attempt counted, honest negative results.
- Two of us built models independently, reached the same conclusions, and both run live.
- A live demo that remembers you between tries.

## What we learned

- Comparing you with yourself is the only thing that worked; comparing people with each other didn't.
- The hidden confounder (practice) mattered more than any model choice.
- A tested idea beats a fancy pipeline.

## What's next

Months of typing from one person could show their daily rhythm, their best working hours and early signs of burnout. It would log timing only, never content, and process it on the device. The first step is the after-sleep test with more people.

## Built with

Python, pandas, NumPy, SciPy, scikit-learn, statsmodels, Vercel serverless functions, Neon Postgres, React, Vite, TypeScript, the SWELL knowledge-work dataset (for the transfer-learning experiment), the KSS and the reaction-time test (PVT-B).

## Team

- Aryan: his model, the testing and experiments, and the scoring API
- Ananya: the check-in app, the per-person model the demo runs, and the SWELL experiment
- Christy: the demo page
- Fin: the pitch

## Use of AI

We used Claude Code throughout: to write most of the analysis and API code, to run the overnight model experiments, and to help draft this text. `AI_USAGE.md` in the repo lists what AI generated and what we built ourselves. All data comes from our own check-ins, collected during the event.
