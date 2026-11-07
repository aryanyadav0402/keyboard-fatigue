"""Print each person's latest predicted tiredness (KSS 1-9), for the dashboard, sleep rota and push guard.

    uv run python predict_latest.py --data ../data/sessions          # a table
    uv run python predict_latest.py --data ../data/sessions --json   # the same JSON as serve.py's /api/latest

Models are trained on all panel data. A nap is suggested at a predicted KSS of 7 or more ("sleepy,
but no effort to keep awake"), most tired first, keeping at least 2 panel people awake. The push
guard warns at 8 or more ("sleepy, some effort to keep awake"). For a live feed, run serve.py.
"""

import argparse
import json
import math

from features import build_features
from load_data import load_sessions
from model import latest_predictions
from serve import build_payloads


def main() -> None:
    parser = argparse.ArgumentParser(description="Latest predicted tiredness per person.")
    parser.add_argument("--data", required=True, help="folder of typing-station session files (*.json)")
    parser.add_argument("--json", action="store_true", help="print JSON instead of a table")
    args = parser.parse_args()

    if args.json:
        print(json.dumps(build_payloads(args.data)["latest"], indent=2))
        return
    features = build_features(load_sessions(args.data))
    latest = latest_predictions(features) if len(features) else None
    if latest is None or latest.empty:
        print("No check-ins yet.")
        return
    for row in latest.sort_values("tiredness_pred", ascending=False, na_position="last").itertuples():
        pred = "  –" if math.isnan(row.tiredness_pred) else f"{row.tiredness_pred:3.1f}"
        flags = " ".join(f for f, on in (("NAP", row.nap_suggested), ("PUSH-WARNING", row.push_warning)) if on)
        print(f"{row.person:14} {row.timestamp:%a %H:%M}  predicted KSS {pred}  (rated {row.kss_reported:.0f}, "
              f"{row.model}) {flags}")


if __name__ == "__main__":
    main()
