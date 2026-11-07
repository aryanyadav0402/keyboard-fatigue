"""Print each person's latest predicted tiredness (KSS 1-9), for the dashboard, sleep rota and push guard.

    uv run python predict_latest.py --data ../data/sessions          # a table
    uv run python predict_latest.py --data ../data/sessions --json   # for other programs

Models are trained on all panel data. A nap is suggested at a predicted KSS of 7 or more ("sleepy,
but no effort to keep awake"), most tired first, keeping at least 2 panel people awake. The push
guard warns at 8 or more ("sleepy, some effort to keep awake").
"""

import argparse
import json
import math

from features import build_features
from load_data import load_sessions
from model import latest_predictions

NAP_AT, PUSH_WARNING_AT, KEEP_AWAKE = 7, 8, 2


def main() -> None:
    parser = argparse.ArgumentParser(description="Latest predicted tiredness per person.")
    parser.add_argument("--data", required=True, help="folder of typing-station session files (*.json)")
    parser.add_argument("--json", action="store_true", help="print JSON instead of a table")
    args = parser.parse_args()

    features = build_features(load_sessions(args.data))
    latest = latest_predictions(features) if len(features) else None
    if latest is None or latest.empty:
        print(json.dumps({"people": [], "note": "no check-ins yet"}) if args.json else "No check-ins yet.")
        return

    panel = latest[latest.participant_type == "panel"]
    tired = panel[panel.tiredness_pred >= NAP_AT].sort_values("tiredness_pred", ascending=False)
    nappers = set(tired.person[: max(0, len(panel) - KEEP_AWAKE)])
    latest["nap_suggested"] = latest.person.isin(nappers)
    latest["push_warning"] = latest.tiredness_pred >= PUSH_WARNING_AT

    if args.json:
        people = [{k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in row.items()}
                  for row in latest.assign(timestamp=latest.timestamp.astype(str)).to_dict("records")]
        print(json.dumps({"people": people, "nap_at": NAP_AT, "push_warning_at": PUSH_WARNING_AT}, indent=2))
        return
    for row in latest.sort_values("tiredness_pred", ascending=False).itertuples():
        pred = "  –" if math.isnan(row.tiredness_pred) else f"{row.tiredness_pred:3.1f}"
        flags = " ".join(f for f, on in (("NAP", row.nap_suggested), ("PUSH-WARNING", row.push_warning)) if on)
        print(f"{row.person:14} {row.timestamp:%a %H:%M}  predicted KSS {pred}  (rated {row.kss_reported:.0f}, "
              f"{row.model}) {flags}")


if __name__ == "__main__":
    main()
