"""Fit the models on the team's check-ins and save them as a frozen model file for the hosted API.

    uv run python export_model.py --data ../data/sessions --name real-2026-11-08-0700 --default

The file goes to ../service/models/<name>.json, and ../service/models/index.json lists the models
the API serves (--default makes this one the model used when a request doesn't name one). The file
holds only numbers: no names, timestamps or keystrokes.
"""

import argparse
import json
from pathlib import Path

from features import build_features
from frozen_model import describe_model, export_model
from load_data import load_sessions

MODELS = Path(__file__).resolve().parent.parent / "service" / "models"


def main() -> None:
    parser = argparse.ArgumentParser(description="Save the trained models as a frozen model file.")
    parser.add_argument("--data", required=True, help="folder of typing-station session files (*.json)")
    parser.add_argument("--name", required=True, help="model name, e.g. real-2026-11-08-0700")
    parser.add_argument("--note", default="", help="a short description shown by GET /api/models")
    parser.add_argument("--default", action="store_true", help="serve this model when a request names none")
    parser.add_argument("--out", default=str(MODELS))
    args = parser.parse_args()

    f = build_features(load_sessions(args.data))
    if f.empty:
        raise SystemExit(f"No check-ins found in {args.data}")
    model = export_model(f, args.name, args.note)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.name}.json").write_text(json.dumps(model, indent=1) + "\n")
    index_path = out / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {"default": None, "models": []}
    if args.name not in index["models"]:
        index["models"].append(args.name)
    if args.default or index["default"] is None:
        index["default"] = args.name
    index_path.write_text(json.dumps(index, indent=1) + "\n")
    summary = describe_model(model)
    print(f"Saved {out / (args.name + '.json')} (default: {index['default']})")
    print(f"  trained on {summary['trained_on']}")
    print(f"  first guess ready: {summary['first_guess_ready']} (follows tiredness: "
          f"{summary['first_guess_follows_tiredness']}); personal model ready: {summary['personal_ready']}")


if __name__ == "__main__":
    main()
