"""Write the raw and processed feature files from the app's session files: one row per check-in.

    uv run python export_features.py --data ../data/sessions --out results/latest

features_raw.csv has every feature as measured. features_processed.csv has what the models use:
each feature's change from the person's first two check-ins (columns ending _dev), backspaces per
100 keys as they are, and the KSS rating and hours awake, also relative to those two check-ins.
"""

import argparse
from pathlib import Path

from features import build_features, processed_table, raw_table
from load_data import load_sessions


def main() -> None:
    parser = argparse.ArgumentParser(description="Write features_raw.csv and features_processed.csv.")
    parser.add_argument("--data", required=True, help="folder of typing-station session files (*.json)")
    parser.add_argument("--out", default="results/latest")
    args = parser.parse_args()

    tables = load_sessions(args.data)
    for problem in tables.problems:
        print("  problem:", problem)
    features = build_features(tables)
    if features.empty:
        print("No check-ins found.")
        return
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    raw_table(features).to_csv(out / "features_raw.csv", index=False)
    processed_table(features).to_csv(out / "features_processed.csv", index=False)
    print(f"Wrote {len(features)} check-ins to {out / 'features_raw.csv'} and {out / 'features_processed.csv'}")


if __name__ == "__main__":
    main()
