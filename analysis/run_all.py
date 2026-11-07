"""Run the whole analysis: load check-ins, compute features, fit models, write the report.

    uv run python run_all.py --data ../data/sessions --out results/latest
"""

import argparse
import time

from features import build_features
from load_data import folder_problem, load_sessions
from model import MAIN, run_models
from report import write_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict tiredness from typing: the full analysis.")
    parser.add_argument("--data", required=True, help="folder of typing-station session files (*.json)")
    parser.add_argument("--out", default="results/latest", help="folder for report.md, figures and CSVs")
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    start = time.time()
    tables = load_sessions(args.data)
    print(f"Loaded {len(tables.sessions)} check-ins ({tables.skipped_test_runs} test runs skipped)")
    problem = folder_problem(args.data)
    if problem:
        print(f"  Warning: {problem}. Point --data at the folder that holds the check-in files.")
    for problem in tables.problems:
        print("  problem:", problem)
    features = build_features(tables)
    results = run_models(features, n_perm=args.permutations, seed=args.seed)
    for gap in results["missing"]:
        print("  not enough data:", gap)
    report = write_outputs(tables, features, results, args.out, args.data)
    if "beyond_clock" in results:
        b, o, mae = results["beyond_clock"], results["original"], results["table"].MAE
        print(f"Typing alone: error {mae[MAIN]:.2f}; clock alone: {b['clock_mae']:.2f}; typing + clock: "
              f"{b['main_mae']:.2f} (clock check p = {b['p']:.3f}; original test p = {o['p']:.3f})")
    print(f"Done in {time.time() - start:.0f} s. Report: {report}")


if __name__ == "__main__":
    main()
