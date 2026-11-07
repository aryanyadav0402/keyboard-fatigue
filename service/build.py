"""Copy the analysis code the hosted API needs into service/lib/, ready to deploy.

    cd analysis && uv run python ../service/build.py
    cd ../service && vercel deploy --prod

The API only needs the feature code and frozen model files, never the team's data or scikit-learn.
"""

import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent / "analysis"
MODULES = ["tasks.py", "load_data.py", "features.py", "first_guess.py", "attempt.py", "frozen_model.py"]


def main() -> None:
    lib = HERE / "lib"
    shutil.rmtree(lib, ignore_errors=True)
    lib.mkdir()
    for name in MODULES:
        shutil.copy2(ANALYSIS / name, lib / name)
    index = json.loads((HERE / "models" / "index.json").read_text())
    missing = [name for name in index["models"] if not (HERE / "models" / f"{name}.json").exists()]
    if missing:
        raise SystemExit(f"models listed in index.json but missing: {missing}")
    print(f"Copied {len(MODULES)} modules into {lib}; models: {index['models']} (default {index['default']})")


if __name__ == "__main__":
    main()
