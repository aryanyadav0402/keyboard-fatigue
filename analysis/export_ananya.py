"""Save Ananya's per-person model as a model file for the hosted API.

    uv run python export_ananya.py --package /path/to/final_pkg --name ananya-2026-11-08

The package is her folder with model_weights.json and person_baselines.json. The file goes to
../service/models/<name>.json and is added to ../service/models/index.json, without becoming the
default unless --default is given. People's baselines are stored under scrambled keys, so the file
holds no names. Add --typing-only once her model is trained on the typing task's keys alone.
"""

import argparse
import json
from pathlib import Path

from ananya_model import describe_ananya, export_ananya

MODELS = Path(__file__).resolve().parent.parent / "service" / "models"


def main() -> None:
    parser = argparse.ArgumentParser(description="Save Ananya's model as a model file for the hosted API.")
    parser.add_argument("--package", required=True, help="her folder with model_weights.json and person_baselines.json")
    parser.add_argument("--name", required=True, help="model name, e.g. ananya-2026-11-08")
    parser.add_argument("--note", default="", help="a short description shown by GET /api/models")
    parser.add_argument("--typing-only", action="store_true", help="the model was trained on the typing task's keys only")
    parser.add_argument("--default", action="store_true", help="serve this model when a request names none")
    parser.add_argument("--out", default=str(MODELS))
    args = parser.parse_args()

    model = export_ananya(Path(args.package), args.name, args.note, args.typing_only)
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
    summary = describe_ananya(model)
    print(f"Saved {out / (args.name + '.json')} (default: {index['default']})")
    print(f"  {summary['teammates']} people with baselines, {summary['trained_on']['sessions']} clean sessions; "
          f"typing only: {summary['typing_only']}")


if __name__ == "__main__":
    main()
