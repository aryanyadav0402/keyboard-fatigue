"""GET /api/models: the frozen models this API serves, which one is the default, and what each can do."""

import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lib"))

from frozen_model import check_model, describe_model  # noqa: E402
from ananya_model import KIND as ANANYA, describe_ananya  # noqa: E402


def summary(name: str) -> dict:
    model = check_model(json.loads((ROOT / "models" / f"{name}.json").read_text()))
    return describe_ananya(model) if model.get("kind") == ANANYA else describe_model(model)


INDEX = json.loads((ROOT / "models" / "index.json").read_text())
SUMMARIES = [summary(name) for name in INDEX["models"]]


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:
        body = json.dumps({"default": INDEX["default"], "models": SUMMARIES}).encode()
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:
        pass
