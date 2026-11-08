"""POST /api/score: score one demo attempt with a frozen model (see analysis/frozen_model.py).

The request and reply are the same as the local POST /api/score, plus two reply fields:
`best_reference`, which the page keeps for each username and sends back with that user's next
attempt (this API stores nothing), and `model_version`. ?model=<name> or "model" in the body picks
one of the models listed by GET /api/models; otherwise the default model is used.
"""

import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lib"))

from attempt import BadSession  # noqa: E402
from frozen_model import check_model, score_attempt  # noqa: E402
from attempt_store import store_from_env  # noqa: E402

MAX_BODY_BYTES = 2_000_000
STORE = store_from_env()  # the attempt history database, or None when this deployment has none


def load_models() -> tuple[str, dict]:
    index = json.loads((ROOT / "models" / "index.json").read_text())
    models = {name: check_model(json.loads((ROOT / "models" / f"{name}.json").read_text()))
              for name in index["models"]}
    return index["default"], models


DEFAULT, MODELS = load_models()


class handler(BaseHTTPRequestHandler):
    def cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def send_json(self, code: int, payload) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.cors()
        self.end_headers()

    def do_GET(self) -> None:
        self.send_json(200, {"usage": "POST one demo attempt as JSON; see the README's /api/score section",
                             "default_model": DEFAULT, "models": sorted(MODELS)})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY_BYTES:
            return self.send_json(413, {"error": "that session is too large"})
        try:
            body = json.loads(self.rfile.read(length) or b"null")
        except (ValueError, UnicodeDecodeError):
            return self.send_json(400, {"error": "the body must be JSON"})
        chosen = parse_qs(urlparse(self.path).query).get("model", [None])[0]
        name = chosen or (body.get("model") if isinstance(body, dict) else None) or DEFAULT
        if name not in MODELS:
            return self.send_json(404, {"error": f"no model called {name!r}", "models": sorted(MODELS)})
        try:
            result = score_attempt(body, MODELS[name], store=STORE)  # stores numbers only, never keystrokes
        except BadSession as err:
            return self.send_json(422, {"error": str(err)})
        except Exception as err:
            return self.send_json(500, {"error": f"{type(err).__name__}: {err}"})
        self.send_json(200, result)

    def log_message(self, *args) -> None:  # never log requests: they hold people's typing
        pass
