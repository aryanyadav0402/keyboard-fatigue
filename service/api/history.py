"""GET /api/history?username=<name>: that user's stored attempts, oldest first (scores and times only)."""

import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lib"))

from frozen_model import teammate_key  # noqa: E402
from attempt_store import store_from_env  # noqa: E402

STORE = store_from_env()
PUBLIC = {"personal model": "best-attempt model", "history model": "best-attempt model"}


class handler(BaseHTTPRequestHandler):
    def send_json(self, code: int, payload) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        username = (parse_qs(urlparse(self.path).query).get("username") or [""])[0].strip()
        if not username:
            return self.send_json(400, {"error": "add ?username=<name>"})
        if STORE is None:
            return self.send_json(503, {"error": "this deployment has no database yet"})
        try:
            rows = STORE.history(teammate_key(username))
        except Exception as err:
            return self.send_json(503, {"error": f"the database didn't answer ({type(err).__name__})"})
        attempts = [{"at": r.get("created_at"), "score": None if r.get("score") is None else round(r["score"], 1),
                     "model": PUBLIC.get(r.get("model"), r.get("model")), "rule": r.get("rule")} for r in rows]
        self.send_json(200, {"attempts": attempts, "count": len(attempts)})

    def log_message(self, *args) -> None:  # never log requests
        pass
