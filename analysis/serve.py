"""Serve tiredness predictions to the dashboard, updating whenever a new check-in arrives.

    uv run python serve.py --data ../data/sessions

Run it on the laptop where the check-in app saves its files, pointing --data at the app's
data/sessions folder. It checks the folder every few seconds and rescores when a file appears or
changes. The JSON has CORS headers, so a dashboard served from another address can read it.

  GET /api/status              counts, when it last updated, and what's missing before the model can predict
  GET /api/latest              each person's latest check-in: predicted and rated KSS, nap and push flags
  GET /api/history             every check-in with rated and predicted KSS, for charts over the night
  GET /api/checkin/<id>        one check-in, for showing someone their score right after they finish

The same JSON is also written to --out (status.json, latest.json, history.json) after every update.
By default it only answers on this laptop; --host 0.0.0.0 shares it on the local network.
"""

import argparse
import json
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd

from features import build_features
from load_data import load_sessions
from model import NAP_AT, PUSH_WARNING_AT, add_flags, all_predictions, missing_data, scored

POLL_SECONDS = 5
ENDPOINTS = ["/api/status", "/api/latest", "/api/history", "/api/checkin/<checkin_id>"]


def clean(value):
    """Make one value JSON-friendly: NaN becomes null, numpy numbers become plain ones, times become text."""
    if isinstance(value, str):
        return value
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return round(float(value), 3)
    return value


def records(df: pd.DataFrame) -> list[dict]:
    return [{k: clean(v) for k, v in row.items()} for row in df.to_dict("records")]


def build_payloads(folder: str | Path) -> dict:
    """Everything the dashboard reads, built from the check-in files in folder."""
    tables = load_sessions(folder)
    f = build_features(tables)
    now = datetime.now().isoformat(timespec="seconds")
    thresholds = {"nap_at": NAP_AT, "push_warning_at": PUSH_WARNING_AT}
    status = {"updated_at": now, "data": str(folder), "checkins": len(f),
              "skipped_test_runs": tables.skipped_test_runs, "skipped_in_progress": tables.skipped_partial,
              "problems": tables.problems[:20]}
    if f.empty:
        status.update(people=0, panel_people=0, one_off_people=0, models_ready=False, missing=["no check-ins yet"])
        return {"status": status, "latest": {"updated_at": now, **thresholds, "people": []},
                "history": {"updated_at": now, "checkins": []}}
    preds = all_predictions(f).sort_values("timestamp").reset_index(drop=True)
    latest = add_flags(preds.groupby("person").tail(1).reset_index(drop=True))
    latest = latest.sort_values("tiredness_pred", ascending=False, na_position="last")
    missing = missing_data(scored(f))
    status.update(people=int(f.person.nunique()),
                  panel_people=int(f[f.participant_type.eq("panel")].person.nunique()),
                  one_off_people=int(f[f.participant_type.eq("oneoff")].person.nunique()),
                  models_ready=not missing, missing=missing)
    return {"status": status, "latest": {"updated_at": now, **thresholds, "people": records(latest)},
            "history": {"updated_at": now, "checkins": records(preds)}}


class Feed:
    """The latest payloads, rebuilt in the background whenever the folder changes."""

    def __init__(self, folder: str | Path, out: str | Path | None = None):
        self.folder, self.out = Path(folder), Path(out) if out else None
        self.lock = threading.Lock()
        self.signature = None
        self.payloads = {"status": {"updated_at": None, "models_ready": False, "missing": ["starting up"]},
                         "latest": {"people": []}, "history": {"checkins": []}}

    def current_signature(self) -> tuple:
        files = []
        for p in self.folder.rglob("*.json"):
            try:
                stat = p.stat()
            except FileNotFoundError:  # removed while we were looking
                continue
            files.append((str(p), stat.st_mtime_ns, stat.st_size))
        return tuple(sorted(files))

    def refresh_if_changed(self) -> bool:
        signature = self.current_signature()
        if signature == self.signature:
            return False
        try:
            payloads = build_payloads(self.folder)
        except Exception as err:  # keep serving the last good data, and say what went wrong
            with self.lock:
                self.signature = signature
                self.payloads["status"]["error"] = f"{type(err).__name__}: {err}"
            raise
        with self.lock:
            self.signature, self.payloads = signature, payloads
        if self.out:
            self.out.mkdir(parents=True, exist_ok=True)
            for name, payload in payloads.items():
                (self.out / f"{name}.json").write_text(json.dumps(payload, indent=1))
        return True

    def run_forever(self) -> None:
        while True:
            try:
                if self.refresh_if_changed():
                    status = self.get("status")
                    print(f"{datetime.now():%H:%M:%S} updated: {status['checkins']} check-ins, "
                          f"model {'ready' if status['models_ready'] else 'waiting for more data'}", flush=True)
            except Exception as err:
                print(f"{datetime.now():%H:%M:%S} update failed: {type(err).__name__}: {err}", flush=True)
            time.sleep(POLL_SECONDS)

    def get(self, name: str) -> dict:
        with self.lock:
            return self.payloads[name]


def make_handler(feed: Feed):
    class Handler(BaseHTTPRequestHandler):
        def cors(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
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
            path = self.path.split("?", 1)[0].rstrip("/")
            if path in ("/api/status", "/api/latest", "/api/history"):
                return self.send_json(200, feed.get(path.rsplit("/", 1)[1]))
            if path.startswith("/api/checkin/"):
                checkin_id = path.rsplit("/", 1)[1]
                found = [c for c in feed.get("history")["checkins"] if c["checkin_id"] == checkin_id]
                if found:
                    return self.send_json(200, found[0])
                return self.send_json(404, {"error": "no check-in with that id yet; try again in a few seconds"})
            if path in ("", "/api"):
                return self.send_json(200, {"endpoints": ENDPOINTS})
            self.send_json(404, {"error": "unknown path", "endpoints": ENDPOINTS})

        def log_message(self, *args) -> None:  # keep the console for update messages
            pass

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve live tiredness predictions as JSON for the dashboard.")
    parser.add_argument("--data", required=True, help="the check-in app's data/sessions folder")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to share on the local network")
    parser.add_argument("--out", default="results/live", help="also write the JSON here after every update")
    args = parser.parse_args()

    feed = Feed(args.data, args.out)
    try:
        feed.refresh_if_changed()
    except Exception as err:
        print(f"First update failed ({type(err).__name__}: {err}); will retry when the folder changes.")
    threading.Thread(target=feed.run_forever, daemon=True).start()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(feed))
    shown = "localhost" if args.host in ("127.0.0.1", "localhost") else args.host
    print(f"Serving predictions at http://{shown}:{args.port}/api/latest (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
