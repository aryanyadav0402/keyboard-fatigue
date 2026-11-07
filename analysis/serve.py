"""Serve tiredness predictions to the dashboard, updating whenever a new check-in arrives.

    uv run python serve.py --data ../data/sessions

Run it on the laptop where the check-in app saves its files, pointing --data at the app's
data/sessions folder. It checks the folder every few seconds and rescores when a file appears or
changes. The JSON has CORS headers, so a dashboard served from another address can read it.

  GET /api/status              counts, when it last updated, and what's missing before the model can predict
  GET /api/latest              each person's latest check-in: predicted and rated KSS, nap and push flags
  GET /api/history             every check-in with rated and predicted KSS, for charts over the night
  GET /api/checkin/<id>        one check-in, for showing someone their score right after they finish
  POST /api/score              score one demo attempt sent in the body, compared with that user's best (see demo.py)
  GET /api/best/<username>     a demo user's best attempt so far

The same JSON is also written to --out (status.json, latest.json, history.json) after every update.
Demo users' best attempts are kept in --db, a SQLite file; check-in files are never written.
By default it only answers on this laptop; --host 0.0.0.0 shares it on the local network.
"""

import argparse
import json
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

import pandas as pd

from demo import BadSession, clean, get_best, score_session
from features import build_features
from load_data import Tables, empty_tables, folder_problem, load_sessions, normalise_name
from model import NAP_AT, PUSH_WARNING_AT, add_flags, all_predictions, missing_data, scored

POLL_SECONDS = 5
MAX_BODY_BYTES = 2_000_000
ENDPOINTS = ["GET /api/status", "GET /api/latest", "GET /api/history", "GET /api/checkin/<checkin_id>",
             "POST /api/score", "GET /api/best/<username>"]


def records(df: pd.DataFrame) -> list[dict]:
    return [{k: clean(v) for k, v in row.items()} for row in df.to_dict("records")]


def build_payloads(folder: str | Path, tables: Tables | None = None) -> dict:
    """Everything the dashboard reads, built from the check-in files in folder."""
    tables = tables if tables is not None else load_sessions(folder)
    f = build_features(tables)
    now = datetime.now().isoformat(timespec="seconds")
    thresholds = {"nap_at": NAP_AT, "push_warning_at": PUSH_WARNING_AT}
    status = {"updated_at": now, "data": str(folder), "checkins": len(f),
              "skipped_test_runs": tables.skipped_test_runs, "skipped_in_progress": tables.skipped_partial,
              "problems": tables.problems[:20]}
    if f.empty:
        status.update(people=0, panel_people=0, one_off_people=0, models_ready=False, missing=[folder_problem(folder) or "no check-ins yet"])
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
        self.tables = empty_tables()  # until the first update
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
            tables = load_sessions(self.folder)
            payloads = build_payloads(self.folder, tables)
        except Exception as err:  # keep serving the last good data, and say what went wrong
            with self.lock:
                self.signature = signature
                self.payloads["status"]["error"] = f"{type(err).__name__}: {err}"
            raise
        with self.lock:
            self.signature, self.payloads, self.tables = signature, payloads, tables
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

    def training_tables(self) -> Tables:
        with self.lock:
            return self.tables


def make_handler(feed: Feed, db_path: str | Path | None = None):
    class Handler(BaseHTTPRequestHandler):
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
            path = self.path.split("?", 1)[0].rstrip("/")
            if path in ("/api/status", "/api/latest", "/api/history"):
                return self.send_json(200, feed.get(path.rsplit("/", 1)[1]))
            if path.startswith("/api/best/"):
                best = get_best(db_path, normalise_name(unquote(path.rsplit("/", 1)[1])))
                if best is None:
                    return self.send_json(404, {"error": "no attempts stored for that username yet"})
                return self.send_json(200, {"username": best["username"], "score": round(best["score"], 1),
                                            "attempts": best["attempts"], "best_at": best["best_at"],
                                            "rule": best["rule"], "features": best["features"].get("raw")})
            if path.startswith("/api/checkin/"):
                checkin_id = path.rsplit("/", 1)[1]
                found = [c for c in feed.get("history")["checkins"] if c["checkin_id"] == checkin_id]
                if found:
                    return self.send_json(200, found[0])
                return self.send_json(404, {"error": "no check-in with that id yet; try again in a few seconds"})
            if path in ("", "/api"):
                return self.send_json(200, {"endpoints": ENDPOINTS})
            self.send_json(404, {"error": "unknown path", "endpoints": ENDPOINTS})

        def do_POST(self) -> None:
            if self.path.split("?", 1)[0].rstrip("/") != "/api/score":
                return self.send_json(404, {"error": "unknown path", "endpoints": ENDPOINTS})
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY_BYTES:
                return self.send_json(413, {"error": "that session is too large"})
            try:
                body = json.loads(self.rfile.read(length) or b"null")
            except (ValueError, UnicodeDecodeError):
                return self.send_json(400, {"error": "the body must be JSON"})
            try:
                result = score_session(body, feed.training_tables(), db_path)  # only the best attempt is stored
            except BadSession as err:
                return self.send_json(422, {"error": str(err)})
            except Exception as err:
                return self.send_json(500, {"error": f"{type(err).__name__}: {err}"})
            self.send_json(200, result)

        def log_message(self, *args) -> None:  # keep the console for update messages
            pass

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve live tiredness predictions as JSON for the dashboard.")
    parser.add_argument("--data", required=True, help="the check-in app's data/sessions folder")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to share on the local network")
    parser.add_argument("--out", default="results/live", help="also write the JSON here after every update")
    parser.add_argument("--db", default="results/demo_users.sqlite", help="SQLite file for demo users' best attempts")
    args = parser.parse_args()

    feed = Feed(args.data, args.out)
    try:
        feed.refresh_if_changed()
    except Exception as err:
        print(f"First update failed ({type(err).__name__}: {err}); will retry when the folder changes.")
    problem = folder_problem(args.data)
    if problem:
        print(f"Warning: {problem}. Point --data at the folder that holds the check-in files.", flush=True)
    threading.Thread(target=feed.run_forever, daemon=True).start()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(feed, args.db))
    shown = "localhost" if args.host in ("127.0.0.1", "localhost") else args.host
    print(f"Serving predictions at http://{shown}:{args.port}/api/latest (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
