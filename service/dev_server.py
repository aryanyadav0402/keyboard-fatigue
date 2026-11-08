"""Run the hosted API on this laptop, to try it before deploying.

    cd analysis && uv run python ../service/build.py && uv run python ../service/dev_server.py

It serves the same /api/score and /api/models as Vercel, at http://localhost:8766.
"""

import importlib.util
import sys
from http.server import ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8766


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / "api" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.handler


ROUTES = {"/api/score": load("score"), "/api/models": load("models"), "/api/history": load("history")}


class Router(ROUTES["/api/score"]):
    """Hand each request to the endpoint its path names, like Vercel does."""

    def handle_one_request(self) -> None:
        self.raw_requestline = self.rfile.readline(65537)
        if not self.raw_requestline or not self.parse_request():
            return
        handler = ROUTES.get(self.path.split("?", 1)[0].rstrip("/"))
        if handler is None:
            self.send_error(404)
            return
        method = getattr(handler, f"do_{self.command}", None)
        if method is None:
            self.send_error(405)
            return
        method(self)
        self.wfile.flush()


if __name__ == "__main__":
    print(f"Hosted API on this laptop: http://localhost:{PORT}/api/score and /api/models (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", PORT), Router).serve_forever()
