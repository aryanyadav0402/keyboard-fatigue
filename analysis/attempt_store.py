"""Attempt history for the hosted API: every scored attempt, as numbers, under a scrambled user key.

The demo page only sends the session. The API stores each scored attempt (the rule-adjusted typing
features, the features it shows, the score, the first guess and which model answered) and builds each
user's baseline from their first attempts. It never stores keystrokes, the typed text or the name.

- PostgresStore: the deployed database. Vercel's Supabase or Neon integration provides the address in
  POSTGRES_URL (or DATABASE_URL); the table is created on first use.
- MemoryStore: tests, and the local dev server when KF_MEMORY_STORE=1.
"""

import os
from datetime import datetime, timezone
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

SCHEMA = [
    """create table if not exists attempts (
         id bigserial primary key,
         user_key text not null,
         created_at timestamptz not null default now(),
         model_version text not null,
         model text not null,
         score double precision,
         first_guess double precision,
         rule text,
         adj jsonb not null,
         shown jsonb not null)""",
    "create index if not exists attempts_user_time on attempts (user_key, created_at)",
]
FIELDS = ["model_version", "model", "score", "first_guess", "rule", "adj", "shown"]
# connection settings libpq understands; anything else in the URL (Supabase adds supa=...) is dropped
LIBPQ_PARAMS = {"sslmode", "sslrootcert", "sslcert", "sslkey", "connect_timeout", "application_name",
                "options", "target_session_attrs", "channel_binding"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class MemoryStore:
    """Keeps attempts in memory, oldest first."""

    def __init__(self):
        self.rows = []

    def history(self, user_key: str, limit: int = 200) -> list[dict]:
        return [dict(r) for r in self.rows if r["user_key"] == user_key][:limit]

    def add(self, record: dict) -> None:
        self.rows.append(dict(record, created_at=now_iso()))


def clean_url(url: str) -> str:
    parts = urlsplit(url)
    query = [(k, v) for k, v in parse_qsl(parts.query) if k in LIBPQ_PARAMS]
    return urlunsplit(parts._replace(query=urlencode(query)))


class PostgresStore:
    """Attempts in a Postgres table, one short connection per call (enough for a demo)."""

    def __init__(self, url: str):
        import psycopg  # only needed where the database is

        self.psycopg = psycopg
        self.url = clean_url(url)
        self.ready = False

    def connect(self):
        # prepare_threshold=None: poolers in transaction mode (Supabase, Neon) can't keep prepared statements
        return self.psycopg.connect(self.url, connect_timeout=5, prepare_threshold=None, autocommit=True)

    def ensure_table(self, conn) -> None:
        if not self.ready:
            for statement in SCHEMA:
                conn.execute(statement)
            self.ready = True

    def history(self, user_key: str, limit: int = 200) -> list[dict]:
        with self.connect() as conn:
            self.ensure_table(conn)
            rows = conn.execute(
                "select created_at, " + ", ".join(FIELDS) + " from attempts where user_key = %s "
                "order by created_at, id limit %s", (user_key, limit)).fetchall()
        return [{"user_key": user_key, "created_at": r[0].isoformat(timespec="seconds"),
                 **dict(zip(FIELDS, r[1:]))} for r in rows]

    def add(self, record: dict) -> None:
        from psycopg.types.json import Jsonb

        values = [Jsonb(record[f]) if f in ("adj", "shown") else record.get(f) for f in FIELDS]
        with self.connect() as conn:
            self.ensure_table(conn)
            conn.execute("insert into attempts (user_key, " + ", ".join(FIELDS) + ") values ("
                         + ", ".join(["%s"] * (len(FIELDS) + 1)) + ")", [record["user_key"], *values])


_MEMORY = None


def store_from_env():
    """The store this deployment should use, or None when it has no database."""
    global _MEMORY
    if os.environ.get("KF_MEMORY_STORE"):
        _MEMORY = _MEMORY or MemoryStore()  # one shared store for every endpoint in this process
        return _MEMORY
    url = os.environ.get("POSTGRES_URL") or os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_URL_NON_POOLING")
    if not url:
        return None
    try:
        return PostgresStore(url)
    except Exception:  # no driver or a bad address: score without history rather than not at all
        return None
