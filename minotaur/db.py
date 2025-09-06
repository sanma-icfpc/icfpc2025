from __future__ import annotations

import contextlib
import sqlite3
import threading
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


_LOCAL = threading.local()


def get_conn(path: str) -> sqlite3.Connection:
    conn: Optional[sqlite3.Connection] = getattr(_LOCAL, "conn", None)
    if conn is not None:
        return conn
    conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
    conn.row_factory = sqlite3.Row
    with conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
    _LOCAL.conn = conn
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trials (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              ticket TEXT UNIQUE,
              problem_id TEXT,
              params_json TEXT,
              agent_name TEXT,
              git_sha TEXT,
              base_priority INTEGER,
              effective_priority INTEGER,
              status TEXT,
              enqueued_at TEXT,
              started_at TEXT,
              finished_at TEXT,
              lease_expire_at TEXT,
              session_id TEXT,
              upstream_query_count INTEGER DEFAULT 0
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_trials_status_enq ON trials(status, enqueued_at);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_trials_eff ON trials(effective_priority DESC, enqueued_at);"
        )
        # uniqueness for running at most 1
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS uniq_single_running ON trials(status) WHERE status='running';"
        )
        # idempotency intentionally unsupported; no table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS settings (
              key TEXT PRIMARY KEY,
              value TEXT,
              updated_at TEXT
            );
            """
        )


def query_one(conn: sqlite3.Connection, sql: str, args: Iterable[Any] = ()) -> Optional[sqlite3.Row]:
    cur = conn.execute(sql, tuple(args))
    return cur.fetchone()


def query_all(conn: sqlite3.Connection, sql: str, args: Iterable[Any] = ()) -> List[sqlite3.Row]:
    cur = conn.execute(sql, tuple(args))
    return cur.fetchall()


def exec_sql(conn: sqlite3.Connection, sql: str, args: Iterable[Any] = ()) -> None:
    with conn:
        conn.execute(sql, tuple(args))


@contextlib.contextmanager
def tx(conn: sqlite3.Connection):
    try:
        conn.execute("BEGIN IMMEDIATE;")
        yield
        conn.execute("COMMIT;")
    except Exception:
        conn.execute("ROLLBACK;")
        raise
