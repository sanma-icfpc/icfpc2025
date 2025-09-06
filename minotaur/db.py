from __future__ import annotations

import contextlib
import sqlite3
import threading
from typing import Any, Iterable, List, Optional


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
        # Drop legacy tables if present; keep request logs across restarts
        conn.execute("DROP TABLE IF EXISTS trials;")
        conn.execute("DROP TABLE IF EXISTS settings;")
        # Challenges (one per agent's select-to-finish window)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS challenges (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              ticket TEXT UNIQUE,
              agent_id TEXT,
              agent_name TEXT,
              source_ip TEXT,
              git_sha TEXT,
              problem_id TEXT,
              status TEXT,
              enqueued_at TEXT,
              started_at TEXT,
              finished_at TEXT,
              lease_expire_at TEXT,
              session_id TEXT,
              base_priority INTEGER,
              effective_priority INTEGER,
              upstream_query_count INTEGER DEFAULT 0
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chal_status_enq ON challenges(status, enqueued_at);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chal_eff ON challenges(effective_priority DESC, enqueued_at);"
        )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS uniq_single_running_chal ON challenges(status) WHERE status='running';"
        )
        # Per-request log within a challenge (phased flow)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS challenge_requests (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              challenge_id INTEGER NOT NULL,
              api TEXT,
              req_key TEXT,
              phase TEXT,
              status_code INTEGER,
              req_body TEXT,
              res_body TEXT,
              ts TEXT,
              FOREIGN KEY(challenge_id) REFERENCES challenges(id) ON DELETE CASCADE
            );
            """
        )
        # Agent priorities (per X-Agent-Name), includes a default row name='*'
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_priorities (
              name TEXT PRIMARY KEY,
              priority INTEGER NOT NULL,
              updated_at TEXT,
              vtime REAL NOT NULL DEFAULT 0.0,
              service REAL NOT NULL DEFAULT 0.0,
              pinned INTEGER NOT NULL DEFAULT 0
            );
            """
        )
        # Ensure default row exists
        cur = conn.execute("SELECT 1 FROM agent_priorities WHERE name='*' LIMIT 1")
        if cur.fetchone() is None:
            conn.execute(
                "INSERT INTO agent_priorities(name, priority, updated_at, vtime, service, pinned) VALUES('*', 50, datetime('now'), 0.0, 0.0, 0)"
            )
        # Migrate missing columns in existing DBs
        try:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(agent_priorities)").fetchall()}
            if "vtime" not in cols:
                conn.execute("ALTER TABLE agent_priorities ADD COLUMN vtime REAL NOT NULL DEFAULT 0.0")
            if "service" not in cols:
                conn.execute("ALTER TABLE agent_priorities ADD COLUMN service REAL NOT NULL DEFAULT 0.0")
            if "pinned" not in cols:
                conn.execute("ALTER TABLE agent_priorities ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0")
        except Exception:
            pass


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
