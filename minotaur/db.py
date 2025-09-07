from __future__ import annotations

import contextlib
import os
from pathlib import Path
import sqlite3
import threading
from typing import Any, Iterable, List, Optional


_LOCAL = threading.local()


def _ensure_parent_dir_for_path(db_path: str) -> None:
    """Ensure the parent directory for a disk-backed SQLite path exists.

    - No-op for in-memory or non-file paths/URIs.
    - For file: URIs, extracts the filesystem path portion and mkdirs its parent.
    """
    try:
        p = db_path.strip()
        # Skip in-memory DBs
        if p == ":memory:" or (p.startswith("file:") and "mode=memory" in p):
            return
        fs_path: Optional[str] = None  # type: ignore
        if p.startswith("file:"):
            # Extract path portion up to '?' (sqlite URI semantics)
            raw = p[len("file:"):]
            q = raw.find("?")
            if q != -1:
                raw = raw[:q]
            fs_path = raw
            # Normalize relative path under our package base dir to avoid CWD issues
            if fs_path and not os.path.isabs(fs_path):
                try:
                    from .config import BASE_DIR  # lazy import to avoid cycles
                    fs_path = os.path.join(BASE_DIR, fs_path)
                except Exception:
                    pass
        elif not p.startswith("sqlite:"):
            # Treat as a normal filesystem path
            fs_path = p
        if fs_path:
            parent = Path(fs_path).expanduser().resolve().parent
            try:
                parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
    except Exception:
        # Best-effort only; connection attempt will surface errors if any
        pass


def get_conn(path: str) -> sqlite3.Connection:
    conn: Optional[sqlite3.Connection] = getattr(_LOCAL, "conn", None)
    if conn is not None:
        return conn
    # Enable URI when using shared in-memory DB or file: URIs
    use_uri = path.startswith("file:") or "mode=memory" in path or path.startswith("sqlite:")
    # Ensure directory exists for on-disk databases to avoid SQLITE_CANTOPEN
    _ensure_parent_dir_for_path(path)

    # Optional debug logging gate
    _dbg = (os.getenv("MINOTAUR_DB_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"})
    if _dbg:
        try:
            fs_path: Optional[str] = None
            if path.startswith("file:"):
                raw = path[len("file:"):]
                q = raw.find("?")
                if q != -1:
                    raw = raw[:q]
                fs_path = raw
                if fs_path and not os.path.isabs(fs_path):
                    try:
                        from .config import BASE_DIR  # lazy import
                        fs_path = os.path.join(BASE_DIR, fs_path)
                    except Exception:
                        pass
            elif not path.startswith("sqlite:"):
                fs_path = path
            parent = None
            exists = None
            writable = None
            parent_exists = None
            parent_writable = None
            if fs_path:
                exists = os.path.exists(fs_path)
                writable = os.access(fs_path, os.W_OK) if exists else None
                parent = os.path.dirname(os.path.abspath(fs_path))
                parent_exists = os.path.isdir(parent)
                parent_writable = os.access(parent, os.W_OK) if parent_exists else None
            print(f"[db.debug] opening sqlite path='{path}' uri={use_uri} fs_path={fs_path} cwd='{os.getcwd()}' exists={exists} writable={writable} parent='{parent}' parent_exists={parent_exists} parent_writable={parent_writable}", flush=True)
        except Exception:
            pass

    try:
        conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None, uri=use_uri)
    except Exception as e:
        if _dbg:
            # Emit a hint about typical CANTOPEN causes
            try:
                print(f"[db.debug] sqlite connect failed for path='{path}': {e}", flush=True)
            except Exception:
                pass
        raise
    conn.row_factory = sqlite3.Row
    with conn:
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
        except Exception:
            # WAL is not supported for in-memory DBs; ignore
            pass
        conn.execute("PRAGMA synchronous=NORMAL;")
        # Cap per-connection cache to reduce RSS on small machines.
        # Negative value = KiB units. Default to 2048 KiB (2 MiB) per connection; override via env.
        try:
            _cache_kb_env = os.getenv("MINOTAUR_SQLITE_CACHE_KB") or os.getenv("SQLITE_CACHE_KB")
            cache_kb = int(_cache_kb_env) if _cache_kb_env else 2048
            cache_kb = max(256, min(65536, cache_kb))  # clamp between 256 KiB and 64 MiB
            conn.execute(f"PRAGMA cache_size={-abs(cache_kb)};")
        except Exception:
            pass
        # Allow page cache to spill to disk if needed (usually ON by default, but be explicit)
        try:
            conn.execute("PRAGMA cache_spill=1;")
        except Exception:
            pass
        # Optional: force temp storage to file to avoid large in-memory temps; opt-in via env
        try:
            if (os.getenv("MINOTAUR_SQLITE_TEMP_TO_FILE") or "").strip().lower() in {"1","true","yes","on"}:
                conn.execute("PRAGMA temp_store=FILE;")
        except Exception:
            pass
    _LOCAL.conn = conn
    return conn


class ConnProxy:
    """Lightweight per-thread connection proxy.

    Avoids sharing a single sqlite3.Connection across threads by delegating all
    operations to a thread-local connection obtained via get_conn(path).
    """

    def __init__(self, path: str) -> None:
        self._path = path

    def _conn(self) -> sqlite3.Connection:
        return get_conn(self._path)

    # Common operations used by the codebase
    def execute(self, *args, **kwargs):
        return self._conn().execute(*args, **kwargs)

    def executemany(self, *args, **kwargs):
        return self._conn().executemany(*args, **kwargs)

    def cursor(self):
        return self._conn().cursor()

    # Context manager passthrough (no-op under autocommit but keep parity)
    def __enter__(self):
        return self._conn().__enter__()

    def __exit__(self, exc_type, exc, tb):
        return self._conn().__exit__(exc_type, exc, tb)

    # Generic attribute access passthrough
    def __getattr__(self, name):
        return getattr(self._conn(), name)


def init_schema(conn: sqlite3.Connection) -> None:
    with conn:
        # Drop legacy tables if present; keep request logs across restarts
        conn.execute("DROP TABLE IF EXISTS trials;")
        conn.execute("DROP TABLE IF EXISTS settings;")

        # Determine if challenges table needs reset (schema change)
        need_reset = False
        try:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(challenges)").fetchall()}
            # Require new score_query_count column; if missing, drop and recreate
            if "score_query_count" not in cols:
                need_reset = True
        except Exception:
            need_reset = True

        if need_reset:
            # Drop dependent table first due to FK
            conn.execute("DROP TABLE IF EXISTS challenge_requests;")
            conn.execute("DROP TABLE IF EXISTS challenges;")

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
              upstream_query_count INTEGER DEFAULT 0,
              score_query_count INTEGER
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
