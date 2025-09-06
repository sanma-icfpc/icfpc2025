from __future__ import annotations

import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional, Callable
from .logging import JsonlLogger

from .config import Settings
from . import db as dbm


def utcnow_str() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class Coordinator:
    """Simple single-threaded sweeper for granting and timeouts.

    Designed for single-process deployment. Uses SQLite IMMEDIATE transactions
    to ensure at most 1 running trial.
    """

    def __init__(self, settings: Settings, conn, logger: Optional[JsonlLogger] = None):
        self.s = settings
        self.conn = conn
        self._stop = threading.Event()
        self._tick = 0
        self.on_change: Optional[Callable[[str], None]] = None
        self.logger = logger

    def start(self) -> None:
        t = threading.Thread(target=self._run, name="minotaur-sweeper", daemon=True)
        t.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._sweep_timeouts()
                self._grant_if_idle()
            except Exception:
                # swallow; keep sweeping
                pass
            time.sleep(1.0)

    # Grant queued trial if no running exists
    def _grant_if_idle(self) -> None:
        emitted: Optional[str] = None
        with dbm.tx(self.conn):
            run = dbm.query_one(self.conn, "SELECT id FROM challenges WHERE status='running' LIMIT 1")
            if run is not None:
                return
            row = dbm.query_one(
                self.conn,
                "SELECT id, ticket, base_priority, effective_priority FROM challenges WHERE status='queued' ORDER BY effective_priority DESC, enqueued_at ASC LIMIT 1",
            )
            if row is None:
                return
            sid = str(uuid.uuid4())
            lease = datetime.now(timezone.utc) + timedelta(seconds=self.s.trial_ttl_sec)
            cur = self.conn.execute(
                "UPDATE challenges SET status='running', started_at=?, lease_expire_at=?, session_id=? WHERE id=? AND status='queued'",
                (utcnow_str(), lease.isoformat().replace("+00:00", "Z"), sid, row["id"]),
            )
            if cur.rowcount:
                emitted = "grant"
                if self.logger:
                    try:
                        self.logger.write({
                            "ev": "scheduler",
                            "action": "grant",
                            "challenge_id": int(row["id"]),
                            "ticket": row["ticket"],
                            "base_priority": int(row["base_priority"]),
                            "effective_priority": int(row["effective_priority"]),
                        })
                    except Exception:
                        pass
        if emitted and self.on_change:
            try:
                self.on_change(emitted)
            except Exception:
                pass

    # Mark timed-out
    def _sweep_timeouts(self) -> None:
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        emitted: Optional[str] = None
        with dbm.tx(self.conn):
            r = dbm.query_one(
                self.conn,
                "SELECT id, ticket, agent_name, agent_id FROM challenges WHERE status='running' AND lease_expire_at < ? LIMIT 1",
                (now,),
            )
            cur = self.conn.execute(
                "UPDATE challenges SET status='timeout', finished_at=? WHERE status='running' AND lease_expire_at < ?",
                (now, now),
            )
            if cur.rowcount:
                emitted = "timeout"
                if self.logger:
                    try:
                        self.logger.write({
                            "ev": "scheduler",
                            "action": "timeout",
                            "challenge_id": int(r["id"]) if r and r["id"] is not None else None,
                            "ticket": r["ticket"] if r else None,
                            "agent_name": r["agent_name"] if r else None,
                            "agent_id": r["agent_id"] if r else None,
                        })
                    except Exception:
                        pass
        if emitted and self.on_change:
            try:
                self.on_change(emitted)
            except Exception:
                pass
