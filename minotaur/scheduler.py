from __future__ import annotations

import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from .config import Settings
from . import db as dbm


def utcnow_str() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class Coordinator:
    """Simple single-threaded sweeper for granting and timeouts.

    Designed for single-process deployment. Uses SQLite IMMEDIATE transactions
    to ensure at most 1 running trial.
    """

    def __init__(self, settings: Settings, conn):
        self.s = settings
        self.conn = conn
        self._stop = threading.Event()
        self._tick = 0

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
        with dbm.tx(self.conn):
            run = dbm.query_one(self.conn, "SELECT id FROM trials WHERE status='running' LIMIT 1")
            if run is not None:
                return
            row = dbm.query_one(
                self.conn,
                "SELECT id, ticket, base_priority, effective_priority FROM trials WHERE status='queued' ORDER BY effective_priority DESC, enqueued_at ASC LIMIT 1",
            )
            if row is None:
                return
            sid = str(uuid.uuid4())
            lease = datetime.now(timezone.utc) + timedelta(seconds=self.s.trial_ttl_sec)
            self.conn.execute(
                "UPDATE trials SET status='running', started_at=?, lease_expire_at=?, session_id=? WHERE id=? AND status='queued'",
                (utcnow_str(), lease.isoformat().replace("+00:00", "Z"), sid, row["id"]),
            )

    # Mark timed-out
    def _sweep_timeouts(self) -> None:
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        with dbm.tx(self.conn):
            self.conn.execute(
                "UPDATE trials SET status='timeout', finished_at=? WHERE status='running' AND lease_expire_at < ?",
                (now, now),
            )

