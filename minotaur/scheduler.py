from __future__ import annotations

import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional, Callable, List, Dict
from .logging import JsonlLogger

from .config import Settings
from . import db as dbm
from .sched_fair import priority_weight, ensure_agent_entry, pick_pinned


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
        # When a persistent pin exists but the agent has nothing queued,
        # briefly hold grants to allow the pinned agent to enqueue again.
        # This reduces races where others get granted between back-to-back
        # runs of the pinned agent.
        self._pin_hold_until: float = 0.0
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
            did_cancel_any = False
            def _next_queued_for_name(name: Optional[str]):
                # Fetch earliest queued row for agent name (or anonymous '*'), skipping ones marked by cancel_queued event.
                while True:
                    if name == '*':
                        rr = dbm.query_one(
                            self.conn,
                            "SELECT id, ticket FROM challenges WHERE status='queued' AND agent_name IS NULL ORDER BY enqueued_at ASC LIMIT 1",
                        )
                    else:
                        rr = dbm.query_one(
                            self.conn,
                            "SELECT id, ticket FROM challenges WHERE status='queued' AND agent_name=? ORDER BY enqueued_at ASC LIMIT 1",
                            (name,),
                        )
                    if rr is None:
                        return None
                    # Skip if a queued-cancel event exists for this challenge; convert it to cancelled now
                    canc = dbm.query_one(
                        self.conn,
                        "SELECT 1 FROM challenge_requests WHERE challenge_id=? AND api='event' AND req_key='cancel_queued' LIMIT 1",
                        (int(rr["id"]),),
                    )
                    if canc is not None:
                        try:
                            self.conn.execute(
                                "UPDATE challenges SET status='cancelled_queue', finished_at=? WHERE id=? AND status='queued'",
                                (utcnow_str(), int(rr["id"])),
                            )
                        except Exception:
                            pass
                        nonlocal did_cancel_any
                        did_cancel_any = True
                        # Loop and try next queued row
                        continue
                    return rr
            # Pinned override
            pin_name, pin_one_shot = pick_pinned(self.conn)
            chosen_name: Optional[str] = None
            chosen_id: Optional[int] = None
            chosen_ticket: Optional[str] = None
            if pin_name:
                r = _next_queued_for_name(pin_name)
                if r is not None:
                    chosen_name = pin_name
                    chosen_id = int(r["id"])
                    chosen_ticket = r["ticket"]
                    if self.logger:
                        try:
                            self.logger.write({
                                "ev": "scheduler",
                                "action": "pinned_override",
                                "agent_name": pin_name,
                                "one_shot": pin_one_shot,
                                "challenge_id": chosen_id,
                                "ticket": chosen_ticket,
                            })
                        except Exception:
                            pass
                # If a one-shot selection exists but it's not in queue, try falling back to persistent pin
                elif pin_one_shot:
                    try:
                        r2 = dbm.query_one(
                            self.conn,
                            "SELECT name FROM agent_priorities WHERE pinned=1 LIMIT 1",
                        )
                        if r2 and r2["name"]:
                            rr = _next_queued_for_name(r2["name"])
                            if rr is not None:
                                chosen_name = r2["name"]
                                chosen_id = int(rr["id"])
                                chosen_ticket = rr["ticket"]
                                if self.logger:
                                    try:
                                        self.logger.write({
                                            "ev": "scheduler",
                                            "action": "pinned_fallback",
                                            "agent_name": chosen_name,
                                            "one_shot_present": True,
                                            "challenge_id": chosen_id,
                                            "ticket": chosen_ticket,
                                        })
                                    except Exception:
                                        pass
                    except Exception:
                        pass
                # Persistent pin present but nothing queued: optionally hold for a short window
                elif not pin_one_shot:
                    try:
                        import time as _t
                        now = _t.time()
                        if self._pin_hold_until <= 0:
                            self._pin_hold_until = now + float(self.s.pin_hold_sec)
                            return  # defer this grant cycle
                        if now < self._pin_hold_until:
                            return  # still within hold window; defer
                        # hold elapsed; clear and continue to fair selection fallback
                        self._pin_hold_until = 0.0
                    except Exception:
                        pass
            if chosen_id is None:
                # Build candidates by agent name (queued only); treat NULL agent_name as '*'
                qrows = dbm.query_all(
                    self.conn,
                    "SELECT COALESCE(agent_name,'*') AS name, MIN(enqueued_at) AS first_enq, COUNT(1) AS n FROM challenges WHERE status='queued' GROUP BY COALESCE(agent_name,'*')",
                )
                if not qrows:
                    return
                # Ensure entries exist and compute selection key: (vtime, first_enq)
                # Fetch current vtime via agent_priorities table
                # Also include weight for logging
                candidates: List[Dict] = []
                # Ensure min_v init through ensure_agent_entry per seen name
                for r in qrows:
                    nm = r["name"]
                    ensure_agent_entry(self.conn, self.s.base_priority_default, nm)
                for r in qrows:
                    nm = r["name"]
                    apr = dbm.query_one(
                        self.conn,
                        "SELECT vtime, priority FROM agent_priorities WHERE name=?",
                        (nm,),
                    )
                    vt = float(apr["vtime"]) if apr and apr["vtime"] is not None else 0.0
                    w = priority_weight(self.conn, self.s.base_priority_default, nm)
                    candidates.append({
                        "name": nm,
                        "vtime": vt,
                        "weight": w,
                        "first_enq": r["first_enq"],
                        "count": int(r["n"]),
                    })
                # Log candidate snapshot before pick
                if self.logger:
                    try:
                        self.logger.write({
                            "ev": "scheduler",
                            "action": "cfs_candidates",
                            "candidates": candidates[:50],
                        })
                    except Exception:
                        pass
                candidates.sort(key=lambda x: (x["vtime"], x["first_enq"]))
                pick = candidates[0]
                chosen_name = pick["name"]
                # Grant earliest enqueued for that agent, skipping cancel_queued-marked rows
                rr = _next_queued_for_name(chosen_name)
                if rr is None:
                    return
                chosen_id = int(rr["id"])
                chosen_ticket = rr["ticket"]
                if self.logger:
                    try:
                        self.logger.write({
                            "ev": "scheduler",
                            "action": "cfs_pick",
                            "picked_name": chosen_name,
                            "picked_ticket": chosen_ticket,
                            "candidates": candidates[:50],
                        })
                    except Exception:
                        pass
            # Grant chosen
            sid = str(uuid.uuid4())
            lease = datetime.now(timezone.utc) + timedelta(seconds=self.s.trial_ttl_sec)
            cur = self.conn.execute(
                "UPDATE challenges SET status='running', started_at=?, lease_expire_at=?, session_id=? WHERE id=? AND status='queued'",
                (utcnow_str(), lease.isoformat().replace("+00:00", "Z"), sid, chosen_id),
            )
            if cur.rowcount:
                emitted = "grant"
                # If this was a one-shot pin, clear it now
                if pin_name and pin_one_shot:
                    try:
                        # Clear within the same transaction; avoid nested transactions which roll back the grant
                        self.conn.execute(
                            "UPDATE agent_priorities SET pinned=0 WHERE name=?",
                            (pin_name,),
                        )
                    except Exception:
                        pass
                if self.logger:
                    try:
                        self.logger.write({
                            "ev": "scheduler",
                            "action": "grant",
                            "challenge_id": int(chosen_id),
                            "ticket": chosen_ticket,
                            "agent_name": chosen_name,
                        })
                    except Exception:
                        pass
        if (emitted or did_cancel_any) and self.on_change:
            try:
                self.on_change(emitted or "cancel_queued")
            except Exception:
                pass

    # Mark timed-out
    def _sweep_timeouts(self) -> None:
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        emitted: Optional[str] = None
        with dbm.tx(self.conn):
            r = dbm.query_one(
                self.conn,
                "SELECT id, ticket, agent_name, agent_id, started_at FROM challenges WHERE status='running' AND lease_expire_at < ? LIMIT 1",
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
                # accumulate service for the timed out run
                try:
                    from .sched_fair import accumulate_service
                    if r is not None:
                        accumulate_service(self.conn, self.logger, self.s.base_priority_default, r["agent_name"], r["started_at"], now)
                except Exception:
                    pass
        if emitted and self.on_change:
            try:
                self.on_change(emitted)
            except Exception:
                pass
