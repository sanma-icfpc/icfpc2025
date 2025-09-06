from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import sqlite3


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_iso(ts: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def min_v(conn: sqlite3.Connection) -> float:
    r = conn.execute("SELECT MIN(vtime) AS v FROM agent_priorities").fetchone()
    return float(r["v"]) if r and r["v"] is not None else 0.0


def priority_weight(conn: sqlite3.Connection, default_priority: int, name: Optional[str]) -> int:
    if name:
        r = conn.execute("SELECT priority FROM agent_priorities WHERE name=?", (name,)).fetchone()
        if r is not None:
            try:
                return max(1, int(r["priority"]))
            except Exception:
                pass
    r = conn.execute("SELECT priority FROM agent_priorities WHERE name='*'").fetchone()
    if r is not None:
        try:
            return max(1, int(r["priority"]))
        except Exception:
            pass
    return max(1, int(default_priority))


def ensure_agent_entry(conn: sqlite3.Connection, default_priority: int, name: str) -> None:
    # Create missing row with default priority and baseline vtime=min_v
    r = conn.execute("SELECT 1 FROM agent_priorities WHERE name=?", (name,)).fetchone()
    if r is None:
        base = priority_weight(conn, default_priority, name)
        base_v = min_v(conn)
        conn.execute(
            "INSERT INTO agent_priorities(name, priority, vtime, service, pinned, updated_at) VALUES(?,?,?,?,0,?)",
            (name, base, float(base_v), 0.0, _now_iso()),
        )


def accumulate_service(
    conn: sqlite3.Connection,
    logger,
    default_priority: int,
    name: Optional[str],
    started_at: Optional[str],
    finished_at: Optional[str] = None,
) -> None:
    if not name:
        name = "*"
    try:
        row = conn.execute(
            "SELECT pinned, vtime, service, priority FROM agent_priorities WHERE name=?",
            (name,),
        ).fetchone()
        if row is None:
            ensure_agent_entry(conn, default_priority, name)
            row = conn.execute(
                "SELECT pinned, vtime, service, priority FROM agent_priorities WHERE name=?",
                (name,),
            ).fetchone()
        if row is None:
            return
        if int(row["pinned"]) == 1:
            return  # do not accumulate for pinned agent
        st = _parse_iso(started_at) if started_at else None
        ft = _parse_iso(finished_at) if finished_at else datetime.now(timezone.utc)
        if not st or not ft:
            return
        delta = max(0.0, (ft - st).total_seconds())
        if delta <= 0:
            return
        w = max(1, int(row["priority"]))
        new_service = float(row["service"] or 0.0) + delta
        new_v = float(row["vtime"] or 0.0) + (delta / float(w))
        with conn:
            conn.execute(
                "UPDATE agent_priorities SET service=?, vtime=?, updated_at=? WHERE name=?",
                (new_service, new_v, _now_iso(), name),
            )
        try:
            logger.write({
                "ev": "scheduler",
                "action": "cfs_update",
                "name": name,
                "delta": delta,
                "weight": w,
                "service": new_service,
                "vtime": new_v,
            })
        except Exception:
            pass
    except Exception:
        pass


def pinned_name(conn: sqlite3.Connection) -> Optional[str]:
    r = conn.execute("SELECT name FROM agent_priorities WHERE pinned=1 LIMIT 1").fetchone()
    return r["name"] if r is not None else None


def set_pinned(conn: sqlite3.Connection, name: Optional[str]) -> None:
    with conn:
        conn.execute("UPDATE agent_priorities SET pinned=0")
        if name:
            conn.execute(
                "UPDATE agent_priorities SET pinned=1, updated_at=? WHERE name=?",
                (_now_iso(), name),
            )

