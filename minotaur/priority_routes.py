from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional

from flask import jsonify, render_template, request

from .context import AppCtx
from .api_select import _recalc_priorities


def utcnow_str() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _agent_status(ctx: AppCtx, name: str) -> str:
    r = ctx.conn.execute(
        "SELECT 1 FROM challenges WHERE status='running' AND agent_name=? LIMIT 1",
        (name,),
    ).fetchone()
    if r is not None:
        return "running"
    q = ctx.conn.execute(
        "SELECT 1 FROM challenges WHERE status='queued' AND agent_name=? LIMIT 1",
        (name,),
    ).fetchone()
    return "queued" if q is not None else "inactive"


def _default_priority(ctx: AppCtx) -> int:
    r = ctx.conn.execute("SELECT priority FROM agent_priorities WHERE name='*'").fetchone()
    return int(r["priority"]) if r else int(ctx.s.base_priority_default)


def register_priority_routes(app, ctx: AppCtx) -> None:
    guard = ctx.ui_guard

    @app.route("/minotaur/priorities")
    @guard.require()
    def ui_priorities():
        sort = request.args.get("sort") == "1"
        # Collect names: from agent_priorities plus any seen in challenges
        names = set()
        for r in ctx.conn.execute("SELECT name FROM agent_priorities").fetchall():
            names.add(r["name"])
        for r in ctx.conn.execute(
            "SELECT DISTINCT agent_name AS name FROM challenges WHERE agent_name IS NOT NULL AND status IN ('queued','running')"
        ).fetchall():
            if r["name"]:
                names.add(r["name"])
        rows: List[Dict] = []
        for n in names:
            pr = ctx.conn.execute("SELECT priority, vtime, service, pinned FROM agent_priorities WHERE name=?", (n,)).fetchone()
            p = int(pr["priority"]) if pr else _default_priority(ctx) if n != "*" else _default_priority(ctx)
            v = float(pr["vtime"]) if pr and pr["vtime"] is not None else 0.0
            sv = float(pr["service"]) if pr and pr["service"] is not None else 0.0
            pinned = int(pr["pinned"]) if pr and pr["pinned"] is not None else 0
            rows.append({"name": n, "priority": p, "vtime": v, "service": sv, "status": _agent_status(ctx, n) if n != "*" else "inactive", "pinned": pinned})
        # Ensure default row exists and is last
        if "*" not in names:
            rows.append({"name": "*", "priority": _default_priority(ctx), "status": "inactive"})
        if sort:
            rows.sort(key=lambda x: (x["priority"], x["name"] != "*", x["name"]))
            rows.reverse()
        else:
            rows.sort(key=lambda x: (x["name"] == "*", x["name"]))  # name order, default last
        return render_template("priority_pane.html", rows=rows)

    @app.route("/minotaur/priorities/apply", methods=["POST"])
    @guard.require()
    def ui_priorities_apply():
        data = request.get_json(silent=True) or {}
        items = data.get("items") or []
        now = utcnow_str()
        with ctx.conn:
            for it in items:
                name = (it.get("name") or "").strip()
                if not name:
                    continue
                prio = int(it.get("priority") or 0)
                prio = max(1, min(100, prio))
                ctx.conn.execute(
                    "INSERT INTO agent_priorities(name, priority, updated_at) VALUES(?,?,?) ON CONFLICT(name) DO UPDATE SET priority=excluded.priority, updated_at=excluded.updated_at",
                    (name, prio, now),
                )
        # Recalculate queued base/effective priorities based on new mapping
        rows = ctx.conn.execute(
            "SELECT id, agent_name, enqueued_at FROM challenges WHERE status='queued' ORDER BY enqueued_at ASC"
        ).fetchall()
        for r in rows:
            nm = r["agent_name"]
            pr = ctx.conn.execute("SELECT priority FROM agent_priorities WHERE name=?", (nm,)).fetchone()
            if pr is None:
                pr = ctx.conn.execute("SELECT priority FROM agent_priorities WHERE name='*'").fetchone()
            base = int(pr["priority"]) if pr else int(ctx.s.base_priority_default)
            with ctx.conn:
                ctx.conn.execute("UPDATE challenges SET base_priority=? WHERE id=?", (base, int(r["id"])) )
        _recalc_priorities(ctx)
        if ctx.coord and ctx.coord.on_change:
            try:
                ctx.coord.on_change("priorities_apply")
            except Exception:
                pass
        return jsonify({"ok": True})

    @app.route("/minotaur/priorities/data")
    @guard.require()
    def ui_priorities_data():
        names = set()
        for r in ctx.conn.execute("SELECT name FROM agent_priorities").fetchall():
            names.add(r["name"])
        for r in ctx.conn.execute(
            "SELECT DISTINCT agent_name AS name FROM challenges WHERE agent_name IS NOT NULL AND status IN ('queued','running')"
        ).fetchall():
            if r["name"]:
                names.add(r["name"])
        rows = []
        for n in names:
            pr = ctx.conn.execute("SELECT priority, vtime, service, pinned FROM agent_priorities WHERE name=?", (n,)).fetchone()
            p = int(pr["priority"]) if pr else _default_priority(ctx) if n != "*" else _default_priority(ctx)
            v = float(pr["vtime"]) if pr and pr["vtime"] is not None else 0.0
            sv = float(pr["service"]) if pr and pr["service"] is not None else 0.0
            pinned = int(pr["pinned"]) if pr and pr["pinned"] is not None else 0
            rows.append({"name": n, "priority": p, "vtime": v, "service": sv, "status": _agent_status(ctx, n) if n != "*" else "inactive", "pinned": pinned})
        return jsonify({"rows": rows})

    @app.route("/minotaur/priorities/pin", methods=["POST"])
    @guard.require()
    def ui_priorities_pin():
        data = request.get_json(silent=True) or {}
        name = (data.get("name") or "").strip()
        pin = bool(data.get("pinned"))
        if not name or name == "*":
            return jsonify({"ok": False, "error": "invalid_name"}), 400
        # Only one persistent pin (pinned=1) can be active; do not clear one-shot (pinned=2)
        with ctx.conn:
            ctx.conn.execute("UPDATE agent_priorities SET pinned=0 WHERE pinned=1")
            if pin:
                ctx.conn.execute(
                    "INSERT INTO agent_priorities(name, priority, updated_at, vtime, service, pinned) VALUES(?, (SELECT priority FROM agent_priorities WHERE name='*'), datetime('now'), 0.0, 0.0, 1) ON CONFLICT(name) DO UPDATE SET pinned=1",
                    (name,),
                )
        if ctx.coord and ctx.coord.on_change:
            try:
                ctx.coord.on_change("priorities_pin")
            except Exception:
                pass
        return jsonify({"ok": True})

    @app.route("/minotaur/priorities/delete", methods=["POST"])
    @guard.require()
    def ui_priorities_delete():
        data = request.get_json(silent=True) or {}
        name = (data.get("name") or "").strip()
        if not name or name == "*":
            return jsonify({"ok": False, "error": "invalid_name"}), 400
        with ctx.conn:
            ctx.conn.execute("DELETE FROM agent_priorities WHERE name=?", (name,))
        if ctx.coord and ctx.coord.on_change:
            try:
                ctx.coord.on_change("priorities_delete")
            except Exception:
                pass
        return jsonify({"ok": True})

    @app.route("/minotaur/priorities/select_next", methods=["POST"])
    @guard.require()
    def ui_priorities_select_next():
        data = request.get_json(silent=True) or {}
        name = (data.get("name") or "").strip()
        if not name or name == "*":
            return jsonify({"ok": False, "error": "invalid_name"}), 400
        # Ensure at most one one-shot (pinned=2) exists; then mark this agent as one-shot
        with ctx.conn:
            # Clear any existing one-shot selection(s) but keep persistent pins as-is
            ctx.conn.execute("UPDATE agent_priorities SET pinned=0 WHERE pinned=2")
            ctx.conn.execute(
                "INSERT INTO agent_priorities(name, priority, updated_at, vtime, service, pinned) VALUES(?, (SELECT priority FROM agent_priorities WHERE name='*'), datetime('now'), 0.0, 0.0, 2) ON CONFLICT(name) DO UPDATE SET pinned=2",
                (name,),
            )
        if ctx.coord and ctx.coord.on_change:
            try:
                ctx.coord.on_change("priorities_select_next")
            except Exception:
                pass
        return jsonify({"ok": True})
