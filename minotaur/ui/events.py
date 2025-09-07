from __future__ import annotations

from flask import Blueprint, Response, jsonify, request, stream_with_context
from ..context import AppCtx
from ..sched_fair import accumulate_service
from .. import db as dbm


def create_events_bp(ctx: AppCtx) -> Blueprint:
    bp = Blueprint("ui_events", __name__, url_prefix="/minotaur")
    guard = ctx.ui_guard

    @bp.route("/stream")
    @guard.require()
    def stream():
        headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
        return Response(stream_with_context(ctx.bus.stream()), headers=headers)

    @bp.route("/agent_count")
    @guard.require()
    def agent_count():
        row = ctx.conn.execute(
            "SELECT COUNT(DISTINCT COALESCE(agent_name, ticket)) AS n FROM challenges WHERE status IN ('queued','running')"
        ).fetchone()
        n = int(row["n"]) if row else 0
        return jsonify({"n": n})

    @bp.route("/cancel_running", methods=["POST"])
    @guard.require()
    def cancel_running():
        from ..app import utcnow_str
        now = utcnow_str()
        with ctx.conn:
            cur = ctx.conn.execute(
                "SELECT id, agent_name, started_at FROM challenges WHERE status='running' LIMIT 1"
            ).fetchone()
            if cur is not None:
                ctx.conn.execute(
                    "INSERT INTO challenge_requests(challenge_id, api, req_key, phase, status_code, req_body, res_body, ts) VALUES(?,?,?,?,?,?,?,?)",
                    (int(cur["id"]), "event", "cancel", "cancel", 0, "{}", "{}", now),
                )
            ctx.conn.execute(
                "UPDATE challenges SET status='terminated_running', finished_at=? WHERE status='running'",
                (now,),
            )
        try:
            if cur is not None:
                accumulate_service(ctx.conn, ctx.logger, ctx.s.base_priority_default, cur["agent_name"], cur["started_at"], now)
        except Exception:
            pass
        try:
            ctx.bus.emit("cancel")
        except Exception:
            pass
        if ctx.coord and ctx.coord.on_change:
            try:
                ctx.coord.on_change("cancel")
            except Exception:
                pass
        return jsonify({"ok": True})

    @bp.route("/cancel_queued", methods=["POST"])
    @guard.require()
    def cancel_queued():
        from ..app import utcnow_str
        try:
            body = request.get_json(silent=True) or {}
        except Exception:
            body = {}
        ticket = body.get("ticket")
        id_ = body.get("id")
        now = utcnow_str()
        cancelled = False
        with dbm.tx(ctx.conn):
            if ticket:
                cur = ctx.conn.execute(
                    "SELECT id FROM challenges WHERE status='queued' AND ticket=? LIMIT 1",
                    (ticket,),
                ).fetchone()
            elif id_ is not None:
                try:
                    id_i = int(id_)
                except Exception:
                    id_i = None
                cur = (
                    ctx.conn.execute(
                        "SELECT id FROM challenges WHERE status='queued' AND id=? LIMIT 1",
                        (id_i,),
                    ).fetchone()
                    if id_i is not None
                    else None
                )
            else:
                cur = None
            if cur is not None:
                try:
                    ctx.conn.execute(
                        "INSERT INTO challenge_requests(challenge_id, api, req_key, phase, status_code, req_body, res_body, ts) VALUES(?,?,?,?,?,?,?,?)",
                        (int(cur["id"]), "event", "cancel_queued", "cancel", 0, "{}", "{}", now),
                    )
                except Exception:
                    pass
                ctx.conn.execute(
                    "UPDATE challenges SET status='cancelled_queue', finished_at=? WHERE id=?",
                    (now, int(cur["id"]))
                )
                cancelled = True
        try:
            ctx.bus.emit("cancel")
        except Exception:
            pass
        if ctx.coord and ctx.coord.on_change:
            try:
                ctx.coord.on_change("cancel")
            except Exception:
                pass
        return jsonify({"ok": True, "cancelled": cancelled})

    return bp

