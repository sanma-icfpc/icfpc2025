from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from flask import jsonify, make_response, request

from .context import AppCtx
from .proxy import UpstreamError
from .sched_fair import accumulate_service


def utcnow_str() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _log_ch_req(ctx: AppCtx, ch_id: int, api: str, req_key: str, phase: str, status_code: int, req_obj: Dict[str, Any], res_obj: Dict[str, Any]) -> None:
    try:
        ctx.conn.execute(
            "INSERT INTO challenge_requests(challenge_id, api, req_key, phase, status_code, req_body, res_body, ts) VALUES(?,?,?,?,?,?,?,?)",
            (int(ch_id), api, req_key, phase, int(status_code), json.dumps(req_obj, ensure_ascii=False), json.dumps(res_obj, ensure_ascii=False), utcnow_str()),
        )
    except Exception:
        pass


def _current_running_session(ctx: AppCtx) -> Optional[str]:
    row = ctx.conn.execute("SELECT session_id FROM challenges WHERE status='running' LIMIT 1").fetchone()
    return row["session_id"] if row and row["session_id"] else None


def _touch_lease(ctx: AppCtx) -> None:
    lease = (datetime.now(timezone.utc) + timedelta(seconds=ctx.s.trial_ttl_sec)).isoformat().replace("+00:00", "Z")
    with ctx.conn:
        ctx.conn.execute("UPDATE challenges SET lease_expire_at=? WHERE status='running'", (lease,))


def register_api_routes(app, ctx: AppCtx) -> None:
    @app.route("/explore", methods=["POST"])
    def explore_route():
        body = request.get_json(silent=True) or {}
        agent_id = request.headers.get("X-Agent-ID") or None
        git_sha = request.headers.get("X-Agent-Git") or None
        sid = _current_running_session(ctx)
        if not sid:
            return jsonify({"error": "no active session"}), 409
        state = {
            "query_count": (
                ctx.conn.execute("SELECT upstream_query_count FROM challenges WHERE status='running'").fetchone()
                or {"upstream_query_count": 0}
            )["upstream_query_count"]
        }
        rk = str(uuid.uuid4())
        cur = ctx.conn.execute("SELECT id FROM challenges WHERE status='running' LIMIT 1").fetchone()
        ch_id = int(cur["id"]) if cur else None
        if ch_id is not None:
            _log_ch_req(ctx, ch_id, "explore", rk, "agent_in", 0, body, {})
            ctx.bus.emit("change")
            _log_ch_req(ctx, ch_id, "explore", rk, "to_upstream", 0, body, {})
            ctx.bus.emit("change")
        try:
            out = ctx.proxy.forward("/explore", sid, body, state, meta={"agent_id": agent_id, "git_sha": git_sha})
        except UpstreamError as ue:
            if ch_id is not None:
                _log_ch_req(ctx, ch_id, "explore", rk, "from_upstream", int(ue.status), body, ue.payload)
                ctx.bus.emit("change")
            return jsonify({"error": "upstream_error", "detail": ue.payload}), 502
        with ctx.conn:
            ctx.conn.execute(
                "UPDATE challenges SET upstream_query_count=? WHERE status='running'",
                (int(out.get("queryCount", 0)),),
            )
        _touch_lease(ctx)
        if ch_id is not None:
            _log_ch_req(ctx, ch_id, "explore", rk, "from_upstream", 200, body, out)
            ctx.bus.emit("change")
            _log_ch_req(ctx, ch_id, "explore", rk, "agent_out", 200, body, out)
            ctx.bus.emit("change")
        ctx.bus.emit("explore")
        return jsonify(out)



    @app.route("/minotaur_giveup", methods=["POST"])
    def minotaur_giveup_route():
        """Cancel the currently running challenge if its agent_name matches X-Agent-Name.

        This is a Minotaur-only helper (not part of the official API) to allow
        an agent to voluntarily give up its current run without enqueueing another select.
        """
        agent_name = request.headers.get("X-Agent-Name") or None
        if not agent_name:
            return jsonify({"ok": False, "error": "missing_agent_name"}), 400
        now = utcnow_str()
        cancelled = False
        cur = None
        with ctx.conn:
            cur = ctx.conn.execute(
                "SELECT id, agent_name, started_at FROM challenges WHERE status='running' AND agent_name=? LIMIT 1",
                (agent_name,),
            ).fetchone()
            if cur is not None:
                try:
                    ctx.conn.execute(
                        "INSERT INTO challenge_requests(challenge_id, api, req_key, phase, status_code, req_body, res_body, ts) VALUES(?,?,?,?,?,?,?,?)",
                        (int(cur["id"]), "event", "cancel", "cancel", 0, "{}", "{}", now),
                    )
                except Exception:
                    pass
                ctx.conn.execute(
                    "UPDATE challenges SET status='terminated_running', finished_at=? WHERE id=? AND status='running'",
                    (now, int(cur["id"])),
                )
                cancelled = True
        if cancelled:
            try:
                accumulate_service(ctx.conn, ctx.logger, ctx.s.base_priority_default, cur["agent_name"], cur["started_at"], now)
            except Exception:
                pass
            if ctx.bus:
                try:
                    ctx.bus.emit("cancel")
                except Exception:
                    pass
        return jsonify({"ok": True, "cancelled": bool(cancelled)})

    @app.route("/guess", methods=["POST"])
    def guess_route():
        body = request.get_json(silent=True) or {}
        agent_id = request.headers.get("X-Agent-ID") or None
        git_sha = request.headers.get("X-Agent-Git") or None
        sid = _current_running_session(ctx)
        if not sid:
            return jsonify({"error": "no active session"}), 409
        rk = str(uuid.uuid4())
        cur = ctx.conn.execute("SELECT id FROM challenges WHERE status='running' LIMIT 1").fetchone()
        ch_id = int(cur["id"]) if cur else None
        if ch_id is not None:
            _log_ch_req(ctx, ch_id, "guess", rk, "agent_in", 0, body, {})
            ctx.bus.emit("change")
            _log_ch_req(ctx, ch_id, "guess", rk, "to_upstream", 0, body, {})
            ctx.bus.emit("change")
        try:
            out = ctx.proxy.forward("/guess", sid, body, meta={"agent_id": agent_id, "git_sha": git_sha})
        except UpstreamError as ue:
            if ch_id is not None:
                _log_ch_req(ctx, ch_id, "guess", rk, "from_upstream", int(ue.status), body, ue.payload)
                ctx.bus.emit("change")
            return jsonify({"error": "upstream_error", "detail": ue.payload}), 502
        # Regardless of correctness, upstream /guess ends the selection window.
        # Mark the challenge finished. Use 'correct' for correct guesses; otherwise 'incorrect'.
        is_correct = bool(out.get("correct"))
        with ctx.conn:
            rr = ctx.conn.execute(
                "SELECT id, agent_name, started_at, upstream_query_count FROM challenges WHERE status='running' LIMIT 1"
            ).fetchone()
            final_qc = int(rr["upstream_query_count"]) if rr and rr["upstream_query_count"] is not None else None
            if is_correct:
                ctx.conn.execute(
                    "UPDATE challenges SET status='correct', finished_at=?, score_query_count=? WHERE status='running'",
                    (utcnow_str(), final_qc),
                )
            else:
                # Not correct: still finish the challenge window
                ctx.conn.execute(
                    "UPDATE challenges SET status='incorrect', finished_at=? WHERE status='running'",
                    (utcnow_str(),),
                )
        # Accumulate fair-share service time for this run
        try:
            if rr is not None:
                accumulate_service(
                    ctx.conn,
                    ctx.logger,
                    ctx.s.base_priority_default,
                    rr["agent_name"],
                    rr["started_at"],
                    utcnow_str(),
                )
        except Exception:
            pass
        _touch_lease(ctx)
        if ch_id is not None:
            _log_ch_req(ctx, ch_id, "guess", rk, "from_upstream", 200, body, out)
            ctx.bus.emit("change")
            _log_ch_req(ctx, ch_id, "guess", rk, "agent_out", 200, body, out)
            ctx.bus.emit("change")
        ctx.bus.emit("guess")
        return jsonify(out)
