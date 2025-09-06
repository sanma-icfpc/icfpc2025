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
        if bool(out.get("correct")):
            with ctx.conn:
                rr = ctx.conn.execute("SELECT id, agent_name, started_at, upstream_query_count FROM challenges WHERE status='running' LIMIT 1").fetchone()
                # Record final score_query_count as the latest upstream_query_count at success time
                final_qc = int(rr["upstream_query_count"]) if rr and rr["upstream_query_count"] is not None else None
                ctx.conn.execute(
                    "UPDATE challenges SET status='success', finished_at=?, score_query_count=? WHERE status='running'",
                    (utcnow_str(), final_qc),
                )
            try:
                if rr is not None:
                    accumulate_service(ctx.conn, ctx.logger, ctx.s.base_priority_default, rr["agent_name"], rr["started_at"], utcnow_str())
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
