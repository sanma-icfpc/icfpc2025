from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

from flask import jsonify, make_response, request

from .context import AppCtx
from .proxy import UpstreamError


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


def _compute_effective_priority(ctx: AppCtx, base: int, enqueued_at: str) -> int:
    try:
        enq = datetime.fromisoformat(enqueued_at.replace("Z", "+00:00"))
        mins = int((datetime.now(timezone.utc) - enq).total_seconds() // 60)
        ageing = min(ctx.s.ageing_cap, max(0, mins // ctx.s.ageing_every_min))
    except Exception:
        ageing = 0
    return base + ageing


def _recalc_priorities(ctx: AppCtx) -> None:
    rows = ctx.conn.execute(
        "SELECT id, base_priority, enqueued_at FROM challenges WHERE status='queued'"
    ).fetchall()
    with ctx.conn:
        for r in rows:
            eff = _compute_effective_priority(ctx, int(r["base_priority"]), r["enqueued_at"])
            ctx.conn.execute("UPDATE challenges SET effective_priority=? WHERE id=?", (eff, r["id"]))


def _grant_waiting_if_idle(ctx: AppCtx) -> None:
    _recalc_priorities(ctx)
    time.sleep(0.05)


def register_select_routes(app, ctx: AppCtx) -> None:
    def _agent_priority(name: str | None) -> int:
        if name:
            r = ctx.conn.execute(
                "SELECT priority FROM agent_priorities WHERE name=?",
                (name,),
            ).fetchone()
            if r is not None:
                return int(r["priority"])
        r = ctx.conn.execute(
            "SELECT priority FROM agent_priorities WHERE name='*'",
        ).fetchone()
        if r is not None:
            return int(r["priority"])
        return int(ctx.s.base_priority_default)
    @app.route("/select", methods=["POST"])
    def select_route():
        # If scheduler is in drain mode (e.g., memory threshold crossed),
        # stop accepting new selections to avoid building a queue that will
        # be interrupted on restart. Let clients retry soon.
        try:
            if hasattr(ctx, 'coord') and getattr(ctx.coord, 'is_paused', None) and ctx.coord.is_paused():
                resp = make_response((jsonify({"error": "draining", "message": "server is draining; retry later"}), 503))
                try:
                    resp.headers['Retry-After'] = '60'
                except Exception:
                    pass
                return resp
        except Exception:
            pass
        body = request.get_json(silent=True) or {}
        problem = body.get("problemName")
        if not isinstance(problem, str) or not problem:
            return jsonify({"error": "problemName required"}), 400

        agent_id = request.headers.get("X-Agent-ID") or None
        agent_name = request.headers.get("X-Agent-Name") or None
        git_sha = request.headers.get("X-Agent-Git") or None

        base_prio = _agent_priority(agent_name)
        ticket = str(uuid.uuid4())
        enq = utcnow_str()
        eff = base_prio
        with ctx.conn:
            ctx.conn.execute(
                "INSERT INTO challenges(ticket, problem_id, agent_id, agent_name, source_ip, git_sha, base_priority, effective_priority, status, enqueued_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (ticket, problem, agent_id, agent_name, request.remote_addr or "-", git_sha, base_prio, eff, "queued", enq),
            )
        ch = ctx.conn.execute("SELECT id FROM challenges WHERE ticket=?", (ticket,)).fetchone()
        ch_id = int(ch["id"]) if ch else None
        if ch_id is not None:
            rk = str(uuid.uuid4())
            _log_ch_req(ctx, ch_id, "select", rk, "agent_in", 0, {"problemName": problem}, {})
            ctx.bus.emit("change")
            try:
                ctx.logger.write({
                    "ev": "select",
                    "action": "enqueued",
                    "challenge_id": ch_id,
                    "ticket": ticket,
                    "problem": problem,
                    "agent_name": agent_name,
                    "agent_id": agent_id,
                    "git_sha": git_sha,
                    "base_priority": base_prio,
                    "effective_priority": eff,
                    "enqueued_at": enq,
                })
            except Exception:
                pass

        ctx.bus.emit("enqueue")

        # Preemption: if incoming name matches the running one, preempt; otherwise do not preempt
        # When draining, skip preemption to preserve current running until finish
        if agent_name:
            try:
                if hasattr(ctx, 'coord') and getattr(ctx.coord, 'is_paused', None) and ctx.coord.is_paused():
                    agent_name = None  # disable preemption path
            except Exception:
                pass
            preempt_giveup = False
            preempt_grant = False
            try:
                with ctx.conn:
                    running = ctx.conn.execute(
                        "SELECT id, ticket, agent_name, started_at FROM challenges WHERE status='running' LIMIT 1"
                    ).fetchone()
                    if running is not None and running["agent_name"] and running["agent_name"] == agent_name:
                        ctx.conn.execute(
                            "UPDATE challenges SET status='giveup', finished_at=? WHERE id=? AND status='running'",
                            (utcnow_str(), running["id"]),
                        )
                        try:
                            from .sched_fair import accumulate_service
                            accumulate_service(ctx.conn, ctx.logger, ctx.s.base_priority_default, running["agent_name"], running["started_at"], utcnow_str())
                        except Exception:
                            pass
                        try:
                            _log_ch_req(ctx, int(running["id"]), "event", str(uuid.uuid4()), "preempt_giveup", 0, {}, {})
                        except Exception:
                            pass
                        preempt_giveup = True
                    none_running = (
                        ctx.conn.execute("SELECT id FROM challenges WHERE status='running' LIMIT 1").fetchone()
                        is None
                    )
                    other_waiting = ctx.conn.execute(
                        "SELECT 1 FROM challenges WHERE status='queued' AND (agent_name IS NULL OR agent_name<>?) AND ticket<>? LIMIT 1",
                        (agent_name, ticket),
                    ).fetchone() is not None
                    if none_running and not other_waiting:
                        sid = str(uuid.uuid4())
                        lease = (
                            datetime.now(timezone.utc) + timedelta(seconds=ctx.s.trial_ttl_sec)
                        ).isoformat().replace("+00:00", "Z")
                        ctx.conn.execute(
                            "UPDATE challenges SET status='running', started_at=?, lease_expire_at=?, session_id=? WHERE ticket=? AND status='queued'",
                            (utcnow_str(), lease, sid, ticket),
                        )
                        preempt_grant = True
            except Exception:
                pass
            if preempt_giveup:
                ctx.bus.emit("preempt-giveup")
                try:
                    ctx.logger.write({
                        "ev": "select",
                        "action": "preempt_giveup_by_name",
                        "by_agent_name": agent_name,
                        "by_agent_id": agent_id,
                        "ticket": ticket,
                    })
                except Exception:
                    pass
            if preempt_grant:
                ctx.bus.emit("preempt-grant")
                try:
                    ctx.logger.write({
                        "ev": "select",
                        "action": "grant_immediate",
                        "ticket": ticket,
                        "challenge_id": ch_id,
                        "lease": lease,
                    })
                except Exception:
                    pass
            else:
                try:
                    ctx.logger.write({
                        "ev": "select",
                        "action": "defer_to_scheduler_after_preempt",
                        "by_agent_name": agent_name,
                        "ticket": ticket,
                    })
                except Exception:
                    pass

        # Try grant immediately if idle (scheduler nudge)
        _grant_waiting_if_idle(ctx)

        # Blocking until granted
        started = None
        deadline = time.time() + max(5 * 60, ctx.s.trial_ttl_sec)
        while time.time() < deadline:
            row = ctx.conn.execute(
                "SELECT status, session_id, id FROM challenges WHERE ticket=?", (ticket,)
            ).fetchone()
            if row is None:
                break
            if row["status"] == "running" and row["session_id"]:
                started = row["session_id"]
                try:
                    ch_id = int(row["id"]) if row["id"] is not None else ch_id
                except Exception:
                    pass
                break
            time.sleep(0.2)

        if not started:
            try:
                ctx.logger.write({
                    "ev": "select",
                    "action": "queued_async",
                    "ticket": ticket,
                    "challenge_id": ch_id,
                })
            except Exception:
                pass
            return make_response((jsonify({"status": "queued", "ticket": ticket}), 202))

        # Forward upstream select now that we are granted
        if ch_id is not None:
            try:
                rk
            except NameError:
                rk = str(uuid.uuid4())
            _log_ch_req(ctx, ch_id, "select", rk, "to_upstream", 0, {"problemName": problem}, {})
            ctx.bus.emit("change")
        try:
            out = ctx.proxy.forward(
                "/select", started, {"problemName": problem, "id": "ignored"}, meta={"agent_id": agent_id, "git_sha": git_sha}
            )
        except UpstreamError as ue:
            with ctx.conn:
                ctx.conn.execute(
                    "UPDATE challenges SET status='error', finished_at=? WHERE ticket=?",
                    (utcnow_str(), ticket),
                )
            if ch_id is not None:
                _log_ch_req(ctx, ch_id, "select", rk, "from_upstream", int(ue.status), {"problemName": problem}, ue.payload)
                ctx.bus.emit("change")
            return jsonify({"error": "upstream_error", "detail": ue.payload}), 502

        lease = (
            datetime.now(timezone.utc) + timedelta(seconds=ctx.s.trial_ttl_sec)
        ).isoformat().replace("+00:00", "Z")
        with ctx.conn:
            ctx.conn.execute("UPDATE challenges SET lease_expire_at=? WHERE ticket=?", (lease, ticket))
        if ch_id is not None:
            _log_ch_req(ctx, ch_id, "select", rk, "from_upstream", 200, {"problemName": problem}, out)
            ctx.bus.emit("change")
            _log_ch_req(ctx, ch_id, "select", rk, "agent_out", 200, {"problemName": problem}, out)
            ctx.bus.emit("change")
            _log_ch_req(ctx, ch_id, "select", rk, "agent_out", 200, {"problemName": problem}, out)
        try:
            ctx.logger.write({
                "ev": "select",
                "action": "granted",
                "ticket": ticket,
                "challenge_id": ch_id,
                "session_id": started,
                "lease": lease,
            })
        except Exception:
            pass
        return make_response(jsonify(out))
