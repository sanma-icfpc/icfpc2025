from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

from flask import Response, jsonify, make_response, render_template, request, stream_with_context, send_file

from .context import AppCtx
from .sched_fair import accumulate_service
import threading, os, time as _time


def _fmt_ts(ts: str) -> str:
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        disp = dt.strftime("%Y/%m/%d %H:%M:%S")
        delta = datetime.now(timezone.utc) - dt.astimezone(timezone.utc)
        sec = int(delta.total_seconds())
        if sec < 60:
            rel = f"{sec}秒前"
        elif sec < 3600:
            rel = f"{sec//60}分前"
        else:
            rel = f"{sec//3600}時間前"
        return f"{disp} ({rel})"
    except Exception:
        return ts


def _flows(ctx: AppCtx, ch_id: int):
    rows = ctx.conn.execute(
        "SELECT api, req_key, phase, status_code, ts, req_body, res_body FROM challenge_requests WHERE challenge_id=? ORDER BY id ASC",
        (ch_id,),
    ).fetchall()
    d: Dict[str, Dict[str, Any]] = {}
    import json as _json
    for r in rows:
        key = r["req_key"] or f"{r['api']}-{r['ts']}"
        item = d.get(key)
        if not item:
            item = {
                "api": r["api"],
                "phases": set(),
                "ts": r["ts"],
                "codes": {},
                "req_key": key,
                "req_pretty": None,
                "res_pretty": None,
                "summary": "",
            }
            d[key] = item
        ph = r["phase"]
        item["phases"].add(ph)
        try:
            sc = int(r["status_code"]) if r["status_code"] is not None else 0
        except Exception:
            sc = 0
        item["codes"][ph] = sc
        if ph in ("agent_in", "to_upstream") and item["req_pretty"] is None:
            try:
                obj = _json.loads(r["req_body"]) if r["req_body"] else None
                item["req_pretty"] = _json.dumps(obj, ensure_ascii=False, indent=2) if obj is not None else (r["req_body"] or "")
            except Exception:
                item["req_pretty"] = r["req_body"] or ""
        if ph in ("from_upstream", "agent_out"):
            try:
                obj = _json.loads(r["res_body"]) if r["res_body"] else None
                item["res_pretty"] = _json.dumps(obj, ensure_ascii=False, indent=2) if obj is not None else (r["res_body"] or "")
            except Exception:
                item["res_pretty"] = r["res_body"] or ""
        try:
            if r["api"] == "select" and ph == "agent_in" and not item["summary"]:
                obj = _json.loads(r["req_body"]) if r["req_body"] else {}
                pn = obj.get("problemName")
                item["summary"] = f"problem={pn}" if pn else ""
            elif r["api"] == "explore" and ph == "from_upstream":
                robj = _json.loads(r["res_body"]) if r["res_body"] else {}
                results = robj.get("results") or []
                q = robj.get("queryCount")
                lens = [len(x) if isinstance(x, list) else 0 for x in results]
                item["summary"] = f"plans={len(lens)} results={lens} qc={q}"
            elif r["api"] == "guess" and ph == "from_upstream":
                robj = _json.loads(r["res_body"]) if r["res_body"] else {}
                ok = robj.get("correct")
                req = _json.loads(r["req_body"]) if r["req_body"] else {}
                m = req.get("map") or {}
                rooms = m.get("rooms") or []
                conns = m.get("connections") or []
                item["summary"] = f"rooms={len(rooms)} conns={len(conns)} correct={ok}"
        except Exception:
            pass
    out = list(d.values())
    out.sort(key=lambda x: x["ts"])  # oldest first
    return out


def register_ui_routes(app, ctx: AppCtx) -> None:
    guard = ctx.ui_guard

    @app.route("/")
    @guard.require()
    def index():
        return render_template(
            "index.html",
            official_base=ctx.s.official_base,
            trial_ttl_sec=ctx.s.trial_ttl_sec,
            log_dir=ctx.s.log_dir,
        )

    @app.route("/minotaur/status")
    @guard.require()
    def ui_status():
        running = ctx.conn.execute(
            "SELECT * FROM challenges WHERE status='running' ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        queued = ctx.conn.execute(
            "SELECT * FROM challenges WHERE status='queued' ORDER BY effective_priority DESC, enqueued_at ASC LIMIT 20"
        ).fetchall()
        recent = ctx.conn.execute(
            "SELECT * FROM challenges WHERE status IN ('success','timeout','giveup','error','interrupted') ORDER BY finished_at DESC LIMIT 20"
        ).fetchall()
        running_flows = _flows(ctx, int(running["id"])) if running else []
        queued_flows_map = {c["id"]: _flows(ctx, int(c["id"])) for c in queued}
        recent_flows_map = {c["id"]: _flows(ctx, int(c["id"])) for c in recent}
        return render_template(
            "status.html",
            running=running,
            running_flows=running_flows,
            queued=queued,
            queued_flows=queued_flows_map,
            recent=recent,
            recent_flows=recent_flows_map,
        )

    @app.route("/minotaur/stream")
    @guard.require()
    def ui_stream():
        headers = {"Content-Type": "text/event-stream", "Cache-Control": "no-cache"}
        return Response(stream_with_context(ctx.bus.stream()), headers=headers)

    @app.route("/minotaur/settings", methods=["GET", "POST"])
    @guard.require()
    def ui_settings():
        from .config import resolve_under_base, save_persisted_settings

        if request.method == "POST":
            try:
                ob = request.form.get("OFFICIAL_BASE")
                if ob is not None:
                    ctx.s.official_base = ob or None
                ttl = request.form.get("TRIAL_TTL_SEC")
                if ttl is not None:
                    ctx.s.trial_ttl_sec = int(ttl)
                lg = request.form.get("LOG_DIR")
                if lg is not None:
                    ctx.s.log_dir = resolve_under_base(lg)
                save_persisted_settings(
                    ctx.s.settings_file,
                    {
                        "OFFICIAL_BASE": ctx.s.official_base,
                        "TRIAL_TTL_SEC": ctx.s.trial_ttl_sec,
                        "LOG_DIR": ctx.s.log_dir,
                    },
                )
            except Exception:
                pass
            if ctx.coord and ctx.coord.on_change:
                try:
                    ctx.coord.on_change("settings")
                except Exception:
                    pass
            return Response(status=204)
        return render_template(
            "settings_form.html",
            official_base=ctx.s.official_base,
            trial_ttl_sec=ctx.s.trial_ttl_sec,
            log_dir=ctx.s.log_dir,
        )

    @app.route("/minotaur/download_db")
    @guard.require()
    def ui_download_db():
        try:
            dbp = ctx.s.db_path
            # In-memory DB URIs cannot be downloaded
            if dbp.startswith("file:") and ("mode=memory" in dbp):
                return jsonify({"error": "in_memory_db"}), 400
            if not os.path.isfile(dbp):
                return jsonify({"error": "not_found"}), 404
            return send_file(dbp, as_attachment=True, download_name="coordinator.sqlite", mimetype="application/x-sqlite3")
        except Exception:
            return jsonify({"error": "download_failed"}), 500

    @app.route("/minotaur/analytics")
    @guard.require()
    def ui_analytics():
        # Collect axes
        probs = [r["problem_id"] for r in ctx.conn.execute(
            "SELECT DISTINCT problem_id FROM challenges WHERE problem_id IS NOT NULL ORDER BY problem_id"
        ).fetchall()]
        agents_raw = ctx.conn.execute(
            "SELECT DISTINCT agent_name FROM challenges ORDER BY agent_name"
        ).fetchall()
        agents = []
        has_null = False
        for r in agents_raw:
            nm = r["agent_name"]
            if nm is None:
                has_null = True
            else:
                agents.append(nm)
        if has_null:
            agents.append("-")  # placeholder for anonymous
        # Build metrics map
        q = (
            "SELECT agent_name, problem_id, "
            "AVG(CASE WHEN status='success' THEN score_query_count END) AS mean_qc, "
            "MIN(CASE WHEN status='success' THEN score_query_count END) AS min_qc, "
            "SUM(CASE WHEN status='success' THEN 1 ELSE 0 END) AS n_success, "
            "SUM(CASE WHEN status!='success' THEN 1 ELSE 0 END) AS n_non_success "
            "FROM challenges GROUP BY agent_name, problem_id"
        )
        metrics = {}
        for r in ctx.conn.execute(q).fetchall():
            an = r["agent_name"] if r["agent_name"] is not None else "-"
            pid = r["problem_id"]
            metrics[(an, pid)] = {
                "mean_qc": r["mean_qc"],
                "min_qc": r["min_qc"],
                "n_success": int(r["n_success"] or 0),
                "n_non_success": int(r["n_non_success"] or 0),
            }
        return render_template("analytics.html", problems=probs, agents=agents, metrics=metrics)

    @app.route("/minotaur/agent_count")
    @guard.require()
    def ui_agent_count():
        row = ctx.conn.execute(
            "SELECT COUNT(DISTINCT COALESCE(agent_name, ticket)) AS n FROM challenges WHERE status IN ('queued','running')"
        ).fetchone()
        n = int(row["n"]) if row else 0
        return jsonify({"n": n})

    @app.route("/minotaur/cancel_running", methods=["POST"])
    @guard.require()
    def ui_cancel_running():
        from .app import utcnow_str  # reuse utility
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
                "UPDATE challenges SET status='giveup', finished_at=? WHERE status='running'",
                (now,),
            )
        try:
            if cur is not None:
                accumulate_service(ctx.conn, ctx.logger, ctx.s.base_priority_default, cur["agent_name"], cur["started_at"], now)
        except Exception:
            pass
        if ctx.coord and ctx.coord.on_change:
            try:
                ctx.coord.on_change("cancel")
            except Exception:
                pass
        # Return refreshed status via normal polling/SSE
        running = ctx.conn.execute(
            "SELECT * FROM challenges WHERE status='running' ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        queued = ctx.conn.execute(
            "SELECT * FROM challenges WHERE status='queued' ORDER BY effective_priority DESC, enqueued_at ASC LIMIT 20"
        ).fetchall()
        recent = ctx.conn.execute(
            "SELECT * FROM challenges WHERE status IN ('success','timeout','giveup','error','interrupted') ORDER BY finished_at DESC LIMIT 20"
        ).fetchall()
        running_flows = _flows(ctx, int(running["id"])) if running else []
        queued_flows_map = {c["id"]: _flows(ctx, int(c["id"])) for c in queued}
        recent_flows_map = {c["id"]: _flows(ctx, int(c["id"])) for c in recent}
        return render_template(
            "status.html",
            running=running,
            running_flows=running_flows,
            queued=queued,
            queued_flows=queued_flows_map,
            recent=recent,
            recent_flows=recent_flows_map,
        )

    @app.route("/minotaur/shutdown", methods=["POST"])
    @guard.require()
    def ui_shutdown():
        def _killer():
            try:
                _time.sleep(0.2)
            except Exception:
                pass
            os._exit(0)
        threading.Thread(target=_killer, daemon=True).start()
        return jsonify({"ok": True})
