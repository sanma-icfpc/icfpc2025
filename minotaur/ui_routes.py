from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

from flask import Response, jsonify, make_response, render_template, request, stream_with_context, send_file

from .context import AppCtx
from .sched_fair import accumulate_service
from . import db as dbm
import threading, os, time as _time
import platform as _platform
import sys as _sys


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
        # Pagination for recent
        try:
            recent_page = int(request.args.get("recent_page", "1") or "1")
            if recent_page < 1:
                recent_page = 1
        except Exception:
            recent_page = 1
        recent_filter = (request.args.get("recent_filter", "") or "").strip()
        page_size = 20
        offset = (recent_page - 1) * page_size
        running = ctx.conn.execute(
            "SELECT * FROM challenges WHERE status='running' ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        queued = ctx.conn.execute(
            "SELECT * FROM challenges WHERE status='queued' ORDER BY effective_priority DESC, enqueued_at ASC LIMIT 20"
        ).fetchall()
        # Count total recent for pagination
        if recent_filter:
            recent_total_row = ctx.conn.execute(
                "SELECT COUNT(1) AS c FROM challenges WHERE status IN ('success','timeout','giveup','error','interrupted','cancelled_queue','terminated_running') AND (COALESCE(agent_name,'') LIKE ? OR COALESCE(problem_id,'') LIKE ?)",
                (f"%{recent_filter}%", f"%{recent_filter}%"),
            ).fetchone()
        else:
            recent_total_row = ctx.conn.execute(
                "SELECT COUNT(1) AS c FROM challenges WHERE status IN ('success','timeout','giveup','error','interrupted','cancelled_queue','terminated_running')"
            ).fetchone()
        recent_total = int(recent_total_row["c"]) if recent_total_row else 0
        recent_pages = max(1, (recent_total + page_size - 1) // page_size)
        if recent_page > recent_pages:
            recent_page = recent_pages
            offset = (recent_page - 1) * page_size
        if recent_filter:
            recent = ctx.conn.execute(
                "SELECT * FROM challenges WHERE status IN ('success','timeout','giveup','error','interrupted','cancelled_queue','terminated_running') AND (COALESCE(agent_name,'') LIKE ? OR COALESCE(problem_id,'') LIKE ?) ORDER BY finished_at DESC LIMIT ? OFFSET ?",
                (f"%{recent_filter}%", f"%{recent_filter}%", page_size, offset),
            ).fetchall()
        else:
            recent = ctx.conn.execute(
                "SELECT * FROM challenges WHERE status IN ('success','timeout','giveup','error','interrupted','cancelled_queue','terminated_running') ORDER BY finished_at DESC LIMIT ? OFFSET ?",
                (page_size, offset),
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
            recent_page=recent_page,
            recent_pages=recent_pages,
            recent_total=recent_total,
            recent_filter=recent_filter,
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

    @app.route("/minotaur/db_info")
    @guard.require()
    def ui_db_info():
        try:
            dbp = ctx.s.db_path
            if dbp.startswith("file:") and ("mode=memory" in dbp):
                return jsonify({"in_memory": True}), 200
            if not os.path.isfile(dbp):
                return jsonify({"error": "not_found"}), 404
            size_bytes = os.path.getsize(dbp)
            size_mb = round(size_bytes / (1024 * 1024), 1)
            try:
                mtime = os.path.getmtime(dbp)
            except Exception:
                mtime = None
            return jsonify({
                "path": dbp,
                "sizeBytes": int(size_bytes),
                "sizeMB": float(size_mb),
                "mtime": mtime,
            })
        except Exception:
            return jsonify({"error": "stat_failed"}), 500

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
                "UPDATE challenges SET status='terminated_running', finished_at=? WHERE status='running'",
                (now,),
            )
        try:
            if cur is not None:
                accumulate_service(ctx.conn, ctx.logger, ctx.s.base_priority_default, cur["agent_name"], cur["started_at"], now)
        except Exception:
            pass
        # Notify
        try:
            ctx.bus.emit("cancel")
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
            "SELECT * FROM challenges WHERE status IN ('success','timeout','giveup','error','interrupted','cancelled_queue','terminated_running') ORDER BY finished_at DESC LIMIT 20"
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

    @app.route("/minotaur/cancel_queued", methods=["POST"])
    @guard.require()
    def ui_cancel_queued():
        from .app import utcnow_str  # reuse utility
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
                    "UPDATE challenges SET status='cancelled_queue', finished_at=? WHERE id=? AND status='queued'",
                    (now, int(cur["id"])),
                )
                cancelled = True
        # Emit and re-render status
        try:
            if cancelled:
                ctx.bus.emit("cancel_queued")
                if ctx.coord and ctx.coord.on_change:
                    ctx.coord.on_change("cancel_queued")
        except Exception:
            pass
        running = ctx.conn.execute(
            "SELECT * FROM challenges WHERE status='running' ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        queued = ctx.conn.execute(
            "SELECT * FROM challenges WHERE status='queued' ORDER BY effective_priority DESC, enqueued_at ASC LIMIT 20"
        ).fetchall()
        recent = ctx.conn.execute(
            "SELECT * FROM challenges WHERE status IN ('success','timeout','giveup','error','interrupted','cancelled_queue','terminated_running') ORDER BY finished_at DESC LIMIT 20"
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

    @app.route("/minotaur/threads_os")
    @guard.require()
    def ui_threads_os():
        pid = os.getpid()
        plat = (_platform.system() or "").lower()
        # Map Python threads by native id if available (3.8+)
        py_threads = {}
        try:
            import threading as _th
            for t in _th.enumerate():
                try:
                    nid = getattr(t, "native_id", None)
                except Exception:
                    nid = None
                py_threads[int(nid)] = {
                    "py_name": t.name,
                    "daemon": bool(getattr(t, "daemon", False)),
                } if nid is not None else {}
        except Exception:
            pass

        out = {"pid": pid, "platform": _platform.system(), "threads": [], "summary": {}}

        def _merge_py(meta: dict, tid: int) -> dict:
            pt = py_threads.get(int(tid)) or {}
            if pt:
                meta = dict(meta)
                meta.update(pt)
            return meta

        if plat == "linux":
            base = f"/proc/{pid}/task"
            try:
                tids = []
                for name in os.listdir(base):
                    try:
                        tids.append(int(name))
                    except Exception:
                        continue
                tids.sort()
                try:
                    clk_tck = os.sysconf(os.sysconf_names.get('SC_CLK_TCK', 'SC_CLK_TCK'))
                except Exception:
                    clk_tck = 100
                for tid in tids:
                    tdir = f"{base}/{tid}"
                    st = {"id": tid}
                    # status
                    try:
                        with open(f"{tdir}/status", "r", encoding="utf-8", errors="ignore") as f:
                            for line in f:
                                if line.startswith("Name:"):
                                    st["name"] = line.split(":",1)[1].strip()
                                elif line.startswith("State:"):
                                    # e.g., "State:\tS (sleeping)"
                                    val = line.split(":",1)[1].strip()
                                    st["state"] = val
                                elif line.startswith("voluntary_ctxt_switches:"):
                                    try:
                                        st["vcsw"] = int(line.rsplit(None,1)[-1])
                                    except Exception:
                                        pass
                                elif line.startswith("nonvoluntary_ctxt_switches:"):
                                    try:
                                        st["nvcsw"] = int(line.rsplit(None,1)[-1])
                                    except Exception:
                                        pass
                    except Exception:
                        pass
                    # stat (CPU times)
                    try:
                        with open(f"{tdir}/stat", "r", encoding="utf-8", errors="ignore") as f:
                            sline = f.readline().strip()
                        # parse comm in parentheses
                        rpar = sline.rfind(")")
                        part = sline[rpar+2:].split()
                        # fields: 3=state, 14=utime, 15=stime -> indices 11 and 12 after slicing
                        ut = int(part[11]) if len(part) > 11 else 0
                        st_ = int(part[12]) if len(part) > 12 else 0
                        st["cpu_time_sec"] = (ut + st_) / float(clk_tck or 100)
                    except Exception:
                        pass
                    # wchan
                    try:
                        with open(f"{tdir}/wchan", "r", encoding="utf-8", errors="ignore") as f:
                            st["wchan"] = (f.read().strip() or None)
                    except Exception:
                        pass
                    out["threads"].append(_merge_py(st, tid))
                # Summary by state initial letter (e.g., R,S,D,...)
                summ = {}
                for t in out["threads"]:
                    stv = t.get("state") or "?"
                    key = stv.split()[0] if stv else "?"
                    summ[key] = int(summ.get(key, 0)) + 1
                out["summary"] = summ
            except Exception as e:
                out["error"] = f"linux_proc_read_failed: {e}"
        else:
            # Try psutil if available for other platforms
            try:
                import psutil  # type: ignore
                p = psutil.Process(pid)
                thr = p.threads()
                for th in thr:
                    meta = {
                        "id": int(getattr(th, 'id', 0)),
                        "cpu_time_sec": float(getattr(th, 'user_time', 0.0)) + float(getattr(th, 'system_time', 0.0)),
                    }
                    out["threads"].append(_merge_py(meta, meta["id"]))
                out["summary"] = {"count": len(out["threads"]) }
            except Exception:
                # Fallback: only Python threads
                out["threads"] = []
                try:
                    import threading as _th
                    for t in _th.enumerate():
                        try:
                            nid = getattr(t, "native_id", None)
                        except Exception:
                            nid = None
                        out["threads"].append({
                            "id": int(nid) if nid is not None else None,
                            "py_name": t.name,
                            "daemon": bool(getattr(t, "daemon", False)),
                        })
                    out["summary"] = {"count": len(out["threads"]) }
                except Exception:
                    pass
        return jsonify(out)
