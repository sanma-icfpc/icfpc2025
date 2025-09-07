from __future__ import annotations

from typing import Any, Dict
from flask import jsonify, render_template, request

from .context import AppCtx
from .ui.helpers import flows as _flows


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
                "SELECT COUNT(1) AS c FROM challenges WHERE status IN (" \
                "'correct','incorrect','success','finished_guess','timeout','giveup','error','interrupted','cancelled_queue','terminated_running') " \
                "AND (COALESCE(agent_name,'') LIKE ? OR COALESCE(problem_id,'') LIKE ?)",
                (f"%{recent_filter}%", f"%{recent_filter}%"),
            ).fetchone()
        else:
            recent_total_row = ctx.conn.execute(
                "SELECT COUNT(1) AS c FROM challenges WHERE status IN (" \
                "'correct','incorrect','success','finished_guess','timeout','giveup','error','interrupted','cancelled_queue','terminated_running')"
            ).fetchone()
        recent_total = int(recent_total_row["c"]) if recent_total_row else 0
        recent_pages = max(1, (recent_total + page_size - 1) // page_size)
        if recent_page > recent_pages:
            recent_page = recent_pages
            offset = (recent_page - 1) * page_size

        if recent_filter:
            recent = ctx.conn.execute(
                "SELECT * FROM challenges WHERE status IN (" \
                "'correct','incorrect','success','finished_guess','timeout','giveup','error','interrupted','cancelled_queue','terminated_running') " \
                "AND (COALESCE(agent_name,'') LIKE ? OR COALESCE(problem_id,'') LIKE ?) " \
                "ORDER BY finished_at DESC LIMIT ? OFFSET ?",
                (f"%{recent_filter}%", f"%{recent_filter}%", page_size, offset),
            ).fetchall()
        else:
            recent = ctx.conn.execute(
                "SELECT * FROM challenges WHERE status IN (" \
                "'correct','incorrect','success','finished_guess','timeout','giveup','error','interrupted','cancelled_queue','terminated_running') " \
                "ORDER BY finished_at DESC LIMIT ? OFFSET ?",
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

    @app.route("/minotaur/sched_info")
    @guard.require()
    def ui_sched_info():
        import time as _t
        try:
            running = ctx.conn.execute(
                "SELECT id, agent_name, started_at FROM challenges WHERE status='running' ORDER BY started_at DESC LIMIT 1"
            ).fetchone()
            queued_total = ctx.conn.execute(
                "SELECT COUNT(1) AS c FROM challenges WHERE status='queued'"
            ).fetchone()
            qt = int(queued_total["c"]) if queued_total else 0
            pin = ctx.conn.execute(
                "SELECT name, pinned FROM agent_priorities WHERE pinned IN (1,2) LIMIT 1"
            ).fetchone()
            pin_info = None
            if pin is not None:
                pin_info = {
                    "name": pin["name"],
                    "mode": "persistent" if int(pin["pinned"]) == 1 else "one_shot",
                }
            # Coordinator hold window insight (best-effort)
            hold_until = None
            hold_remaining = None
            try:
                hu = getattr(ctx.coord, "_pin_hold_until", 0.0)
                if isinstance(hu, (int, float)) and hu > 0:
                    hold_until = float(hu)
                    hold_remaining = max(0.0, hold_until - _t.time())
            except Exception:
                pass
            return jsonify({
                "running": dict(running) if running else None,
                "queued_total": qt,
                "pinned": pin_info,
                "pin_hold_until": hold_until,
                "pin_hold_remaining": hold_remaining,
            })
        except Exception:
            return jsonify({"error": "sched_info_failed"}), 500

