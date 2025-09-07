from __future__ import annotations

from flask import Blueprint, render_template, request
from ..context import AppCtx
from .helpers import flows as _flows


def create_panes_bp(ctx: AppCtx) -> Blueprint:
    bp = Blueprint("ui_panes", __name__, url_prefix="/minotaur")
    guard = ctx.ui_guard

    @bp.route("/pane_running")
    @guard.require()
    def pane_running():
        running = ctx.conn.execute(
            "SELECT * FROM challenges WHERE status='running' ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        running_flows = _flows(ctx, int(running["id"])) if running else []
        return render_template(
            "pane_running.html",
            running=running,
            running_flows=running_flows,
        )

    @bp.route("/pane_queued")
    @guard.require()
    def pane_queued():
        queued = ctx.conn.execute(
            "SELECT * FROM challenges WHERE status='queued' ORDER BY effective_priority DESC, enqueued_at ASC LIMIT 20"
        ).fetchall()
        queued_flows_map = {c["id"]: _flows(ctx, int(c["id"])) for c in queued}
        return render_template(
            "pane_queued.html",
            queued=queued,
            queued_flows=queued_flows_map,
        )

    @bp.route("/pane_recent")
    @guard.require()
    def pane_recent():
        try:
            recent_page = int(request.args.get("recent_page", "1") or "1")
            if recent_page < 1:
                recent_page = 1
        except Exception:
            recent_page = 1
        recent_filter = (request.args.get("recent_filter", "") or "").strip()
        page_size = 20
        offset = (recent_page - 1) * page_size
        if recent_filter:
            recent_total_row = ctx.conn.execute(
                "SELECT COUNT(1) AS c FROM challenges WHERE status IN ('correct','incorrect','success','finished_guess','timeout','giveup','error','interrupted','cancelled_queue','terminated_running') AND (COALESCE(agent_name,'') LIKE ? OR COALESCE(problem_id,'') LIKE ?)",
                (f"%{recent_filter}%", f"%{recent_filter}%"),
            ).fetchone()
        else:
            recent_total_row = ctx.conn.execute(
                "SELECT COUNT(1) AS c FROM challenges WHERE status IN ('correct','incorrect','success','finished_guess','timeout','giveup','error','interrupted','cancelled_queue','terminated_running')"
            ).fetchone()
        recent_total = int(recent_total_row["c"]) if recent_total_row else 0
        recent_pages = max(1, (recent_total + page_size - 1) // page_size)
        if recent_page > recent_pages:
            recent_page = recent_pages
            offset = (recent_page - 1) * page_size
        if recent_filter:
            recent = ctx.conn.execute(
                "SELECT * FROM challenges WHERE status IN ('correct','incorrect','success','finished_guess','timeout','giveup','error','interrupted','cancelled_queue','terminated_running') AND (COALESCE(agent_name,'') LIKE ? OR COALESCE(problem_id,'') LIKE ?) ORDER BY finished_at DESC LIMIT ? OFFSET ?",
                (f"%{recent_filter}%", f"%{recent_filter}%", page_size, offset),
            ).fetchall()
        else:
            recent = ctx.conn.execute(
                "SELECT * FROM challenges WHERE status IN ('correct','incorrect','success','finished_guess','timeout','giveup','error','interrupted','cancelled_queue','terminated_running') ORDER BY finished_at DESC LIMIT ? OFFSET ?",
                (page_size, offset),
            ).fetchall()
        recent_flows_map = {c["id"]: _flows(ctx, int(c["id"])) for c in recent}
        return render_template(
            "pane_recent.html",
            recent=recent,
            recent_flows=recent_flows_map,
            recent_page=recent_page,
            recent_pages=recent_pages,
            recent_total=recent_total,
            recent_filter=recent_filter,
        )

    @bp.route("/pane_admin")
    @guard.require()
    def pane_admin():
        return render_template("pane_admin.html")

    @bp.route("/pane_scheduler")
    @guard.require()
    def pane_scheduler():
        return render_template("pane_scheduler.html")

    return bp

