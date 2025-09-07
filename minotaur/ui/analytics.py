from __future__ import annotations

from flask import Blueprint, render_template
from ..context import AppCtx


def _collect_analytics(ctx: AppCtx):
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
        agents.append("-")
    q = (
        "SELECT agent_name, problem_id, "
        "AVG(CASE WHEN status IN ('correct','success') THEN score_query_count END) AS mean_qc, "
        "MIN(CASE WHEN status IN ('correct','success') THEN score_query_count END) AS min_qc, "
        "SUM(CASE WHEN status IN ('correct','success') THEN 1 ELSE 0 END) AS n_correct, "
        "SUM(CASE WHEN status IN ('incorrect','finished_guess') THEN 1 ELSE 0 END) AS n_incorrect "
        "FROM challenges GROUP BY agent_name, problem_id"
    )
    metrics = {}
    for r in ctx.conn.execute(q).fetchall():
        an = r["agent_name"] if r["agent_name"] is not None else "-"
        pid = r["problem_id"]
        metrics[(an, pid)] = {
            "mean_qc": r["mean_qc"],
            "min_qc": r["min_qc"],
            "n_correct": int(r["n_correct"] or 0),
            "n_incorrect": int(r["n_incorrect"] or 0),
        }
    return probs, agents, metrics


def create_analytics_bp(ctx: AppCtx) -> Blueprint:
    bp = Blueprint("ui_analytics", __name__, url_prefix="/minotaur")
    guard = ctx.ui_guard

    @bp.route("/analytics")
    @guard.require()
    def analytics():
        probs, agents, metrics = _collect_analytics(ctx)
        return render_template("analytics.html", problems=probs, agents=agents, metrics=metrics)

    @bp.route("/pane_analytics")
    @guard.require()
    def pane_analytics():
        probs, agents, metrics = _collect_analytics(ctx)
        return render_template("pane_analytics.html", problems=probs, agents=agents, metrics=metrics)

    return bp

