from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from flask import Flask, Response, jsonify, make_response, redirect, render_template, request

from .auth import BasicAuthGuard, User, load_users
from .config import Settings, load_settings_from_env, resolve_under_base
from . import db as dbm
from .logging import JsonlLogger
from .proxy import UpstreamError, UpstreamProxy
from .scheduler import Coordinator


def utcnow_str() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


app = Flask(__name__)
s: Settings = load_settings_from_env()
print(f"[boot] db_path={s.db_path} log_dir={s.log_dir} official_base={'<mock>' if not s.official_base else s.official_base}", flush=True)
conn = dbm.get_conn(s.db_path)
dbm.init_schema(conn)
logger = JsonlLogger(s.log_dir)
proxy = UpstreamProxy(s, logger)
coord = Coordinator(s, conn)
coord.start()

users = load_users(s.auth_file)
print(f"[boot] auth_file={s.auth_file} users={[u.username for u in users]}", flush=True)
_ui_guard = BasicAuthGuard(users)
_cv = threading.Condition()


def _compute_effective_priority(base: int, enqueued_at: str) -> int:
    # ageing: +1 per configured minutes, capped
    try:
        enq = datetime.fromisoformat(enqueued_at.replace("Z", "+00:00"))
        mins = int((datetime.now(timezone.utc) - enq).total_seconds() // 60)
        ageing = min(s.ageing_cap, max(0, mins // s.ageing_every_min))
    except Exception:
        ageing = 0
    eff = base + ageing
    return eff


def _recalc_priorities() -> None:
    rows = dbm.query_all(conn, "SELECT id, base_priority, enqueued_at FROM trials WHERE status='queued'")
    with conn:
        for r in rows:
            eff = _compute_effective_priority(int(r["base_priority"]), r["enqueued_at"])
            conn.execute("UPDATE trials SET effective_priority=? WHERE id=?", (eff, r["id"]))


def _grant_waiting_if_idle() -> None:
    # nudge scheduler and wait a moment
    _recalc_priorities()
    time.sleep(0.05)


# ========== UI ==========
@app.route("/")
@_ui_guard.require()
def index():
    return render_template(
        "index.html",
        official_base=s.official_base,
        mock=s.is_mock,
        trial_ttl_sec=s.trial_ttl_sec,
        log_dir=s.log_dir,
    )


@app.route("/minotaur/status")
@_ui_guard.require()
def ui_status():
    running = dbm.query_one(conn, "SELECT ticket, agent_name, problem_id, lease_expire_at FROM trials WHERE status='running' ORDER BY started_at DESC LIMIT 1")
    queued = dbm.query_all(conn, "SELECT ticket, agent_name, problem_id, effective_priority FROM trials WHERE status='queued' ORDER BY effective_priority DESC, enqueued_at ASC LIMIT 20")
    recent = dbm.query_all(conn, "SELECT ticket, status, problem_id, agent_name FROM trials WHERE status IN ('success','timeout','giveup','error') ORDER BY finished_at DESC LIMIT 20")
    return render_template("status.html", running=running, queued=queued, recent=recent)


@app.route("/minotaur/settings", methods=["GET", "POST"])
@_ui_guard.require()
def ui_settings():
    if request.method == "POST":
        # Minimal set of keys via form fields
        kv = request.form.to_dict()
        now = utcnow_str()
        with conn:
            for k, v in kv.items():
                conn.execute(
                    "INSERT INTO settings(key, value, updated_at) VALUES(?,?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                    (k, v, now),
                )
        # Apply to live settings
        if "OFFICIAL_BASE" in kv:
            s.official_base = kv.get("OFFICIAL_BASE") or None
        if "MOCK" in kv:
            s.mock = (kv.get("MOCK", "").strip().lower() in ("1", "true", "yes", "on"))
        if "TRIAL_TTL_SEC" in kv:
            try:
                s.trial_ttl_sec = int(kv.get("TRIAL_TTL_SEC", s.trial_ttl_sec))
            except Exception:
                pass
        if "LOG_DIR" in kv:
            log_dir_new = resolve_under_base(kv.get("LOG_DIR") or s.log_dir)
            s.log_dir = log_dir_new
            logger.log_dir = log_dir_new
        return Response(status=204)
    # GET not used in UI directly (served by index)
    return Response(status=405)


@app.route("/minotaur/healthz")
def healthz():
    try:
        dbm.query_one(conn, "SELECT 1")
        return jsonify({"status": "ok"})
    except Exception:
        return jsonify({"status": "ng"}), 500


# ========== Transparent API ==========
@app.route("/select", methods=["POST"])
def select_route():
    body = request.get_json(silent=True) or {}
    problem = body.get("problemName")
    if not isinstance(problem, str) or not problem:
        return jsonify({"error": "problemName required"}), 400

    agent_id = request.headers.get("X-Agent-ID") or None
    agent_name = agent_id  # store as agent_name for now
    git_sha = request.headers.get("X-Agent-Git") or None
    base_prio = s.base_priority_default

    ticket = str(uuid.uuid4())
    enq = utcnow_str()
    eff = base_prio
    with conn:
        conn.execute(
            "INSERT INTO trials(ticket, problem_id, params_json, agent_name, git_sha, base_priority, effective_priority, status, enqueued_at) VALUES(?,?,?,?,?,?,?,?,?)",
            (ticket, problem, json.dumps(body.get("params") or {}), agent_name, git_sha, base_prio, eff, "queued", enq),
        )

    # Try grant immediately if idle
    _grant_waiting_if_idle()

    # Blocking until granted
    started = None
    deadline = time.time() + max(5 * 60, s.trial_ttl_sec)  # hard cap wait
    while time.time() < deadline:
        row = dbm.query_one(conn, "SELECT status, session_id FROM trials WHERE ticket=?", (ticket,))
        if row is None:
            break
        if row["status"] == "running" and row["session_id"]:
            started = row["session_id"]
            break
        time.sleep(0.2)

    if not started:
        # Fallback to async ack (no Retry-After header)
        return make_response((jsonify({"status": "queued", "ticket": ticket}), 202))

    # Forward upstream select now that we are granted
    try:
        out = proxy.forward("/select", started, {"problemName": problem, "id": "ignored"})
    except UpstreamError as ue:
        with conn:
            conn.execute(
                "UPDATE trials SET status='error', finished_at=? WHERE ticket=?",
                (utcnow_str(), ticket),
            )
        return jsonify({"error": "upstream_error", "detail": ue.payload}), 502

    # success: set lease and return upstream response
    lease = (datetime.now(timezone.utc) + timedelta(seconds=s.trial_ttl_sec)).isoformat().replace("+00:00", "Z")
    with conn:
        conn.execute("UPDATE trials SET lease_expire_at=? WHERE ticket=?", (lease, ticket))
    return make_response(jsonify(out))


def _current_running_session() -> Optional[str]:
    row = dbm.query_one(conn, "SELECT session_id FROM trials WHERE status='running' LIMIT 1")
    return row["session_id"] if row and row["session_id"] else None


def _touch_lease() -> None:
    lease = (datetime.now(timezone.utc) + timedelta(seconds=s.trial_ttl_sec)).isoformat().replace("+00:00", "Z")
    with conn:
        conn.execute("UPDATE trials SET lease_expire_at=? WHERE status='running'", (lease,))


@app.route("/explore", methods=["POST"])
def explore_route():
    body = request.get_json(silent=True) or {}
    sid = _current_running_session()
    if not sid:
        return jsonify({"error": "no active session"}), 409
    state = {"query_count": (dbm.query_one(conn, "SELECT upstream_query_count FROM trials WHERE status='running'") or {"upstream_query_count": 0})["upstream_query_count"]}
    try:
        out = proxy.forward("/explore", sid, body, state)
    except UpstreamError as ue:
        return jsonify({"error": "upstream_error", "detail": ue.payload}), 502
    # update qc and lease
    with conn:
        conn.execute("UPDATE trials SET upstream_query_count=? WHERE status='running'", (int(out.get("queryCount", 0)),))
    _touch_lease()
    return jsonify(out)


@app.route("/guess", methods=["POST"])
def guess_route():
    body = request.get_json(silent=True) or {}
    sid = _current_running_session()
    if not sid:
        return jsonify({"error": "no active session"}), 409
    try:
        out = proxy.forward("/guess", sid, body)
    except UpstreamError as ue:
        return jsonify({"error": "upstream_error", "detail": ue.payload}), 502
    if bool(out.get("correct")):
        with conn:
            conn.execute(
                "UPDATE trials SET status='success', finished_at=? WHERE status='running'",
                (utcnow_str(),),
            )
    _touch_lease()
    return jsonify(out)


def main() -> None:
    app.run(host="0.0.0.0", port=s.port, debug=False)


if __name__ == "__main__":
    main()
