from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from flask import Flask, Response, jsonify, make_response, redirect, render_template, request, stream_with_context

from .auth import BasicAuthGuard, User, load_users
from .config import (
    Settings,
    load_settings_from_env,
    resolve_under_base,
    load_persisted_settings,
    save_persisted_settings,
)
from . import db as dbm
from .logging import JsonlLogger
from .proxy import UpstreamError, UpstreamProxy
from .scheduler import Coordinator
from typing import Iterator


def utcnow_str() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


app = Flask(__name__)
s: Settings = load_settings_from_env()

def _ts() -> str:
    return utcnow_str()

def _log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)
# Apply persisted YAML settings (if present)
persist = load_persisted_settings(s.settings_file)
if persist:
    try:
        if "OFFICIAL_BASE" in persist:
            s.official_base = persist.get("OFFICIAL_BASE") or None
        if "TRIAL_TTL_SEC" in persist:
            s.trial_ttl_sec = int(persist.get("TRIAL_TTL_SEC", s.trial_ttl_sec))
        if "LOG_DIR" in persist:
            s.log_dir = resolve_under_base(persist.get("LOG_DIR") or s.log_dir)
        _log(f"[boot] loaded settings from YAML {s.settings_file}: keys={list(persist.keys())}")
    except Exception:
        _log(f"[boot] failed to apply YAML settings from {s.settings_file}")
_log(f"[boot] db_path={s.db_path} log_dir={s.log_dir} official_base={s.official_base or '<unset>'}")
conn = dbm.get_conn(s.db_path)
dbm.init_schema(conn)
# On boot, mark lingering running/queued as error (move to recent)
try:
    now = utcnow_str()
    with conn:
        conn.execute("UPDATE trials SET status='error', finished_at=? WHERE status IN ('running','queued')", (now,))
    _log("[boot] reaped lingering trials: running/queued -> error")
except Exception:
    pass
logger = JsonlLogger(s.log_dir)
proxy = UpstreamProxy(s, logger)
coord = Coordinator(s, conn)

users = load_users(s.auth_file)
_log(f"[boot] auth_file={s.auth_file} users={[u.username for u in users]}")
_ui_guard = BasicAuthGuard(users)
_cv = threading.Condition()


# ===== Access logging (directional) =====
AGENT_ENDPOINTS = {"/select", "/explore", "/guess"}


def _body_snippet() -> str:
    try:
        if request.method in ("POST", "PUT", "PATCH"):
            data = request.get_data(cache=True, as_text=True)  # safe to read
            if not data:
                return ""
            s = data.strip().replace("\n", " ")
            if len(s) > 500:
                s = s[:500] + "..."
            return s
        return ""
    except Exception:
        return ""


@app.before_request
def _log_request() -> None:
    try:
        prefix = "[webui -> minotaur]"
        if request.path in AGENT_ENDPOINTS:
            aid = request.headers.get("X-Agent-ID") or "-"
            git = request.headers.get("X-Agent-Git") or "-"
            prefix = f"[agent -> minotaur] agent_id={aid} git={git}"
        if request.path == "/minotaur/agent_count":
            return  # reduce noise for frequent polling
        b = _body_snippet()
        extra = f" body={b}" if b else ""
        _log(f"{prefix} {request.method} {request.path}{extra}")
    except Exception:
        pass


@app.after_request
def _log_response(resp: Response):
    try:
        if request.path == "/minotaur/agent_count":
            return resp
        prefix = "[webui <- minotaur]"
        if request.path in AGENT_ENDPOINTS:
            prefix = "[agent <- minotaur]"
        _log(f"{prefix} {request.method} {request.path} -> {resp.status_code}")
    except Exception:
        pass
    return resp


# ===== Server-Sent Events (UI即時反映) =====
class EventBus:
    def __init__(self) -> None:
        self._cv = threading.Condition()
        self._seq = 0

    def emit(self, kind: str = "change") -> None:
        try:
            with self._cv:
                self._seq += 1
                self._cv.notify_all()
            _log(f"[event] emit kind={kind} seq={self._seq}")
        except Exception:
            pass

    def stream(self) -> Iterator[bytes]:
        last = 0
        while True:
            with self._cv:
                # 60秒でping
                self._cv.wait(timeout=60.0)
                cur = self._seq
            if cur != last:
                last = cur
                yield f"event: change\n".encode("utf-8")
                yield f"data: {cur}\n\n".encode("utf-8")
            else:
                # keep-alive
                yield b"event: ping\ndata: .\n\n"


_bus = EventBus()


@app.route("/minotaur/stream")
@_ui_guard.require()
def ui_stream():
    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
    }
    return Response(stream_with_context(_bus.stream()), headers=headers)

# Wire coordinator change notifications after EventBus is ready, then start it
coord.on_change = lambda ev: _bus.emit(ev)
coord.start()


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
        # Persist to YAML only (SQLite への保存は廃止)
        try:
            save_persisted_settings(s.settings_file, kv)
            print(f"[settings] saved to YAML {s.settings_file} keys={list(kv.keys())}", flush=True)
        except Exception:
            pass
        # Apply to live settings
        if "OFFICIAL_BASE" in kv:
            s.official_base = kv.get("OFFICIAL_BASE") or None
        if "TRIAL_TTL_SEC" in kv:
            try:
                s.trial_ttl_sec = int(kv.get("TRIAL_TTL_SEC", s.trial_ttl_sec))
            except Exception:
                pass
        if "LOG_DIR" in kv:
            log_dir_new = resolve_under_base(kv.get("LOG_DIR") or s.log_dir)
            s.log_dir = log_dir_new
            logger.log_dir = log_dir_new
        try:
            print(f"[settings] applied OFFICIAL_BASE={s.official_base or '<unset>'} TRIAL_TTL_SEC={s.trial_ttl_sec} LOG_DIR={s.log_dir}", flush=True)
        except Exception:
            pass
        _bus.emit("settings")
        return Response(status=204)
    # GET: return form partial for modal
    return render_template(
        "settings_form.html",
        official_base=s.official_base,
        trial_ttl_sec=s.trial_ttl_sec,
        log_dir=s.log_dir,
    )


@app.route("/minotaur/agent_count")
@_ui_guard.require()
def ui_agent_count():
    # 接続中のエージェント数（queued/running の重複しない agent_name 単位。agent_name が無い場合は ticket で数える）
    row = dbm.query_one(
        conn,
        "SELECT COUNT(DISTINCT COALESCE(agent_name, ticket)) AS n FROM trials WHERE status IN ('queued','running')",
    )
    n = int(row["n"]) if row else 0
    # UIの部品として自身を再生成（hx属性を保持したまま outerHTML 置換可能）
    html = (
        f"<span id=\"agent-count\" class=\"font-mono\" "
        f"hx-get=\"/minotaur/agent_count\" hx-trigger=\"load, sse:change\" hx-swap=\"outerHTML\">{n}</span>"
    )
    return make_response(html)


@app.route("/minotaur/cancel_running", methods=["POST"])
@_ui_guard.require()
def ui_cancel_running():
    now = utcnow_str()
    with conn:
        conn.execute(
            "UPDATE trials SET status='giveup', finished_at=? WHERE status='running'",
            (now,),
        )
    # nudge scheduler to grant next
    _grant_waiting_if_idle()
    # Return refreshed status partial so HTMX can replace it immediately
    _bus.emit("cancel")
    running = dbm.query_one(conn, "SELECT ticket, agent_name, problem_id, lease_expire_at FROM trials WHERE status='running' ORDER BY started_at DESC LIMIT 1")
    queued = dbm.query_all(conn, "SELECT ticket, agent_name, problem_id, effective_priority FROM trials WHERE status='queued' ORDER BY effective_priority DESC, enqueued_at ASC LIMIT 20")
    recent = dbm.query_all(conn, "SELECT ticket, status, problem_id, agent_name FROM trials WHERE status IN ('success','timeout','giveup','error') ORDER BY finished_at DESC LIMIT 20")
    return render_template("status.html", running=running, queued=queued, recent=recent)


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
    try:
        _log(f"[agent] /select agent_id={agent_id or '-'} git={git_sha or '-'} problem={problem}")
    except Exception:
        pass
    base_prio = s.base_priority_default

    ticket = str(uuid.uuid4())
    enq = utcnow_str()
    eff = base_prio
    with conn:
        conn.execute(
            "INSERT INTO trials(ticket, problem_id, params_json, agent_name, git_sha, base_priority, effective_priority, status, enqueued_at) VALUES(?,?,?,?,?,?,?,?,?)",
            (ticket, problem, json.dumps(body.get("params") or {}), agent_name, git_sha, base_prio, eff, "queued", enq),
        )
    try:
        qn = dbm.query_one(conn, "SELECT COUNT(1) AS n FROM trials WHERE status='queued'")
        _log(f"[agent] enqueued ticket={ticket} queued={int(qn['n']) if qn else 0}")
    except Exception:
        pass
    _bus.emit("enqueue")

    # Preemption: if the same agent is currently running, force-finish it and immediately grant this ticket
    if agent_name:
        preempt_giveup = False
        preempt_grant = False
        try:
            with dbm.tx(conn):
                running = dbm.query_one(conn, "SELECT id, ticket FROM trials WHERE status='running' AND agent_name=? LIMIT 1", (agent_name,))
                if running is not None:
                    # Force-terminate the running trial for this agent
                    conn.execute(
                        "UPDATE trials SET status='giveup', finished_at=? WHERE id=? AND status='running'",
                        (utcnow_str(), running["id"]),
                    )
                    preempt_giveup = True
                # If no other running exists now, grant this ticket immediately
                none_running = dbm.query_one(conn, "SELECT id FROM trials WHERE status='running' LIMIT 1") is None
                if none_running:
                    sid = str(uuid.uuid4())
                    lease = (datetime.now(timezone.utc) + timedelta(seconds=s.trial_ttl_sec)).isoformat().replace("+00:00", "Z")
                    conn.execute(
                        "UPDATE trials SET status='running', started_at=?, lease_expire_at=?, session_id=? WHERE ticket=? AND status='queued'",
                        (utcnow_str(), lease, sid, ticket),
                    )
                    preempt_grant = True
        except Exception:
            # fall back to normal path
            pass
        if preempt_giveup:
            _bus.emit("preempt-giveup")
        if preempt_grant:
            _bus.emit("preempt-grant")

    # If someone else is running but no one else is queued (except this ticket), preempt to keep flow snappy
    # Guard carefully to avoid self-preempting this very ticket.
    try:
        did_giveup = False
        did_grant = False
        with dbm.tx(conn):
            myrow = dbm.query_one(conn, "SELECT id, status FROM trials WHERE ticket=?", (ticket,))
            running = dbm.query_one(conn, "SELECT id, ticket FROM trials WHERE status='running' LIMIT 1")
            cnt_other = dbm.query_one(
                conn,
                "SELECT COUNT(1) AS n FROM trials WHERE status='queued' AND ticket<>?",
                (ticket,),
            )
            n_other = int(cnt_other["n"]) if cnt_other else 0
            if (
                running is not None
                and running["ticket"] != ticket
                and myrow is not None
                and myrow["status"] == "queued"
                and n_other == 0
            ):
                # preempt current (other) running and grant this ticket immediately
                conn.execute(
                    "UPDATE trials SET status='giveup', finished_at=? WHERE id=? AND status='running'",
                    (utcnow_str(), running["id"]),
                )
                did_giveup = True
                sid = str(uuid.uuid4())
                lease = (datetime.now(timezone.utc) + timedelta(seconds=s.trial_ttl_sec)).isoformat().replace("+00:00", "Z")
                conn.execute(
                    "UPDATE trials SET status='running', started_at=?, lease_expire_at=?, session_id=? WHERE ticket=? AND status='queued'",
                    (utcnow_str(), lease, sid, ticket),
                )
                did_grant = True
        if did_giveup:
            _bus.emit("preempt-giveup")
        if did_grant:
            _bus.emit("preempt-grant")
    except Exception:
        pass

    # Try grant immediately if idle (scheduler nudge)
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
        out = proxy.forward("/select", started, {"problemName": problem, "id": "ignored"}, meta={"agent_id": agent_id, "git_sha": git_sha})
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
    _log(f"[agent] granted ticket={ticket} sid={started} lease={lease}")
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
    agent_id = request.headers.get("X-Agent-ID") or None
    git_sha = request.headers.get("X-Agent-Git") or None
    sid = _current_running_session()
    if not sid:
        return jsonify({"error": "no active session"}), 409
    state = {"query_count": (dbm.query_one(conn, "SELECT upstream_query_count FROM trials WHERE status='running'") or {"upstream_query_count": 0})["upstream_query_count"]}
    try:
        out = proxy.forward("/explore", sid, body, state, meta={"agent_id": agent_id, "git_sha": git_sha})
    except UpstreamError as ue:
        return jsonify({"error": "upstream_error", "detail": ue.payload}), 502
    # update qc and lease
    with conn:
        conn.execute("UPDATE trials SET upstream_query_count=? WHERE status='running'", (int(out.get("queryCount", 0)),))
    _touch_lease()
    try:
        plans = body.get("plans") or []
        _log(f"[agent] /explore agent_id={agent_id or '-'} sid={sid} plans={len(plans)} qc->{int(out.get('queryCount',0))}")
    except Exception:
        pass
    _bus.emit("explore")
    return jsonify(out)


@app.route("/guess", methods=["POST"])
def guess_route():
    body = request.get_json(silent=True) or {}
    agent_id = request.headers.get("X-Agent-ID") or None
    git_sha = request.headers.get("X-Agent-Git") or None
    sid = _current_running_session()
    if not sid:
        return jsonify({"error": "no active session"}), 409
    try:
        out = proxy.forward("/guess", sid, body, meta={"agent_id": agent_id, "git_sha": git_sha})
    except UpstreamError as ue:
        return jsonify({"error": "upstream_error", "detail": ue.payload}), 502
    if bool(out.get("correct")):
        with conn:
            conn.execute(
                "UPDATE trials SET status='success', finished_at=? WHERE status='running'",
                (utcnow_str(),),
            )
    _touch_lease()
    _log(f"[agent] /guess agent_id={agent_id or '-'} sid={sid} correct={bool(out.get('correct'))}")
    _bus.emit("guess")
    return jsonify(out)


def main() -> None:
    app.run(host="0.0.0.0", port=s.port, debug=False)


if __name__ == "__main__":
    main()
