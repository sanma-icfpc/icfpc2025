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
        conn.execute("UPDATE challenges SET status='error', finished_at=? WHERE status IN ('running','queued')", (now,))
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
    rows = dbm.query_all(conn, "SELECT id, base_priority, enqueued_at FROM challenges WHERE status='queued'")
    with conn:
        for r in rows:
            eff = _compute_effective_priority(int(r["base_priority"]), r["enqueued_at"])
            conn.execute("UPDATE challenges SET effective_priority=? WHERE id=?", (eff, r["id"]))


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
    running = dbm.query_one(conn, "SELECT * FROM challenges WHERE status='running' ORDER BY started_at DESC LIMIT 1")
    queued = dbm.query_all(conn, "SELECT * FROM challenges WHERE status='queued' ORDER BY effective_priority DESC, enqueued_at ASC LIMIT 20")
    recent = dbm.query_all(conn, "SELECT * FROM challenges WHERE status IN ('success','timeout','giveup','error') ORDER BY finished_at DESC LIMIT 20")
    # Build per-request flows grouped by req_key
    def fmt_ts(ts: str) -> str:
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

    def flows(ch_id: int):
        rows = dbm.query_all(conn, "SELECT api, req_key, phase, status_code, ts, req_body, res_body FROM challenge_requests WHERE challenge_id=? ORDER BY id ASC", (ch_id,))
        d = {}
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
            item["phases"].add(ph)  # phases: agent_in,to_upstream,from_upstream,agent_out
            try:
                sc = int(r["status_code"]) if r["status_code"] is not None else 0
            except Exception:
                sc = 0
            item["codes"][ph] = sc
            # Prefer earliest req and latest res; pretty-print JSON if possible
            import json as _json
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
            # Build concise summary per API
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
        # sort by ts
        out = list(d.values())
        out.sort(key=lambda x: x["ts"])  # oldest first
        return out

    running_flows = flows(running["id"]) if running else []
    queued_flows_map = {c["id"]: flows(c["id"]) for c in queued}
    recent_flows_map = {c["id"]: flows(c["id"]) for c in recent}
    return render_template("status.html", running=running, running_flows=running_flows, queued=queued, queued_flows=queued_flows_map, recent=recent, recent_flows=recent_flows_map)


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
        # identify running challenge to add event log
        cur = dbm.query_one(conn, "SELECT id FROM challenges WHERE status='running' LIMIT 1")
        if cur is not None:
            conn.execute(
                "INSERT INTO challenge_requests(challenge_id, api, req_key, phase, status_code, req_body, res_body, ts) VALUES(?,?,?,?,?,?,?,?)",
                (int(cur["id"]), "event", str(uuid.uuid4()), "cancel", 0, "{}", "{}", now),
            )
        conn.execute("UPDATE challenges SET status='giveup', finished_at=? WHERE status='running'", (now,))
    # nudge scheduler to grant next
    _grant_waiting_if_idle()
    # Return refreshed status partial so HTMX can replace it immediately
    _bus.emit("cancel")
    running = dbm.query_one(conn, "SELECT * FROM challenges WHERE status='running' ORDER BY started_at DESC LIMIT 1")
    queued = dbm.query_all(conn, "SELECT * FROM challenges WHERE status='queued' ORDER BY effective_priority DESC, enqueued_at ASC LIMIT 20")
    recent = dbm.query_all(conn, "SELECT * FROM challenges WHERE status IN ('success','timeout','giveup','error') ORDER BY finished_at DESC LIMIT 20")
    def fmt_ts(ts: str) -> str:
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
    def flows(ch_id: int):
        rows = dbm.query_all(conn, "SELECT api, req_key, phase, status_code, ts, req_body, res_body FROM challenge_requests WHERE challenge_id=? ORDER BY id ASC", (ch_id,))
        d = {}
        import json as _json
        for r in rows:
            key = r["req_key"] or f"{r['api']}-{r['ts']}"
            item = d.get(key)
            if not item:
                item = {
                    "api": r["api"],
                    "phases": set(),
                    "ts": r["ts"],
                    "tstr": fmt_ts(r["ts"]),
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
    running_flows = flows(running["id"]) if running else []
    queued_flows = {c["id"]: flows(c["id"]) for c in queued}
    recent_flows = {c["id"]: flows(c["id"]) for c in recent}
    return render_template("status.html", running=running, running_flows=running_flows, queued=queued, queued_flows=queued_flows, recent=recent, recent_flows=recent_flows)


@app.route("/minotaur/healthz")
def healthz():
    try:
        dbm.query_one(conn, "SELECT 1")
        return jsonify({"status": "ok"})
    except Exception:
        return jsonify({"status": "ng"}), 500


# ========== Transparent API ==========
def _log_ch_req(ch_id: int, api: str, req_key: str, phase: str, status_code: int, req_obj: Dict[str, Any], res_obj: Dict[str, Any]) -> None:
    try:
        conn.execute(
            "INSERT INTO challenge_requests(challenge_id, api, req_key, phase, status_code, req_body, res_body, ts) VALUES(?,?,?,?,?,?,?,?)",
            (int(ch_id), api, req_key, phase, int(status_code), json.dumps(req_obj, ensure_ascii=False), json.dumps(res_obj, ensure_ascii=False), utcnow_str()),
        )
    except Exception:
        pass
@app.route("/select", methods=["POST"])
def select_route():
    body = request.get_json(silent=True) or {}
    problem = body.get("problemName")
    if not isinstance(problem, str) or not problem:
        return jsonify({"error": "problemName required"}), 400

    agent_id = request.headers.get("X-Agent-ID") or None
    agent_name = request.headers.get("X-Agent-Name") or None
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
            "INSERT INTO challenges(ticket, problem_id, agent_id, agent_name, source_ip, git_sha, base_priority, effective_priority, status, enqueued_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (ticket, problem, agent_id, agent_name, request.remote_addr or "-", git_sha, base_prio, eff, "queued", enq),
        )
    ch = dbm.query_one(conn, "SELECT id FROM challenges WHERE ticket=?", (ticket,))
    ch_id = int(ch["id"]) if ch else None
    if ch_id is not None:
        rk = str(uuid.uuid4())
        _log_ch_req(ch_id, "select", rk, "agent_in", 0, {"problemName": problem}, {})
    try:
        qn = dbm.query_one(conn, "SELECT COUNT(1) AS n FROM challenges WHERE status='queued'")
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
                running = None
                if agent_name is not None and agent_id is not None:
                    running = dbm.query_one(conn, "SELECT id, ticket FROM challenges WHERE status='running' AND agent_name=? AND agent_id=? LIMIT 1", (agent_name, agent_id))
                if running is not None:
                    # Force-terminate the running trial for this agent
                    conn.execute(
                        "UPDATE challenges SET status='giveup', finished_at=? WHERE id=? AND status='running'",
                        (utcnow_str(), running["id"]),
                    )
                    try:
                        _log_ch_req(int(running["id"]), "event", str(uuid.uuid4()), "preempt_giveup", 0, {}, {})
                    except Exception:
                        pass
                    preempt_giveup = True
                # If no other running exists now, grant this ticket immediately
                none_running = dbm.query_one(conn, "SELECT id FROM challenges WHERE status='running' LIMIT 1") is None
                if none_running:
                    sid = str(uuid.uuid4())
                    lease = (datetime.now(timezone.utc) + timedelta(seconds=s.trial_ttl_sec)).isoformat().replace("+00:00", "Z")
                    conn.execute(
                        "UPDATE challenges SET status='running', started_at=?, lease_expire_at=?, session_id=? WHERE ticket=? AND status='queued'",
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
            myrow = dbm.query_one(conn, "SELECT id, status FROM challenges WHERE ticket=?", (ticket,))
            running = dbm.query_one(conn, "SELECT id, ticket FROM challenges WHERE status='running' LIMIT 1")
            cnt_other = dbm.query_one(
                conn,
                "SELECT COUNT(1) AS n FROM challenges WHERE status='queued' AND ticket<>?",
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
                    "UPDATE challenges SET status='giveup', finished_at=? WHERE id=? AND status='running'",
                    (utcnow_str(), running["id"]),
                )
                try:
                    _log_ch_req(int(running["id"]), "event", str(uuid.uuid4()), "preempt_giveup", 0, {}, {})
                except Exception:
                    pass
                did_giveup = True
                sid = str(uuid.uuid4())
                lease = (datetime.now(timezone.utc) + timedelta(seconds=s.trial_ttl_sec)).isoformat().replace("+00:00", "Z")
                conn.execute(
                    "UPDATE challenges SET status='running', started_at=?, lease_expire_at=?, session_id=? WHERE ticket=? AND status='queued'",
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
        row = dbm.query_one(conn, "SELECT status, session_id, id FROM challenges WHERE ticket=?", (ticket,))
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
        # Fallback to async ack (no Retry-After header)
        return make_response((jsonify({"status": "queued", "ticket": ticket}), 202))

    # Forward upstream select now that we are granted
    if ch_id is not None:
        try:
            rk
        except NameError:
            rk = str(uuid.uuid4())
        _log_ch_req(ch_id, "select", rk, "to_upstream", 0, {"problemName": problem}, {})
    try:
        out = proxy.forward("/select", started, {"problemName": problem, "id": "ignored"}, meta={"agent_id": agent_id, "git_sha": git_sha})
    except UpstreamError as ue:
        with conn:
            conn.execute(
                "UPDATE challenges SET status='error', finished_at=? WHERE ticket=?",
                (utcnow_str(), ticket),
            )
        if ch_id is not None:
            _log_ch_req(ch_id, "select", rk, "from_upstream", int(ue.status), {"problemName": problem}, ue.payload)
        return jsonify({"error": "upstream_error", "detail": ue.payload}), 502

    # success: set lease and return upstream response
    lease = (datetime.now(timezone.utc) + timedelta(seconds=s.trial_ttl_sec)).isoformat().replace("+00:00", "Z")
    with conn:
        conn.execute("UPDATE challenges SET lease_expire_at=? WHERE ticket=?", (lease, ticket))
    if ch_id is not None:
        _log_ch_req(ch_id, "select", rk, "from_upstream", 200, {"problemName": problem}, out)
        _log_ch_req(ch_id, "select", rk, "agent_out", 200, {"problemName": problem}, out)
        _log_ch_req(ch_id, "select", rk, "agent_out", 200, {"problemName": problem}, out)
    _log(f"[agent] granted ticket={ticket} sid={started} lease={lease}")
    return make_response(jsonify(out))


def _current_running_session() -> Optional[str]:
    row = dbm.query_one(conn, "SELECT session_id FROM challenges WHERE status='running' LIMIT 1")
    return row["session_id"] if row and row["session_id"] else None


def _touch_lease() -> None:
    lease = (datetime.now(timezone.utc) + timedelta(seconds=s.trial_ttl_sec)).isoformat().replace("+00:00", "Z")
    with conn:
        conn.execute("UPDATE challenges SET lease_expire_at=? WHERE status='running'", (lease,))


@app.route("/explore", methods=["POST"])
def explore_route():
    body = request.get_json(silent=True) or {}
    agent_id = request.headers.get("X-Agent-ID") or None
    git_sha = request.headers.get("X-Agent-Git") or None
    sid = _current_running_session()
    if not sid:
        return jsonify({"error": "no active session"}), 409
    state = {"query_count": (dbm.query_one(conn, "SELECT upstream_query_count FROM challenges WHERE status='running'") or {"upstream_query_count": 0})["upstream_query_count"]}
    rk = str(uuid.uuid4())
    cur = dbm.query_one(conn, "SELECT id FROM challenges WHERE status='running' LIMIT 1")
    ch_id = int(cur["id"]) if cur else None
    if ch_id is not None:
        _log_ch_req(ch_id, "explore", rk, "agent_in", 0, body, {})
        _log_ch_req(ch_id, "explore", rk, "to_upstream", 0, body, {})
    try:
        out = proxy.forward("/explore", sid, body, state, meta={"agent_id": agent_id, "git_sha": git_sha})
    except UpstreamError as ue:
        if ch_id is not None:
            _log_ch_req(ch_id, "explore", rk, "from_upstream", int(ue.status), body, ue.payload)
        return jsonify({"error": "upstream_error", "detail": ue.payload}), 502
    # update qc and lease
    with conn:
        conn.execute("UPDATE challenges SET upstream_query_count=? WHERE status='running'", (int(out.get("queryCount", 0)),))
    _touch_lease()
    try:
        plans = body.get("plans") or []
        _log(f"[agent] /explore agent_id={agent_id or '-'} sid={sid} plans={len(plans)} qc->{int(out.get('queryCount',0))}")
    except Exception:
        pass
    if ch_id is not None:
        _log_ch_req(ch_id, "explore", rk, "from_upstream", 200, body, out)
        _log_ch_req(ch_id, "explore", rk, "agent_out", 200, body, out)
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
    rk = str(uuid.uuid4())
    cur = dbm.query_one(conn, "SELECT id FROM challenges WHERE status='running' LIMIT 1")
    ch_id = int(cur["id"]) if cur else None
    if ch_id is not None:
        _log_ch_req(ch_id, "guess", rk, "agent_in", 0, body, {})
        _log_ch_req(ch_id, "guess", rk, "to_upstream", 0, body, {})
    try:
        out = proxy.forward("/guess", sid, body, meta={"agent_id": agent_id, "git_sha": git_sha})
    except UpstreamError as ue:
        if ch_id is not None:
            _log_ch_req(ch_id, "guess", rk, "from_upstream", int(ue.status), body, ue.payload)
        return jsonify({"error": "upstream_error", "detail": ue.payload}), 502
    running_id_row = dbm.query_one(conn, "SELECT id FROM challenges WHERE status='running' LIMIT 1")
    if bool(out.get("correct")):
        with conn:
            conn.execute(
                "UPDATE challenges SET status='success', finished_at=? WHERE status='running'",
                (utcnow_str(),),
            )
    _touch_lease()
    _log(f"[agent] /guess agent_id={agent_id or '-'} sid={sid} correct={bool(out.get('correct'))}")
    if ch_id is not None:
        _log_ch_req(ch_id, "guess", rk, "from_upstream", 200, body, out)
        _log_ch_req(ch_id, "guess", rk, "agent_out", 200, body, out)
    _bus.emit("guess")
    return jsonify(out)


def main() -> None:
    app.run(host="0.0.0.0", port=s.port, debug=False)


if __name__ == "__main__":
    main()
