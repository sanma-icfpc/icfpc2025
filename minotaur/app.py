from __future__ import annotations

import threading
from datetime import datetime, timezone

from flask import Flask, Response, jsonify, request

from . import db as dbm
from .auth import BasicAuthGuard, load_users
from .config import (
    Settings,
    load_settings_from_env,
    resolve_under_base,
    load_persisted_settings,
)
from .context import AppCtx
from .eventbus import EventBus
from .logging import JsonlLogger
from .proxy import UpstreamProxy
from .scheduler import Coordinator
from .ui_routes import register_ui_routes
from .api_routes import register_api_routes
from .api_select import register_select_routes


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

# Database
conn = dbm.get_conn(s.db_path)
dbm.init_schema(conn)
# Reap lingering trials
try:
    now = utcnow_str()
    with conn:
        conn.execute(
            "UPDATE challenges SET status='interrupted', finished_at=? WHERE status IN ('running','queued')",
            (now,),
        )
    _log("[boot] reaped lingering trials: running/queued -> interrupted")
except Exception:
    pass

# Infra
logger = JsonlLogger(s.log_dir)
proxy = UpstreamProxy(s, logger)
coord = Coordinator(s, conn, logger)

# Auth
users = load_users(s.auth_file)
_log(f"[boot] auth_file={s.auth_file} users={[u.username for u in users]}")
_ui_guard = BasicAuthGuard(users)

# Events
bus = EventBus()
coord.on_change = lambda ev: bus.emit(ev)
coord.start()

# Access logging (directional)
AGENT_ENDPOINTS = {"/select", "/explore", "/guess"}


def _body_snippet() -> str:
    try:
        if request.method in ("POST", "PUT", "PATCH"):
            data = request.get_data(cache=True, as_text=True)
            if not data:
                return ""
            s_ = data.strip().replace("\n", " ")
            if len(s_) > 500:
                s_ = s_[:500] + "..."
            return s_
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
            return
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


# Register route groups
ctx = AppCtx(s=s, conn=conn, logger=logger, proxy=proxy, coord=coord, ui_guard=_ui_guard, bus=bus)
register_ui_routes(app, ctx)
register_select_routes(app, ctx)
register_api_routes(app, ctx)


@app.route("/minotaur/healthz")
def healthz():
    try:
        dbm.query_one(conn, "SELECT 1")
        return jsonify({"status": "ok"})
    except Exception:
        return jsonify({"status": "ng"}), 500


def main() -> None:
    app.run(host="0.0.0.0", port=s.port, debug=False)


if __name__ == "__main__":
    main()
