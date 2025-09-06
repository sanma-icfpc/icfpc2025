from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import requests

from .config import Settings
from .logging import JsonlLogger


class SessionPool:
    def __init__(self) -> None:
        self._pool: Dict[str, requests.Session] = {}
        self._lock = threading.Lock()

    def get(self, sid: str) -> requests.Session:
        with self._lock:
            s = self._pool.get(sid)
            if s is None:
                s = requests.Session()
                self._pool[sid] = s
            return s


class UpstreamProxy:
    def __init__(self, settings: Settings, logger: JsonlLogger) -> None:
        self.s = settings
        self.log = logger
        self.pool = SessionPool()

    def _log(self, dir_: str, path: str, session: Optional[str], status_code: int, req_body: Dict[str, Any], res_body: Dict[str, Any], meta: Optional[Dict[str, Any]] = None) -> None:
        rec = {
            "dir": dir_,
            "path": path,
            "session": session,
            "status_code": status_code,
            "req_body": req_body,
            "res_body": res_body,
        }
        if meta:
            # Surface selected agent metadata for analysis
            if "agent_id" in meta:
                rec["agent_id"] = meta.get("agent_id")
            if "git_sha" in meta:
                rec["agent_git"] = meta.get("git_sha")
        self.log.write(rec)

    def forward(self, path: str, session_id: str, body: Dict[str, Any], state: Optional[Dict[str, Any]] = None, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.s.official_base:
            raise RuntimeError("OFFICIAL_BASE is not configured")

        url = self.s.official_base.rstrip("/") + path
        timeout = (5, 20)  # connect, read
        sess = self.pool.get(session_id)
        # replace id with our team id for upstream
        body2 = dict(body)
        if self.s.icfp_id:
            body2["id"] = self.s.icfp_id
        try:
            aid = (meta or {}).get("agent_id")
            git = (meta or {}).get("git_sha")
            print(f"[{_ts()}] [minotaur -> upstream] POST {url} sid={session_id} agent_id={aid or '-'} git={git or '-'} body={body2}", flush=True)
        except Exception:
            pass
        resp = sess.post(url, json=body2, timeout=timeout)
        text = resp.text
        try:
            data = resp.json()
        except Exception:
            data = {"_raw": text}
        try:
            print(f"[{_ts()}] [upstream -> minotaur] {path} {resp.status_code} body={data}", flush=True)
        except Exception:
            pass
        self._log("fwd" if resp.ok else "fwd_err", path, session_id, resp.status_code, body2, data, meta)
        if not resp.ok:
            raise UpstreamError(resp.status_code, data)
        return data


class UpstreamError(Exception):
    def __init__(self, status: int, payload: Dict[str, Any]):
        super().__init__(f"upstream {status}")
        self.status = status
        self.payload = payload
def _ts() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
