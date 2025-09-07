from __future__ import annotations

from flask import Blueprint, jsonify
from ..context import AppCtx
from typing import Dict, Any, Optional
import os, time as _time2


_THREAD_SAMPLES: Dict[int, Dict[str, float]] = {}


def create_admin_threads_bp(ctx: AppCtx) -> Blueprint:
    bp = Blueprint("ui_admin_threads", __name__, url_prefix="/minotaur")
    guard = ctx.ui_guard

    @bp.route("/threads_os")
    @guard.require()
    def threads_os():
        import platform as _platform
        import threading as _th
        out: Dict[str, Any] = {"pid": os.getpid(), "platform": _platform.system(), "threads": [], "summary": {}, "ts": _time2.time()}

        def _merge_py(meta: Dict[str, Any], tid: Optional[int]) -> Dict[str, Any]:
            try:
                # Map Python threads by native id if available (3.8+)
                py_threads: Dict[int, Dict[str, Any]] = {}
                for t in _th.enumerate():
                    try:
                        nid = getattr(t, "native_id", None)
                    except Exception:
                        nid = None
                    if nid is not None:
                        py_threads[int(nid)] = {
                            "py_name": t.name,
                            "daemon": bool(getattr(t, "daemon", False)),
                        }
                if tid is not None and int(tid) in py_threads:
                    meta.update(py_threads[int(tid)])
            except Exception:
                pass
            return meta

        plat = (_platform.system() or "").lower()
        if plat == "linux":
            try:
                import glob
                pid = os.getpid()
                f = f"/proc/{pid}/task/*"
                for tdir in glob.glob(f):
                    try:
                        tid = int(os.path.basename(tdir))
                    except Exception:
                        continue
                    st = {"id": tid}
                    try:
                        with open(f"{tdir}/status", "r", encoding="utf-8", errors="ignore") as f:
                            for ln in f:
                                if ln.startswith("State:"):
                                    st["state"] = ln.split(":", 1)[1].strip()
                                    break
                    except Exception:
                        pass
                    try:
                        with open(f"{tdir}/stat", "r", encoding="utf-8", errors="ignore") as f:
                            cols = f.read().strip().split()
                            if len(cols) >= 14:
                                utime = float(cols[13])
                                stime = float(cols[14]) if len(cols) > 14 else 0.0
                                st["cpu_time_sec"] = (utime + stime) / 100.0
                    except Exception:
                        pass
                    try:
                        with open(f"{tdir}/wchan", "r", encoding="utf-8", errors="ignore") as f:
                            st["wchan"] = (f.read().strip() or None)
                    except Exception:
                        pass
                    out["threads"].append(_merge_py(st, tid))
                summ = {}
                for t in out["threads"]:
                    stv = t.get("state") or "?"
                    key = stv.split()[0] if stv else "?"
                    summ[key] = int(summ.get(key, 0)) + 1
                out["summary"] = summ
            except Exception as e:
                out["error"] = f"linux_proc_read_failed: {e}"
        else:
            try:
                import psutil  # type: ignore
                p = psutil.Process(os.getpid())
                thr = p.threads()
                for th in thr:
                    meta = {
                        "id": int(getattr(th, 'id', 0)),
                        "cpu_time_sec": float(getattr(th, 'user_time', 0.0)) + float(getattr(th, 'system_time', 0.0)),
                    }
                    out["threads"].append(_merge_py(meta, meta["id"]))
                out["summary"] = {"count": len(out["threads"]) }
            except Exception:
                out["threads"] = []
                try:
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
        # Compute simple running/idle estimate based on CPU time deltas across calls
        now_ts = out.get("ts") if isinstance(out.get("ts"), (int, float)) else _time2.time()
        try:
            for t in out["threads"]:
                tid = t.get("id")
                if isinstance(tid, int):
                    cpu = float(t.get("cpu_time_sec") or 0.0)
                    prev = _THREAD_SAMPLES.get(tid)
                    if prev is None:
                        _THREAD_SAMPLES[tid] = {"cpu": cpu, "last": float(now_ts)}
                        t["statusKind"] = "waiting"
                        t["idleForSec"] = 0.0
                    else:
                        progressed = (cpu > prev.get("cpu", 0.0) + 1e-6)
                        if progressed:
                            prev["cpu"] = cpu
                            prev["last"] = float(now_ts)
                            t["statusKind"] = "running"
                            t["idleForSec"] = 0.0
                        else:
                            t["statusKind"] = "waiting"
                            t["idleForSec"] = max(0.0, float(now_ts) - float(prev.get("last", now_ts)))
                else:
                    t["statusKind"] = t.get("statusKind") or "waiting"
                    t["idleForSec"] = t.get("idleForSec") or 0.0
        except Exception:
            pass
        return jsonify(out)

    return bp
