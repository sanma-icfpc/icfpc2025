from __future__ import annotations

from flask import Blueprint, jsonify, request
from ..context import AppCtx
from typing import Dict, List
import os, time as _t


def create_admin_sys_bp(ctx: AppCtx) -> Blueprint:
    bp = Blueprint("ui_admin_sys", __name__, url_prefix="/minotaur")
    guard = ctx.ui_guard

    @bp.route("/fd_info")
    @guard.require()
    def fd_info():
        try:
            import platform as _platform
            import time as _time2
            plat = (_platform.system() or "").lower()
            import json as _json  # noqa: F401
            soft = None
            hard = None
            open_cnt = None
            method = None
            kind = "fd"
            by_kind: Dict[str, int] = {}
            samples: List[str] = []
            max_samples = 200
            try:
                import resource  # type: ignore
                soft_h, hard_h = resource.getrlimit(resource.RLIMIT_NOFILE)
                soft = int(soft_h) if soft_h is not None else None
                hard = int(hard_h) if hard_h is not None else None
            except Exception:
                pass
            if plat == "linux":
                p = f"/proc/{os.getpid()}/fd"
                try:
                    if os.path.isdir(p):
                        it = os.scandir(p)
                        try:
                            for de in it:
                                try:
                                    target = os.readlink(de.path)
                                except Exception:
                                    target = "?"
                                t = target
                                k = "file"
                                if t.startswith("socket:"):
                                    k = "socket"
                                elif t.startswith("pipe:"):
                                    k = "pipe"
                                elif t.startswith("anon_inode:"):
                                    k = "anon_inode"
                                elif t.startswith("eventfd"):
                                    k = "eventfd"
                                elif t.startswith("memfd:"):
                                    k = "memfd"
                                elif t.startswith("/dev/pts") or t.startswith("/dev/tty"):
                                    k = "tty"
                                by_kind[k] = int(by_kind.get(k, 0)) + 1
                                if len(samples) < max_samples:
                                    samples.append(t)
                            open_cnt = sum(by_kind.values())
                            method = "proc_fd"
                        finally:
                            try:
                                it.close()  # type: ignore[attr-defined]
                            except Exception:
                                pass
                except Exception:
                    pass
            out = {
                "platform": _platform.system(),
                "open": open_cnt,
                "kind": kind,
                "soft": soft,
                "hard": hard,
                "byKind": by_kind,
                "samples": samples,
                "method": method,
                "ts": _time2.time(),
            }
            return jsonify(out)
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    @bp.route("/mem_info")
    @guard.require()
    def mem_info():
        import psutil  # type: ignore
        import gc as _gc
        try:
            p = psutil.Process(os.getpid())
            mem = p.memory_info()
            rss = getattr(mem, "rss", None)
            vms = getattr(mem, "vms", None)
            thr_mb = int(getattr(ctx.s, "rss_reboot_mb", 0) or 0)
            rss_mb_val = (rss / (1024 * 1024)) if rss is not None else None
            over_thr = bool(thr_mb > 0 and (rss_mb_val is not None) and (rss_mb_val >= thr_mb))
            out = {
                "rss": int(rss) if rss is not None else None,
                "rssMB": rss_mb_val,
                "vms": int(vms) if vms is not None else None,
                "vmsMB": (vms / (1024 * 1024)) if vms is not None else None,
                "threads": int(p.num_threads()),
                "py_threads": None,
                "py_gc": None,
                "ts": _t.time(),
                # Augment for UI badges
                "rssThrMB": thr_mb,
                "draining": bool(getattr(getattr(ctx, "coord", None), "is_paused", lambda: False)()),
                "overThr": over_thr,
            }
            try:
                import threading as _th
                out["py_threads"] = len(list(_th.enumerate()))
            except Exception:
                pass
            try:
                out["py_gc"] = list(_gc.get_count())
            except Exception:
                pass
            return jsonify(out)
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    @bp.route("/mem_objects")
    @guard.require()
    def mem_objects():
        import gc as _gc
        import collections as _c
        try:
            objs = _gc.get_objects()  # type: ignore[attr-defined]
            c = _c.Counter(type(o).__name__ for o in objs)
            limit = 40
            try:
                limit = int(request.args.get("limit", "40") or "40")
            except Exception:
                limit = 40
            top = c.most_common(max(1, min(200, limit)))
            out = {"topTypes": [{"type": k, "count": v} for k, v in top]}
            try:
                out["gcCount"] = tuple(_gc.get_count())
            except Exception:
                pass
            try:
                out["gcThreshold"] = tuple(_gc.get_threshold())
            except Exception:
                pass
            try:
                out["gcStats"] = _gc.get_stats()  # type: ignore[attr-defined]
            except Exception:
                pass
            out["elapsedMs"] = 0
            return jsonify(out)
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    @bp.route("/gc_collect", methods=["POST"])
    @guard.require()
    def gc_collect():
        import gc as _gc
        try:
            before = tuple(_gc.get_count())
        except Exception:
            before = None
        try:
            collected = int(_gc.collect())
            try:
                unreachable = int(getattr(_gc, "garbage", []) and len(_gc.garbage) or 0)
            except Exception:
                unreachable = None
            after = None
            try:
                after = tuple(_gc.get_count())
            except Exception:
                pass
            return jsonify({
                "ok": True,
                "collected": collected,
                "unreachable": unreachable,
                "before": before,
                "after": after,
            })
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    @bp.route("/shutdown", methods=["POST"])
    @guard.require()
    def shutdown():
        try:
            # Let the WSGI server handle process exit
            os._exit(0)
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    return bp
