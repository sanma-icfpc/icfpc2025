from __future__ import annotations

import os
import threading
import time
from typing import Any


def _ts() -> str:
    try:
        from datetime import datetime, timezone

        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    except Exception:
        return "?"


def _log(msg: str) -> None:
    try:
        print(f"[{_ts()}] {msg}", flush=True)
    except Exception:
        pass


class MemoryRebooter:
    """
    Background watcher that gracefully terminates the process when RSS exceeds
    the configured threshold and no agent is currently running.

    - Controlled by Settings.rss_reboot_mb (<= 0 disables)
    - Checks every few seconds with psutil
    - Exits via os._exit(0) when safe
    """

    def __init__(self, ctx: Any, interval_sec: float = 3.0) -> None:
        self.ctx = ctx
        self.interval_sec = max(1.0, float(interval_sec))
        self._th = threading.Thread(target=self._run, name="MemoryRebooter", daemon=True)

    def start(self) -> None:
        try:
            self._th.start()
        except Exception:
            _log("[rebooter] failed to start thread")

    def _current_rss_mb(self) -> float:
        try:
            import psutil  # type: ignore

            p = psutil.Process(os.getpid())
            mem = p.memory_info()
            rss = getattr(mem, "rss", None)
            if rss is None:
                return 0.0
            return float(rss) / (1024.0 * 1024.0)
        except Exception:
            return 0.0

    def _has_running(self) -> bool:
        try:
            row = self.ctx.conn.execute(
                "SELECT 1 FROM challenges WHERE status='running' LIMIT 1"
            ).fetchone()
            return bool(row is not None)
        except Exception:
            return False

    def _run(self) -> None:
        # Avoid excessive logging: only log when crossing threshold state
        above_logged = False
        paused_once = False
        crossed_at: float | None = None
        while True:
            try:
                thr = int(getattr(self.ctx.s, "rss_reboot_mb", 0) or 0)
            except Exception:
                thr = 0
            if thr > 0:
                cur = self._current_rss_mb()
                if cur >= thr:
                    if not above_logged:
                        _log(f"[rebooter] RSS {cur:.1f} MB >= threshold {thr} MB; waiting for safe point…")
                        above_logged = True
                    # Pause granting new runs once when threshold is crossed
                    if not paused_once:
                        try:
                            self.ctx.coord.pause_grants(True)
                            paused_once = True
                            crossed_at = time.time()
                            _log("[rebooter] paused scheduler grants (drain mode)")
                        except Exception:
                            pass
                    if not self._has_running():
                        waited = 0.0
                        try:
                            if crossed_at is not None:
                                waited = max(0.0, time.time() - crossed_at)
                        except Exception:
                            waited = 0.0
                        _log(
                            f"[rebooter] No running agents and RSS {cur:.1f} MB >= {thr} MB — safe to terminate (waited {waited:.1f}s since drain start)"
                        )
                        # Prefer a graceful shutdown path if provided by hosting; fallback to hard exit
                        try:
                            # If app exposes a shutdown hook in context, call it; else os._exit
                            shut = getattr(self.ctx, "shutdown", None)
                            if callable(shut):
                                shut()
                            else:
                                os._exit(0)
                        except Exception:
                            try:
                                os._exit(0)
                            except Exception:
                                pass
                else:
                    if above_logged:
                        _log(f"[rebooter] RSS back under threshold: {cur:.1f} MB < {thr} MB")
                    above_logged = False
                    if paused_once:
                        try:
                            self.ctx.coord.pause_grants(False)
                            _log("[rebooter] resumed scheduler grants (threshold cleared)")
                        except Exception:
                            pass
                        paused_once = False
                        crossed_at = None
            time.sleep(self.interval_sec)
