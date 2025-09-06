from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict


class JsonlLogger:
    def __init__(self, log_dir: str) -> None:
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self._lock = threading.Lock()
        self._date = ""
        self._fh = None

    def _ensure_file(self) -> None:
        d = datetime.now(timezone.utc).strftime("%Y%m%d")
        if self._fh is not None and self._date == d:
            return
        # rotate
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:
                pass
            self._fh = None
        self._date = d
        path = os.path.join(self.log_dir, f"http-{d}.jsonl")
        self._fh = open(path, "a", encoding="utf-8")

    def write(self, rec: Dict[str, Any]) -> None:
        with self._lock:
            self._ensure_file()
            rec = dict(rec)
            rec.setdefault("ts", datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
            self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._fh.flush()

