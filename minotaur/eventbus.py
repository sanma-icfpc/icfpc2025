from __future__ import annotations

import threading
from typing import Iterator


class EventBus:
    def __init__(self) -> None:
        self._cv = threading.Condition()
        self._seq = 0

    def emit(self, kind: str = "change") -> None:
        try:
            with self._cv:
                self._seq += 1
                self._cv.notify_all()
        except Exception:
            pass

    def stream(self) -> Iterator[bytes]:
        last = 0
        while True:
            with self._cv:
                self._cv.wait(timeout=60.0)
                cur = self._seq
            if cur != last:
                last = cur
                yield b"event: change\n"
                yield f"data: {cur}\n\n".encode("utf-8")
            else:
                yield b"event: ping\ndata: .\n\n"

