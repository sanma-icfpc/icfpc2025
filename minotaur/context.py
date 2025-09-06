from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import Settings


@dataclass
class AppCtx:
    s: Settings
    conn: Any
    logger: Any
    proxy: Any
    coord: Any
    ui_guard: Any
    bus: Any

