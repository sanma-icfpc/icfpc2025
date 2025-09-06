from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


def getenv_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


BASE_DIR = str(Path(__file__).resolve().parent)


def resolve_under_base(path_str: str) -> str:
    p = Path(path_str)
    return str(p if p.is_absolute() else (Path(BASE_DIR) / p))


@dataclass
class Settings:
    # Upstream / Mock
    official_base: Optional[str]
    mock: bool
    icfp_id: Optional[str]

    # Server
    port: int
    log_dir: str
    db_path: str

    # Scheduling / Lease
    trial_ttl_sec: int
    base_priority_default: int
    ageing_every_min: int
    ageing_cap: int
    post_success_retry_cap: int
    post_success_penalty: int

    # Rate limiting (global)
    max_qps: int

    # UI Auth
    auth_file: str

    @property
    def is_mock(self) -> bool:
        return self.mock or not self.official_base


def load_settings_from_env() -> Settings:
    return Settings(
        official_base=os.getenv("OFFICIAL_BASE") or None,
        mock=getenv_bool("MOCK", False),
        icfp_id=os.getenv("ICFP_ID"),
        port=int(os.getenv("PORT", "5000")),
        log_dir=resolve_under_base(os.getenv("LOG_DIR") or "logs"),
        # Accept both legacy typo and correct var name; default under package dir
        db_path=(os.getenv("MINOTAUR_DB") or os.getenv("MINOTUAR_DB") or str(Path(BASE_DIR) / "coordinator.sqlite")),
        trial_ttl_sec=int(os.getenv("TRIAL_TTL_SEC", "60")),
        base_priority_default=int(os.getenv("BASE_PRIORITY_DEFAULT", "50")),
        ageing_every_min=int(os.getenv("AGEING_EVERY_MIN", "5")),
        ageing_cap=int(os.getenv("AGEING_CAP", "100")),
        post_success_retry_cap=int(os.getenv("POST_SUCCESS_RETRY_CAP", "3")),
        post_success_penalty=int(os.getenv("POST_SUCCESS_PENALTY", "-10")),
        max_qps=int(os.getenv("MAX_QPS", "0")),
        # Default to package path users.yaml; resolve relative paths under package dir
        auth_file=resolve_under_base(os.getenv("AUTH_FILE") or "users.yaml"),
    )
