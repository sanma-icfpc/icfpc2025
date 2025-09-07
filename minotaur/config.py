from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


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
    # Persisted settings YAML path
    settings_file: str

    # WSGI/Waitress
    waitress_threads: int
    # Scheduler tuning
    pin_hold_sec: float

    @property
    def is_mock(self) -> bool:
        return self.mock or not self.official_base


def load_settings_from_env() -> Settings:
    # Resolve DB path with optional in-memory mode
    env_db = os.getenv("MINOTAUR_DB") or os.getenv("MINOTUAR_DB") or None
    mem_flag = getenv_bool("DB_IN_MEMORY") or getenv_bool("MINOTAUR_DB_IN_MEMORY")
    db_path: str
    if mem_flag or (env_db or "").strip().lower() in {":memory:", "memory"}:
        # Use a shared in-memory database URI so that multiple connections (threads)
        # see the same database. Requires sqlite3.connect(..., uri=True).
        db_path = "file:minotaur?mode=memory&cache=shared"
        print('using in-memory DB')
    else:
        # If MINOTAUR_DB is set and is a relative filesystem path (not a URI),
        # resolve it under the package base dir for stability across CWDs.
        if env_db:
            env_db_s = env_db.strip()
            if not (env_db_s.startswith("file:") or env_db_s.startswith("sqlite:")):
                p = Path(env_db_s)
                if not p.is_absolute():
                    env_db_s = str(Path(BASE_DIR) / p)
            db_path = env_db_s
        else:
            db_path = str(Path(BASE_DIR) / "coordinator.sqlite")
        print('using disk-backed DB')

    return Settings(
        official_base=os.getenv("OFFICIAL_BASE") or None,
        icfp_id=os.getenv("ICFP_ID"),
        port=int(os.getenv("PORT", "19384")),
        log_dir=resolve_under_base(os.getenv("LOG_DIR") or "logs"),
        # Accept both legacy typo and correct var name; default under package dir
        db_path=db_path,
        trial_ttl_sec=int(os.getenv("TRIAL_TTL_SEC", "60")),
        base_priority_default=int(os.getenv("BASE_PRIORITY_DEFAULT", "50")),
        ageing_every_min=int(os.getenv("AGEING_EVERY_MIN", "5")),
        ageing_cap=int(os.getenv("AGEING_CAP", "100")),
        post_success_retry_cap=int(os.getenv("POST_SUCCESS_RETRY_CAP", "3")),
        post_success_penalty=int(os.getenv("POST_SUCCESS_PENALTY", "-10")),
        max_qps=int(os.getenv("MAX_QPS", "0")),
        # Default to package path users.yaml; resolve relative paths under package dir
        auth_file=resolve_under_base(os.getenv("AUTH_FILE") or "users.yaml"),
        settings_file=resolve_under_base(os.getenv("SETTINGS_FILE") or "settings.yaml"),
        waitress_threads=int(os.getenv("WAITRESS_THREADS", "128")),
        pin_hold_sec=float(os.getenv("PIN_HOLD_SEC", "10.0")),
    )


def load_persisted_settings(path: str) -> Dict[str, Any]:
    try:
        if yaml is None:
            return {}
        p = Path(path)
        if not p.exists():
            return {}
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {str(k): v for k, v in data.items()}
        return {}
    except Exception:
        return {}


def save_persisted_settings(path: str, kv: Dict[str, Any]) -> None:
    try:
        if yaml is None:
            return
        p = Path(path)
        # merge with existing
        cur = load_persisted_settings(path)
        cur.update(kv)
        p.write_text(yaml.safe_dump(cur, allow_unicode=True, sort_keys=True), encoding="utf-8")
    except Exception:
        pass
