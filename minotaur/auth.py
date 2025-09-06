from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
from dataclasses import dataclass
import sys
from typing import Dict, List, Optional, Tuple


import yaml  # type: ignore


@dataclass
class User:
    username: str
    password: str  # may be plaintext or pbkdf2$sha256$<salt_b64>$<hash_b64>


def load_users(auth_file: str) -> List[User]:
    try:
        print(f"[auth] loading users from {auth_file}", file=sys.stderr, flush=True)
        if not os.path.exists(auth_file):
            print(f"[auth] file not found: {auth_file}", file=sys.stderr, flush=True)
            return []
        _, ext = os.path.splitext(auth_file.lower())
        with open(auth_file, "r", encoding="utf-8") as f:
            text = f.read()
        data: List[Dict]
        if ext in (".yaml", ".yml"):
            try:
                data = yaml.safe_load(text) or []  # type: ignore
            except Exception as e:
                print(f"[auth] YAML parse error: {e}", file=sys.stderr, flush=True)
                raise
        else:
            # accept JSON fallback: [{"username":"u","password":"..."}, ...]
            data = json.loads(text) if text.strip() else []
        out: List[User] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            u = str(item.get("username", "")).strip()
            p = str(item.get("password", "")).strip()
            if u and p:
                out.append(User(username=u, password=p))
        print(f"[auth] users loaded: {[u.username for u in out]}", file=sys.stderr, flush=True)
        return out
    except Exception as e:
        print(f"[auth] error loading users: {e}. If missing, install PyYAML: pip install pyyaml", file=sys.stderr, flush=True)
        return []


def pbkdf2_sha256_hash(password: str, salt: Optional[bytes] = None, rounds: int = 200_000) -> str:
    import os as _os

    s = salt if salt is not None else _os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), s, rounds, dklen=32)
    return "pbkdf2$sha256$" + base64.b64encode(s).decode("ascii") + "$" + base64.b64encode(dk).decode("ascii")


def pbkdf2_sha256_verify(password: str, encoded: str, rounds: int = 200_000) -> bool:
    try:
        scheme, algo, salt_b64, hash_b64 = encoded.split("$", 3)
        if scheme != "pbkdf2" or algo != "sha256":
            return False
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(hash_b64)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, rounds, dklen=len(expected))
        return hmac.compare_digest(dk, expected)
    except Exception:
        return False


def parse_basic_auth(auth_header: Optional[str]) -> Optional[Tuple[str, str]]:
    if not auth_header:
        return None
    try:
        scheme, b64 = auth_header.split(" ", 1)
        if scheme.lower() != "basic":
            return None
        raw = base64.b64decode(b64).decode("utf-8")
        if ":" not in raw:
            return None
        user, pwd = raw.split(":", 1)
        return user, pwd
    except Exception:
        return None


class BasicAuthGuard:
    def __init__(self, users: List[User]):
        self._users: Dict[str, User] = {u.username: u for u in users}
        # デフォルト非表示。AUTH_DEBUG=1 もしくは MINOTAUR_AUTH_DEBUG=1 のときのみ有効化。
        v1 = os.getenv("AUTH_DEBUG", "").strip().lower()
        v2 = os.getenv("MINOTAUR_AUTH_DEBUG", "").strip().lower()
        self._debug = v1 in ("1", "true", "yes", "on") or v2 in ("1", "true", "yes", "on")

    def check(self, username: str, password: str) -> bool:
        u = self._users.get(username)
        if not u:
            return False
        stored = u.password
        if stored.startswith("pbkdf2$"):
            return pbkdf2_sha256_verify(password, stored)
        # plaintext fallback
        return hmac.compare_digest(stored, password)

    def require(self):  # Flask decorator without extra deps
        from functools import wraps
        from flask import request, Response

        def decorator(f):
            @wraps(f)
            def wrapper(*args, **kwargs):
                auth_header = request.headers.get("Authorization")
                creds = parse_basic_auth(auth_header)
                # Debugログ（デフォルトOFF）
                if self._debug:
                    try:
                        # パスワードは伏せ字にする
                        dbg_creds = None
                        if creds:
                            u, p = creds
                            dbg_creds = (u, "***")
                        print(
                            f"[auth debug] header_present={bool(auth_header)} creds={dbg_creds} users={[u for u in self._users.keys()]}",
                            file=sys.stderr,
                            flush=True,
                        )
                    except Exception:
                        pass
                if not creds or not self.check(*creds):
                    return Response(
                        status=401,
                        headers={"WWW-Authenticate": 'Basic realm="Minotaur UI"'},
                    )
                return f(*args, **kwargs)

            return wrapper

        return decorator


if __name__ == "__main__":
    # Small CLI to hash passwords: python -m minotaur.auth <password>
    import sys

    if len(sys.argv) == 2:
        print(pbkdf2_sha256_hash(sys.argv[1]))
    else:
        print("Usage: python -m minotaur.auth <password>")
