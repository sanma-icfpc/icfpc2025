"""
WSGI entry for running from within the `minotaur/` directory.

Usage (from minotaur/ as CWD):
  - uv:  uv run waitress-serve --listen=*:5000 run:app
  - pip: python -m waitress --listen=*:5000 run:app

This module ensures the repository root is on sys.path so that
`minotaur.app` can be imported as a package.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root (parent of this directory) is importable
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from minotaur.app import app  # noqa: E402  (import after sys.path tweak)

