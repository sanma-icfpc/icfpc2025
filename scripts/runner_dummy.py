#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Dict, List, Optional
import random

# Public agent name for this dummy solver
DEFAULT_AGENT_NAME = "tsuzuki:dummy"


def _random_guess_map(n: int) -> Dict:
    """Build a simple, valid map structure with n rooms.

    - Each room has a random label in 0..3.
    - Doors are paired within the same room: a <-> (a+3)%6 to ensure a valid pairing.
    - startingRoom is 0.
    """
    rooms = [random.randint(0, 3) for _ in range(n)]
    connections: List[Dict] = []
    # Pair doors within each room: 0<->3, 1<->4, 2<->5
    for r in range(n):
        for a in (0, 1, 2):
            b = a + 3
            connections.append(
                {"from": {"room": r, "door": a}, "to": {"room": r, "door": b}}
            )
    return {"rooms": rooms, "startingRoom": 0, "connections": connections}


def solve(
    client: object,
    problem: str,
    size: int,
    plan_mode: str,
    status: Optional[object] = None,
    cancel_event: Optional[object] = None,
    use_graffiti: bool = True,
) -> Dict:
    """Dummy solve: perform a few explores, then return a random guess map.

    Expects `client` to expose `explore(plans: List[str]) -> (results, queryCount)`.
    """
    # Prepare a small set of short plans to keep costs low
    seed_plans = [
        "0", "1", "2", "3", "4", "5",
        "01", "23", "45",
        "012", "345",
    ]
    # Add a few random short walks (length 1..3)
    for _ in range(12):
        L = random.randint(1, 3)
        seed_plans.append("".join(random.choice("012345") for _ in range(L)))
    # Deduplicate while preserving order
    seen = set()
    plans: List[str] = []
    for p in seed_plans:
        if p not in seen:
            seen.add(p)
            plans.append(p)

    # Perform explores (ignore results aside from basic logging)
    try:
        _results, qc = client.explore(plans)  # type: ignore[attr-defined]
        if status is not None:
            try:
                status.add_free(f"Dummy explored {len(plans)} plans; server qc={qc}")
            except Exception:
                pass
    except Exception as ex:
        if status is not None:
            try:
                status.add_free(f"/explore failed in dummy: {type(ex).__name__}: {ex}")
            except Exception:
                pass
        # Continue anyway; we are a dummy solver

    # Make a random guess map (may be incorrect)
    guess_map = _random_guess_map(size)
    return guess_map

