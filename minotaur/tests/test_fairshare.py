from __future__ import annotations

import time

from minotaur.fairshare import FairShareCFS


def test_basic_proportional_share_ratios():
    cfs = FairShareCFS()
    cfs.set_priority("A", 100)
    cfs.set_priority("B", 50)
    cfs.set_priority("C", 10)
    t0 = time.monotonic()
    enq = [("A", t0), ("B", t0), ("C", t0)]
    for _ in range(600):
        name = cfs.pick_next(enq)
        assert name in {"A", "B", "C"}
        cfs.on_run(name, 1.0)
    snap = {name: (svc, v) for name, w, svc, v in cfs.snapshot()}
    vmin = min(v for svc, v in snap.values())
    vmax = max(v for svc, v in snap.values())
    assert vmax - vmin < 0.2
    sA, _ = snap["A"]
    sB, _ = snap["B"]
    sC, _ = snap["C"]
    assert sA > sB > sC
    assert 1.6 < (sA / sB) < 2.4
    assert 8.0 < (sA / sC) < 12.5


def test_new_arrival_initialized_to_min_v():
    cfs = FairShareCFS()
    cfs.set_priority("A", 100)
    cfs.set_priority("B", 50)
    t0 = time.monotonic()
    enq = [("A", t0), ("B", t0)]
    for _ in range(150):
        name = cfs.pick_next(enq)
        cfs.on_run(name, 1.0)
    snap_before = {name: (svc, v) for name, w, svc, v in cfs.snapshot()}
    vmin_before = min(v for _, v in snap_before.values())
    cfs.on_arrival("C", priority=10)
    snap_mid = {name: (svc, v) for name, w, svc, v in cfs.snapshot()}
    vC = snap_mid["C"][1]
    assert abs(vC - vmin_before) < 1e-6
    enq2 = [("A", t0), ("B", t0), ("C", t0)]
    for _ in range(300):
        name = cfs.pick_next(enq2)
        cfs.on_run(name, 1.0)
    snap_after = {name: (svc, v) for name, w, svc, v in cfs.snapshot()}
    vmin = min(v for _, v in snap_after.values())
    vmax = max(v for _, v in snap_after.values())
    assert vmax - vmin < 0.3


def test_tie_breaks_by_enqueue_time():
    cfs = FairShareCFS()
    cfs.set_priority("D", 50)
    cfs.set_priority("E", 50)
    t0 = time.monotonic()
    cfs.on_run("D", 10.0)
    cfs.on_run("E", 10.0)
    pick = cfs.pick_next([("D", t0), ("E", t0 + 1)])
    assert pick == "D"

