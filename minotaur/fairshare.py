from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class _Entity:
    name: str
    weight: int  # >=1
    vtime: float = 0.0  # virtual runtime = service/weight
    service: float = 0.0  # cumulative wall-clock service


class FairShareCFS:
    """Completely Fair Scheduler (CFS)-style proportional-share scheduler.

    - Each agent `i` has a weight `w_i` derived from its priority (0..100 → w=max(1,priority)).
    - Virtual runtime v_i = service_i / w_i. The next agent is the ready one with minimal v_i
      (ties broken by earliest enqueue time supplied at pick time).
    - On run of Δ seconds: service_i += Δ; v_i += Δ / w_i.
    - On first arrival of a new agent, initialize v_i to the current minimum v among all known
      agents (so it neither leaps ahead nor starts infinitely behind).
    """

    def __init__(self, default_priority: int = 50) -> None:
        self._ents: Dict[str, _Entity] = {}
        self.default_weight = self._priority_to_weight(default_priority)

    @staticmethod
    def _priority_to_weight(priority: int) -> int:
        try:
            p = int(priority)
        except Exception:
            p = 0
        return max(1, p)

    def set_priority(self, name: str, priority: int) -> None:
        e = self._ents.get(name)
        w = self._priority_to_weight(priority)
        if e is None:
            self._ents[name] = _Entity(name=name, weight=w, vtime=self._min_v(), service=0.0)
        else:
            e.weight = w

    def _min_v(self) -> float:
        if not self._ents:
            return 0.0
        return min(e.vtime for e in self._ents.values())

    def on_arrival(self, name: str, priority: Optional[int] = None) -> None:
        """Notify scheduler that an agent became ready (first time or again).

        If the agent is new, initialize with vtime = min_v among existing agents and set its weight
        from priority (or default).
        """
        if name in self._ents:
            # nothing else needed; the tie-breaking will rely on enqueue time supplied to pick
            return
        w = self._priority_to_weight(self.default_weight if priority is None else priority)
        self._ents[name] = _Entity(name=name, weight=w, vtime=self._min_v(), service=0.0)

    def pick_next(self, ready: Iterable[Tuple[str, float]]) -> Optional[str]:
        """Pick next agent among ready names.

        ready: iterable of (name, enqueue_time) where enqueue_time is a monotonic or wall-clock
               timestamp used only for tie-breaking among equal vtime.
        Returns the chosen agent name or None if no ready agent is found.
        """
        best: Optional[Tuple[float, float, str]] = None  # (vtime, enq, name)
        for name, enq in ready:
            e = self._ents.get(name)
            if e is None:
                # Implicit arrival with default weight
                self.on_arrival(name, None)
                e = self._ents[name]
            key = (e.vtime, float(enq), name)
            if best is None or key < best:
                best = key
        return best[2] if best else None

    def on_run(self, name: str, delta_seconds: float) -> None:
        if delta_seconds <= 0:
            return
        e = self._ents.get(name)
        if e is None:
            # Treat as default arrival
            self.on_arrival(name)
            e = self._ents[name]
        e.service += float(delta_seconds)
        e.vtime += float(delta_seconds) / float(max(1, e.weight))

    def snapshot(self) -> List[Tuple[str, int, float, float]]:
        """Return sorted snapshot: [(name, weight, service, vtime)] by name for tests/debug."""
        out = [(e.name, e.weight, e.service, e.vtime) for e in self._ents.values()]
        out.sort(key=lambda x: x[0])
        return out

