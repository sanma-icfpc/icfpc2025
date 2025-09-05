# -*- coding: utf-8 -*-
"""
ICFP Programming Contest 2025 / Ædificium map learner (Moore L* variant)
- Batches all /explore queries to minimize queryCount.
- Builds an equivalent Moore machine (rooms, doors 0..5) and emits /guess JSON.

Usage:
  1) Set ICFP_ID in env if you already registered, or call client.register(...)
  2) Run main() to solve "probatio", or change PROBLEM_NAME.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set
import os
import json
import random
import requests
import sys

BASE_URL = "https://31pwr5t6ij.execute-api.eu-west-2.amazonaws.com"
ALPHABET = "012345"  # door labels

# ---------- API Client ----------

class AedificiumClient:
    def __init__(self, base_url: str = BASE_URL, team_id: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.id = team_id

    def _post(self, path: str, payload: Dict) -> Dict:
        url = f"{self.base_url}{path}"
        r = requests.post(url, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()

    def register(self, name: str, pl: str, email: str) -> str:
        data = {"name": name, "pl": pl, "email": email}
        j = self._post("/register", data)
        self.id = j["id"]
        return self.id

    def select_problem(self, problem_name: str) -> str:
        assert self.id, "team id is required. call register() or set client.id"
        j = self._post("/select", {"id": self.id, "problemName": problem_name})
        return j["problemName"]

    def explore(self, plans: List[str]) -> Tuple[List[List[int]], int]:
        assert self.id, "team id is required"
        j = self._post("/explore", {"id": self.id, "plans": plans})
        return j["results"], j["queryCount"]

    def guess(self, guess_map: Dict) -> bool:
        assert self.id, "team id is required"
        j = self._post("/guess", {"id": self.id, "map": guess_map})
        return bool(j.get("correct", False))


# ---------- Membership Oracle with batching & caching ----------

class ExploreOracle:
    """
    Caches traces for words; batches pending queries into a single /explore.
    For a plan w of length x, server returns a list of length x+1; we take the last
    element as g(w) (Moore state's label after executing w). The initial state's
    label is the 0-th element of any returned trace.
    """
    def __init__(self, client: AedificiumClient):
        self.client = client
        self.trace_cache: Dict[str, List[int]] = {}    # word -> full trace (len = |w|+1)
        self.last_label: Dict[str, int] = {}           # word -> final label
        self.pending: List[str] = []
        self.init_label: Optional[int] = None

    def ensure(self, words: Set[str]) -> None:
        for w in words:
            if w == "":
                # don't push empty; derive init label from any non-empty query
                continue
            if w not in self.trace_cache and w not in self.pending:
                self.pending.append(w)

    def commit(self) -> None:
        if not self.pending:
            return
        plans = list(dict.fromkeys(self.pending))  # dedupe, keep order
        self.pending.clear()
        results, _qc = self.client.explore(plans)
        print(f"{_qc=}")
        assert len(results) == len(plans)
        for w, trace in zip(plans, results):
            # trace length must be len(w)+1
            self.trace_cache[w] = trace
            self.last_label[w] = trace[-1]
            if self.init_label is None and len(trace) >= 1:
                self.init_label = trace[0]

    def g(self, w: str) -> int:
        """Return label of state after executing w. (Moore output)"""
        if w == "":
            # initial label
            if self.init_label is not None:
                return self.init_label
            # trigger a tiny query to get it
            self.ensure({"0"})
            self.commit()
            assert self.init_label is not None, "failed to acquire initial label"
            return self.init_label
        if w not in self.last_label:
            self.ensure({w})
            self.commit()
        return self.last_label[w]

    def ensure_rows(self, S: List[str], E: List[str]) -> None:
        need: Set[str] = set()
        for u in S:
            for e in E:
                need.add(u + e)
        for u in S:
            for a in ALPHABET:
                ua = u + a
                for e in E:
                    need.add(ua + e)
        self.ensure(need)
        self.commit()

    def predict_trace_end(self, delta, rep_for_state, start_rep: str, word: str, outputs) -> int:
        """
        Using hypothesis (delta, outputs), compute predicted final label for a word.
        - rep_for_state: representative word for each state index
        - outputs[state]: state's Moore output label
        """
        # Simulate from start state 0
        s = 0
        for ch in word:
            s = delta[s][int(ch)]
        return outputs[s]


# ---------- L* for Moore machines ----------

@dataclass
class Hypothesis:
    rows: Dict[str, Tuple[int, ...]]        # row signature for reps in S
    state_index_of_row: Dict[Tuple[int, ...], int]
    rep_for_state: List[str]                # representative word for each state
    outputs: List[int]                      # Moore output per state
    delta: List[List[int]]                  # transitions [state][door] -> state

class LStarMooreLearner:
    def __init__(self, oracle: ExploreOracle, max_loops: int = 200):
        self.oracle = oracle
        self.max_loops = max_loops
        self.S: List[str] = [""]            # access prefixes
        self.E: List[str] = [""]            # distinguishing suffixes (must contain "")
        # Pre-warm a tiny query to get initial label (and server wake-up)
        self.oracle.ensure({"0"})
        self.oracle.commit()

    def row(self, u: str) -> Tuple[int, ...]:
        return tuple(self.oracle.g(u + e) for e in self.E)

    def build_table(self):
        # ensure cache has all values for current S, E
        self.oracle.ensure_rows(self.S, self.E)

    def is_closed(self, rowS: Dict[str, Tuple[int, ...]]) -> Optional[str]:
        rows_set = {rowS[s] for s in self.S}
        for u in list(self.S):
            for a in ALPHABET:
                ua = u + a
                r = self.row(ua)
                if r not in rows_set:
                    return ua
        return None

    def find_inconsistency(self, rowS: Dict[str, Tuple[int, ...]]) -> Optional[Tuple[str, str, str]]:
        # Return (s1, s2, a) witnessing that row(s1)==row(s2) but row(s1a)!=row(s2a)
        buckets: Dict[Tuple[int, ...], List[str]] = {}
        for s in self.S:
            buckets.setdefault(rowS[s], []).append(s)
        for sig, group in buckets.items():
            if len(group) <= 1:
                continue
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    s1, s2 = group[i], group[j]
                    for a in ALPHABET:
                        r1 = self.row(s1 + a)
                        r2 = self.row(s2 + a)
                        if r1 != r2:
                            return (s1, s2, a)
        return None

    def make_hypothesis(self) -> Hypothesis:
        # Build state set as distinct rows over S
        rows_map: Dict[str, Tuple[int, ...]] = {s: self.row(s) for s in self.S}
        uniq_rows: List[Tuple[int, ...]] = []
        rep_for_state: List[str] = []
        idx_of_row: Dict[Tuple[int, ...], int] = {}
        for s in self.S:
            r = rows_map[s]
            if r not in idx_of_row:
                idx = len(uniq_rows)
                idx_of_row[r] = idx
                uniq_rows.append(r)
                rep_for_state.append(s)

        # outputs: label at empty suffix (E[0] == "")
        outputs = [uniq_rows[i][0] for i in range(len(uniq_rows))]

        # delta: state x door -> state
        delta: List[List[int]] = [[0] * 6 for _ in range(len(uniq_rows))]
        for i, rep in enumerate(rep_for_state):
            for d in range(6):
                r_next = self.row(rep + str(d))
                j = idx_of_row[r_next]
                delta[i][d] = j

        return Hypothesis(rows=rows_map,
                          state_index_of_row=idx_of_row,
                          rep_for_state=rep_for_state,
                          outputs=outputs,
                          delta=delta)

    def strengthen_with_counterexample(self, hyp: Hypothesis, max_len: int = 8, trials: int = 200) -> bool:
        """
        Look for a short word where hypothesis' predicted final label
        disagrees with the real oracle. If found, add all prefixes to S.
        Returns True iff strengthened.
        """
        # Build some test words (systematic + random)
        tests: Set[str] = set()
        # small BFS up to length 4
        frontier = [""]
        for _ in range(4):
            new_frontier = []
            for u in frontier:
                for a in ALPHABET:
                    w = u + a
                    tests.add(w)
                    new_frontier.append(w)
            frontier = new_frontier
        # random longer
        for _ in range(trials):
            L = random.randint(1, max_len)
            w = "".join(random.choice(ALPHABET) for _ in range(L))
            tests.add(w)

        # ensure queries
        self.oracle.ensure(set(tests))
        self.oracle.commit()

        for w in tests:
            pred = self.oracle.predict_trace_end(hyp.delta, hyp.rep_for_state, hyp.rep_for_state[0], w, hyp.outputs)
            real = self.oracle.g(w)
            if pred != real:
                # refine: add all prefixes of w to S
                for k in range(1, len(w) + 1):
                    p = w[:k]
                    if p not in self.S:
                        self.S.append(p)
                return True
        return False

    def learn(self) -> Hypothesis:
        loops = 0
        while True:
            loops += 1
            if loops > self.max_loops:
                raise RuntimeError("L* did not converge within loop budget")
            self.build_table()
            rowS = {s: self.row(s) for s in self.S}

            # Closure
            need = self.is_closed(rowS)
            if need is not None:
                if need not in self.S:
                    self.S.append(need)
                continue

            # Consistency
            inc = self.find_inconsistency(rowS)
            if inc is not None:
                s1, s2, a = inc
                # Find distinguishing e in E where row(s1a)[j] != row(s2a)[j]
                r1, r2 = self.row(s1 + a), self.row(s2 + a)
                for j, (x, y) in enumerate(zip(r1, r2)):
                    if x != y:
                        new_e = a + self.E[j]
                        if new_e not in self.E:
                            self.E.append(new_e)
                        break
                continue

            # Closed & consistent => build hypothesis
            hyp = self.make_hypothesis()

            # Try to find counterexample and strengthen S; if none, we are done
            strengthened = self.strengthen_with_counterexample(hyp)
            if strengthened:
                continue
            return hyp


# ---------- Build /guess JSON from hypothesis ----------

def build_guess_from_hypothesis(hyp: Hypothesis) -> Dict:
    n = len(hyp.outputs)
    rooms = hyp.outputs[:]  # 2-bit labels per room
    starting_room = hyp.state_index_of_row[hyp.rows[""]]

    # Pair (room,door) <-> (room,door) as undirected connections
    used = [[False] * 6 for _ in range(n)]
    connections = []

    for r in range(n):
        for a in range(6):
            if used[r][a]:
                continue
            r2 = hyp.delta[r][a]
            # find a partner door b in r2 that brings us back to r
            partner = None
            for b in range(6):
                if not used[r2][b] and hyp.delta[r2][b] == r:
                    partner = b
                    break
            # As a safety net (shouldn't be needed if delta is an involution on (room,door)):
            if partner is None:
                # allow pairing with already-used but matching door (won't double-add)
                for b in range(6):
                    if hyp.delta[r2][b] == r:
                        partner = b
                        break
            if partner is None:
                # last resort: self-pair
                partner = a if r == r2 else 0

            connections.append({
                "from": {"room": r, "door": a},
                "to":   {"room": r2, "door": partner}
            })
            used[r][a] = True
            used[r2][partner] = True

    return {
        "rooms": rooms,
        "startingRoom": starting_room,
        "connections": connections
    }


# ---------- Example runner ----------

def solve(problem_name: str):
    print(f"{problem_name=}")
    # Get ID from env or register once (keep the returned id secret!)
    team_id = os.getenv("ICFP_ID")
    client = AedificiumClient(team_id=team_id)

    if client.id is None:
        # One-time registration (then set ICFP_ID env for later runs)
        # !!! Replace with your team's information
        # client.register(name="Your Team", pl="Python", email="you@example.com")
        # print("Registered. Your secret id:", client.id)
        sys.exit(1)

    # Select a problem (try the tiny 'probatio' first)
    chosen = client.select_problem(problem_name)
    print("Selected:", chosen)

    oracle = ExploreOracle(client)
    learner = LStarMooreLearner(oracle)

    hyp = learner.learn()
    guess_map = build_guess_from_hypothesis(hyp)

    # Optionally pretty print the guess before submitting
    # print(json.dumps(guess_map, indent=2))

    ok = client.guess(guess_map)
    print("Guess correct?", ok)
    print()


if __name__ == "__main__":
    problem_names = ["probatio", "primus", "secundus", "tertius", "quartus", "quintus"]
    while True:
        for problem_name in problem_names:
            # For safety in an example script; remove in production
            try:
                solve(problem_name)
            except requests.HTTPError as e:
                print("HTTP error:", e)
            except Exception as ex:
                print("Error:", ex)
