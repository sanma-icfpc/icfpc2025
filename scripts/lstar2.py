# -*- coding: utf-8 -*-
"""
ICFP 2025 Aedificium learner with charcoal marking (post-LR addendum)
- Active learning for a Moore machine (6 doors) using L* observation table
- Adds "marking splits": injects in-plan label rewrites s[c]w to separate indistinguishable copies
- Batches /explore queries; respects per-plan door-step cap (default 6*n after addendum)

Refs:
- API and plan format (digits 0-5; records are x+1 labels): Task PDF v1.2
- Marking '[c]', and 6n door-step cap per plan: Addendum PDF
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set, Iterable
import os
import json
import random
import requests
import datetime
import traceback


# BASE_URL = "https://31pwr5t6ij.execute-api.eu-west-2.amazonaws.com"
BASE_URL = os.getenv("BASE_URL", "http://tk2-401-41624.vs.sakura.ne.jp:19384/")
DOORS = "012345"  # door labels as characters
AGENT_NAME = os.getenv("AGENT_NAME", "nodchip:lstar2")
AGENT_ID = str(os.getpid())
TIMEOUT_SEC = 6000
PROGRESS_DURATION_SEC = float(os.getenv("PROGRESS_DURATION_SEC", -1.0))

print(f"{BASE_URL=}")
print(f"{AGENT_NAME=}")
print(f"{AGENT_ID=}")
print(f"{PROGRESS_DURATION_SEC=}")

# ==== API client ==== #


class AedificiumClient:
    def __init__(
        self,
        base_url: str = BASE_URL,
        team_id: Optional[str] = None,
    ):
        self.base_url = base_url.rstrip("/")
        print(f"{self.base_url=}", flush=True)
        self.id = team_id

    def _post(self, path: str, payload: Dict) -> Dict:
        headers = {
            "X-Agent-Name": AGENT_NAME,
            "X-Agent-ID": AGENT_ID,
        }
        r = requests.post(
            f"{self.base_url}{path}", json=payload, timeout=TIMEOUT_SEC, headers=headers
        )
        r.raise_for_status()
        return r.json()

    def register(self, name: str, pl: str, email: str) -> str:
        j = self._post("/register", {"name": name, "pl": pl, "email": email})
        self.id = j["id"]
        return self.id

    def select_problem(self, problem_name: str) -> str:
        assert self.id, "need id (call register or set ICFP_ID)"
        j = self._post("/select", {"id": self.id, "problemName": problem_name})
        return j["problemName"]

    def explore(self, plans: List[str]) -> Tuple[List[List[int]], int]:
        """plans: list of plan strings. Brackets like '[2]' are allowed per addendum."""
        assert self.id, "need id"
        j = self._post("/explore", {"id": self.id, "plans": plans})
        return j["results"], j["queryCount"]

    def guess(self, guess_map: Dict) -> bool:
        assert self.id, "need id"
        j = self._post("/guess", {"id": self.id, "map": guess_map})
        return bool(j.get("correct", False))


# ==== Oracle with batching, caching, and per-plan length budgeting ==== #


def door_steps(plan: str) -> int:
    """Count door-steps (digits 0..5); '[c]' does not count."""
    return sum(ch in DOORS for ch in plan)


class ExploreOracle:
    """
    Cache membership: plan -> full trace (list[int], length = door_steps(plan)+1)
    'g(w)' returns final label after executing plan w (digits + optional [c] markings).
    """

    def __init__(self, client: AedificiumClient, max_steps_per_plan: int):
        self.client = client
        self.max_steps = max_steps_per_plan
        self.trace_cache: Dict[str, List[int]] = {}
        self.last_label: Dict[str, int] = {}
        self.pending: List[str] = []
        self.init_label: Optional[int] = None

        self.last_qc_output_datetime = datetime.datetime.now()

    def enqueue(self, plan: str):
        if plan and plan not in self.trace_cache and plan not in self.pending:
            # if door_steps(plan) > self.max_steps:
            #     raise ValueError(f"plan exceeds door-step cap {self.max_steps}: {plan}")
            self.pending.append(plan)

    def ensure_all(self, plans: Iterable[str]):
        for p in plans:
            self.enqueue(p)

    def commit(self):
        if not self.pending:
            return
        # dedupe while preserving order
        seen = set()
        batch = []
        for p in self.pending:
            if p not in seen:
                seen.add(p)
                batch.append(p)
        self.pending.clear()
        results, _qc = self.client.explore(batch)
        if (
            self.last_qc_output_datetime
            + datetime.timedelta(seconds=PROGRESS_DURATION_SEC)
            < datetime.datetime.now()
        ):
            self.last_qc_output_datetime = datetime.datetime.now()
            print(f"{_qc=}", flush=True)
        if len(results) != len(batch):
            raise RuntimeError("API returned results of different length")
        for p, trace in zip(batch, results):
            self.trace_cache[p] = trace
            self.last_label[p] = trace[-1]
            if self.init_label is None and trace:
                self.init_label = trace[0]

    def g(self, plan: str) -> int:
        """Final-state label after plan."""
        if plan == "":
            if self.init_label is None:
                # poke server minimally to get initial label; use a 1-step plan
                self.ensure_all(["0"])
                self.commit()
            assert self.init_label is not None
            return self.init_label
        if plan not in self.last_label:
            self.enqueue(plan)
            self.commit()
        return self.last_label[plan]


# ==== Utilities to compose plans ==== #


def concat(u: str, v: str) -> str:
    """String concat with trivial fast path for empty."""
    if not u:
        return v
    if not v:
        return u
    return u + v


def mark(c: int) -> str:
    """Return bracket token like '[2]'."""
    if c not in (0, 1, 2, 3):
        raise ValueError("mark must be 0..3")
    return f"[{c}]"


# ==== L* for Moore with optional marking-based splitting ==== #


@dataclass
class Hypothesis:
    row_of: Dict[str, Tuple[int, ...]]  # signature for S reps
    idx_of_row: Dict[Tuple[int, ...], int]  # row -> state index
    rep: List[str]  # representative access plan per state
    outputs: List[int]  # Moore output per state
    delta: List[List[int]]  # next state index by door


class LStarMooreWithMarks:
    """
    S: access plans (digits only)
    E: distinguishing suffixes (digits only, for the base table)
    For splitting indistinguishable copies, we synthesize plans s + [c] + w on-demand.
    """

    def __init__(self, oracle: ExploreOracle, rnd: random.Random = random.Random(0)):
        self.o = oracle
        self.S: List[str] = [""]  # access prefixes (digits only)
        self.E: List[str] = [""]  # suffixes (digits only)
        self.rnd = rnd
        # tiny warm-up to get init label
        self.o.ensure_all(["0"])
        self.o.commit()

        self.last_dfs_output_datetime = datetime.datetime.now()

    # --- observation table primitives --- #
    def row(self, u: str) -> Tuple[int, ...]:
        return tuple(self.o.g(concat(u, e)) for e in self.E)

    def ensure_table(self):
        needed: Set[str] = set()
        for u in self.S:
            for e in self.E:
                needed.add(concat(u, e))
        for u in self.S:
            for a in DOORS:
                ua = u + a
                for e in self.E:
                    needed.add(concat(ua, e))
        self.o.ensure_all(needed)
        self.o.commit()

    def is_closed(self, rows: Dict[str, Tuple[int, ...]]) -> Optional[str]:
        R = {rows[s] for s in self.S}
        for s in list(self.S):
            for a in DOORS:
                ua = s + a
                if self.row(ua) not in R:
                    return ua
        return None

    def inconsistency(
        self, rows: Dict[str, Tuple[int, ...]]
    ) -> Optional[Tuple[str, str, str]]:
        buckets: Dict[Tuple[int, ...], List[str]] = {}
        for s in self.S:
            buckets.setdefault(rows[s], []).append(s)
        for sig, group in buckets.items():
            if len(group) <= 1:
                continue
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    s1, s2 = group[i], group[j]
                    for a in DOORS:
                        if self.row(s1 + a) != self.row(s2 + a):
                            return (s1, s2, a)
        return None

    def build_hypothesis(self) -> Hypothesis:
        rows = {s: self.row(s) for s in self.S}
        idx_of_row: Dict[Tuple[int, ...], int] = {}
        rep: List[str] = []
        for s in self.S:
            r = rows[s]
            if r not in idx_of_row:
                idx_of_row[r] = len(rep)
                rep.append(s)
        outputs = [tuple_row[0] for tuple_row in idx_of_row.keys()]
        # transitions
        k = len(rep)
        delta = [[0] * 6 for _ in range(k)]
        for i, s in enumerate(rep):
            for d in range(6):
                r_next = self.row(s + str(d))
                j = idx_of_row[r_next]
                delta[i][d] = j
        return Hypothesis(
            row_of=rows, idx_of_row=idx_of_row, rep=rep, outputs=outputs, delta=delta
        )

    # --- equivalence testing (cheap, batched) --- #
    def find_counterexample(
        self, hyp: Hypothesis, max_len: int = 8, trials: int = 300
    ) -> Optional[str]:
        tests: Set[str] = set()
        # small systematic suite
        frontier = [""]
        for _ in range(3):
            nxt = []
            for u in frontier:
                for a in DOORS:
                    w = u + a
                    tests.add(w)
                    nxt.append(w)
            frontier = nxt
        # random walk tests
        for _ in range(trials):
            L = self.rnd.randint(1, max_len)
            w = "".join(self.rnd.choice(DOORS) for _ in range(L))
            tests.add(w)

        self.o.ensure_all(tests)
        self.o.commit()

        # simulate on hypothesis
        def predict(word: str) -> int:
            s = 0
            for ch in word:
                s = hyp.delta[s][int(ch)]
            return hyp.outputs[s]

        for w in tests:
            if predict(w) != self.o.g(w):
                return w
        return None

    def learn(
        self, marking_colors: List[int], W_suffixes: List[str], max_iters: int = 200
    ) -> Hypothesis:
        it = 0
        while True:
            it += 1
            if it > max_iters:
                raise RuntimeError("did not converge")
            # build/repair the base observation table (no marks)
            self.ensure_table()
            rows = {s: self.row(s) for s in self.S}
            need = self.is_closed(rows)
            if need is not None:
                if need not in self.S:
                    self.S.append(need)
                continue
            inc = self.inconsistency(rows)
            if inc is not None:
                s1, s2, a = inc
                r1, r2 = self.row(s1 + a), self.row(s2 + a)
                # add shortest distinguishing suffix a + e_j
                for j, (x, y) in enumerate(zip(r1, r2)):
                    if x != y:
                        new_e = a + self.E[j]
                        if new_e not in self.E:
                            self.E.append(new_e)
                        break
                continue

            # closed & consistent -> make hypothesis
            hyp = self.build_hypothesis()

            # 2) cheap equivalence check with random/systematic tests (no marks)
            ce = self.find_counterexample(hyp)
            if ce is not None:
                # add all prefixes of ce to S (TTT-like shorter refinement)
                for k in range(1, len(ce) + 1):
                    p = ce[:k]
                    if p not in self.S:
                        self.S.append(p)
                continue

            # done
            return hyp

    def _filled(self, num_rooms, num_copies, graph):
        used = [[False] * 6 for _ in range(num_rooms)]
        for room in range(num_rooms):
            for door in range(6):
                if used[room][door]:
                    continue
                room_group = room // num_copies
                copy_index = room % num_copies

                key = (room_group, copy_index, door)
                if key not in graph:
                    continue

                other_room_group, other_copy_index, other_door = graph[key]
                other_room = other_room_group * num_copies + other_copy_index

                used[room][door] = True
                used[other_room][other_door] = True

        num_total = 0
        num_used = 0
        for room in range(num_rooms):
            for door in range(6):
                num_total += 1
                if used[room][door]:
                    num_used += 1
        return num_used, num_total

    # DFSでコピーされた部屋を含めた地図を復元する。
    def dfs(
        self,
        quotient_graph: List[
            List[Tuple[int, int]]
        ],  # 商グラフ [src部屋グループ番号][src扉番号] -> (dst部屋グループ番号, dst扉番号)
        graph: Dict[
            Tuple[int, int, int], Tuple[int, int, int]
        ],  # 構築中のグラフ (src部屋グループ番号, src部屋のコピー番号, src扉番号) -> (dst部屋グループ番号, dst部屋のコピー番号, dst扉番号)
        edges: Set[
            Tuple[Tuple[int, int, int], Tuple[int, int, int]]
        ],  # これまでに見つけたエッジ
        prev_room_group: int,  # 直前の部屋グループ番号
        prev_copy_index: int,  # 直前の部屋のコピー番号
        prev_next_door: int,  # 直前の部屋の扉番号
        current_room_group: int,  # 現在の部屋グループ番号
        current_last_door: int,  # 現在の部屋に入ってきたときの扉番号
        past_plan: List[str],  # "[(新しい色)](扉番号)"のリスト
        original_states: List[int],  # 元のラベル
        seen_count: List[int],  # 各部屋グループ内で見つけたコピーの数
        path: List[
            Tuple[int, int, int]
        ],  # これまで辿ってきたパス (部屋グループ番号, 部屋のコピー番号, 次に開けるドア番号)
        num_rooms: int,  # 部屋数
        num_copies: int,  # コピーの数
    ):
        num_used, num_total = self._filled(num_rooms, num_copies, graph)
        if (
            self.last_dfs_output_datetime
            + datetime.timedelta(seconds=PROGRESS_DURATION_SEC)
            < datetime.datetime.now()
        ):
            self.last_dfs_output_datetime = datetime.datetime.now()
            print(f"num_used/num_total={num_used}/{num_total} {path=}", flush=True)
        if num_used == num_total:
            # すべての扉をたどったら抜ける。
            return

        plan = "".join(past_plan)
        current_state = self.o.g(plan)
        new_state = None
        current_copy_index = None
        if current_state == original_states[current_room_group]:
            current_copy_index = seen_count[current_room_group]
            new_state = (
                original_states[current_room_group]
                + (seen_count[current_room_group] + 1)
            ) % 4
            seen_count[current_room_group] += 1
            assert seen_count[current_room_group] <= num_copies
        else:
            idx = (current_state - original_states[current_room_group] - 1 + 4) % 4
            assert idx < num_copies
            current_copy_index = idx

        if prev_room_group is not None:
            u = (prev_room_group, prev_copy_index, prev_next_door)
            v = (current_room_group, current_copy_index, current_last_door)
            if (u, v) in edges:
                return
            edges.add((u, v))
            edges.add((v, u))

            graph[u] = v
            graph[v] = u

        if (
            prev_room_group == current_room_group
            and prev_copy_index == current_copy_index
        ):
            # セルフループの場合、planを短くするために抜ける。
            return

        prefetch = list()
        for next_door in range(6):
            if new_state is not None:
                past_plan.append(f"[{new_state}]{next_door}")
            else:
                past_plan.append(f"{next_door}")

            prefetch.append("".join(past_plan))

            past_plan.pop()
        self.o.ensure_all(prefetch)
        self.o.commit()

        for next_door in range(6):
            if new_state is not None:
                past_plan.append(f"[{new_state}]{next_door}")
            else:
                past_plan.append(f"{next_door}")
            path.append((current_room_group, current_copy_index, next_door))

            next_room_group, next_last_door = quotient_graph[current_room_group][
                next_door
            ]
            self.dfs(
                quotient_graph,
                graph,
                edges,
                current_room_group,
                current_copy_index,
                next_door,
                next_room_group,
                next_last_door,
                past_plan,
                original_states,
                seen_count,
                path,
                num_rooms,
                num_copies,
            )

            path.pop()
            past_plan.pop()

    def reconstruct(self, hyp: Hypothesis, num_rooms: int):
        num_room_groups = len(hyp.outputs)
        # [src部屋グループ番号, src扉番号] -> (dst部屋グループ番号, dst扉番号)
        quotient_graph = [[(-1, -1)] * 6 for _ in range(num_room_groups)]

        used = [[False] * 6 for _ in range(num_room_groups)]
        for room in range(num_room_groups):
            for door in range(6):
                if used[room][door]:
                    continue
                r2 = hyp.delta[room][door]
                # find partner door in r2 that returns to r
                partner = None
                for b in range(6):
                    if not used[r2][b] and hyp.delta[r2][b] == room:
                        partner = b
                        break
                if partner is None:
                    # allow already-used match as fallback
                    for b in range(6):
                        if hyp.delta[r2][b] == room:
                            partner = b
                            break
                if partner is None:
                    partner = door if room == r2 else 0
                quotient_graph[room][door] = (r2, partner)
                quotient_graph[r2][partner] = (room, door)
                used[room][door] = True
                used[r2][partner] = True

        graph = dict()
        edges = set()
        past_plan = list()
        original_states = hyp.outputs[:]
        seen_count: List[int] = [0] * num_room_groups
        path = list()
        num_copies = num_rooms // num_room_groups
        self.dfs(
            quotient_graph,
            graph,
            edges,
            None,
            None,
            None,
            0,
            None,
            past_plan,
            original_states,
            seen_count,
            path,
            num_rooms,
            num_copies,
        )

        rooms = []
        for output in hyp.outputs:
            rooms.extend([output] * num_copies)

        starting_room_group = hyp.idx_of_row[hyp.row_of[""]]
        starting_room = starting_room_group * num_copies

        used = [[False] * 6 for _ in range(num_rooms)]
        connections = []
        for room in range(num_rooms):
            for door in range(6):
                if used[room][door]:
                    continue
                room_group = room // num_copies
                copy_index = room % num_copies

                other_room_group, other_copy_index, other_door = graph[
                    (room_group, copy_index, door)
                ]
                other_room = other_room_group * num_copies + other_copy_index

                connections.append(
                    {
                        "from": {"room": room, "door": door},
                        "to": {"room": other_room, "door": other_door},
                    }
                )

                used[room][door] = True
                used[other_room][other_door] = True

        return {
            "rooms": rooms,
            "startingRoom": starting_room,
            "connections": connections,
        }


# ==== Build /guess payload ==== #


def build_guess(hyp: Hypothesis) -> Dict:
    n = len(hyp.outputs)
    rooms = hyp.outputs[:]  # 2-bit labels
    starting_room = hyp.idx_of_row[hyp.row_of[""]]

    used = [[False] * 6 for _ in range(n)]
    connections = []
    for r in range(n):
        for a in range(6):
            if used[r][a]:
                continue
            r2 = hyp.delta[r][a]
            # find partner door in r2 that returns to r
            partner = None
            for b in range(6):
                if not used[r2][b] and hyp.delta[r2][b] == r:
                    partner = b
                    break
            if partner is None:
                # allow already-used match as fallback
                for b in range(6):
                    if hyp.delta[r2][b] == r:
                        partner = b
                        break
            if partner is None:
                partner = a if r == r2 else 0
            connections.append(
                {
                    "from": {"room": r, "door": a},
                    "to": {"room": r2, "door": partner},
                }
            )
            used[r][a] = True
            used[r2][partner] = True

    return {"rooms": rooms, "startingRoom": starting_room, "connections": connections}


# ==== Example runner ==== #


def main(problem_name: str, n_rooms: int):
    # ---- Configuration ----
    MAX_DOOR_STEPS_PER_PLAN = 6 * n_rooms  # per addendum
    team_id = os.getenv("ICFP_ID")  # set your secret id

    client = AedificiumClient(team_id=team_id)
    if client.id is None:
        # One-time registration (then export ICFP_ID for subsequent runs)
        client.register(name="Your Team", pl="Python", email="you@example.com")
        print("Registered; set ICFP_ID to:", client.id)

    chosen = client.select_problem(problem_name)
    print("Selected:", chosen, flush=True)

    oracle = ExploreOracle(client, max_steps_per_plan=MAX_DOOR_STEPS_PER_PLAN)
    learner = LStarMooreWithMarks(oracle)

    # lightweight W set: short suffix suite plus a few randoms
    W_suffixes = ["", "0", "1", "2", "3", "4", "5", "01", "23", "45", "012", "345"]
    rnd = random.Random(42)
    for _ in range(40):
        L = rnd.randint(1, 4)
        W_suffixes.append("".join(rnd.choice(DOORS) for _ in range(L)))
    # colors for splitting: 0/1 (aleph想定). 3重なら [2] も使う
    colors = [0, 1, 2]

    hyp = learner.learn(marking_colors=colors, W_suffixes=W_suffixes, max_iters=300)

    guess_map = learner.reconstruct(hyp, n_rooms)

    print(json.dumps(guess_map), flush=True)
    ok = client.guess(guess_map)
    print("Guess correct?", ok, flush=True)


if __name__ == "__main__":
    problems = [
        # ("probatio", 3),
        # ("primus", 6),
        # ("secundus", 12),
        # ("tertius", 18),
        # ("quartus", 24),
        # ("quintus", 30),
        # ("aleph", 12),
        # ("beth", 24),
        # ("gimel", 36),
        # ("daleth", 48),
        # ("he", 60),
        # ("vau", 18),
        ("zain", 36),
        ("hhet", 54),
        # ("teth", 72),
        # ("iod", 90),
    ]
    while True:
        for problem_name, n_rooms in problems:
            try:
                main(problem_name, n_rooms)
            except requests.HTTPError as e:
                print(
                    "HTTP error:",
                    e.response.text if hasattr(e, "response") else str(e),
                    flush=True,
                )
            except Exception as ex:
                print("Error:", traceback.format_exception(ex), flush=True)
