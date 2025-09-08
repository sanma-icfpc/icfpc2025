from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Set, Iterable
from ortools.sat.python import cp_model
import os
import sys
import json
import time
import argparse
import requests
import subprocess
import threading
import random
import datetime
import traceback

ALPHABET = "012345"  # door labels
CATALOG = {
    "probatio": (3, 1),
    "primus": (6, 1),
    "secundus": (12, 1),
    "tertius": (18, 1),
    "quartus": (24, 1),
    "quintus": (30, 1),
    "aleph": (12, 2),
    "beth": (24, 2),
    "gimel": (36, 2),
    "daleth": (48, 2),
    "he": (60, 2),
    "vau": (18, 3),
    "zain": (36, 3),
    "hhet": (54, 3),
    "teth": (72, 3),
    "iod": (90, 3),
}
PROGRESS_DURATION_SEC = float(os.getenv("PROGRESS_DURATION_SEC", -1.0))

# ---------- API Client ----------

class AedificiumClient:
    def __init__(self, base_url: str, team_id: str = 'ignored', *, agent_name: Optional[str] = None, agent_id: Optional[str] = None, timeout_sec: Optional[int] = None):
        self.base_url = base_url.rstrip("/")
        self.id = team_id
        # Headers (agent identification)
        self.agent_name = 'k3+nodchip:' + (agent_name or os.getenv('X_AGENT_NAME') or os.getenv('AGENT_NAME') or 'cpsat')
        self.agent_id = agent_id or os.getenv('X_AGENT_ID') or os.getenv('AGENT_ID') or str(os.getpid())
        # Long HTTP timeout since /select may block until granted
        self.timeout_sec = int(os.getenv('MINOTAUR_HTTP_TIMEOUT_SEC') or os.getenv('HTTP_TIMEOUT_SEC') or (timeout_sec if timeout_sec is not None else 6000))
        self._headers = {
            'Content-Type': 'application/json',
            'X-Agent-Name': self.agent_name,
            'X-Agent-ID': self.agent_id,
        }

    def _post(self, path: str, payload: Dict) -> Dict:
        url = f"{self.base_url}{path}"
        r = requests.post(url, json=payload, headers=self._headers, timeout=self.timeout_sec)
        r.raise_for_status()
        return r.json()

    def register(self, name: str, pl: str, email: str) -> str:
        data = {"name": name, "pl": pl, "email": email}
        j = self._post("/register", data)
        self.id = j["id"]
        return self.id

    def select_problem(self, problem_name: str) -> str:
        assert self.id, "team id is required. call register() or set client.id"
        j = self._post("/select", {"id": self.id, "problemName": problem_name, "seed":random.randint(0, sys.maxsize)})
        return j["problemName"]

    def explore(self, plans: List[str]) -> Tuple[List[List[int]], int]:
        assert self.id, "team id is required"
        j = self._post("/explore", {"id": self.id, "plans": plans})
        return j["results"], j["queryCount"]

    def guess(self, guess_map: Dict) -> bool:
        assert self.id, "team id is required"
        j = self._post("/guess", {"id": self.id, "map": guess_map})
        return bool(j.get("correct", False))


# -------- de Bruijn (alphabet=6) --------
def de_bruijn_seq(k: int, K: int = 6) -> List[int]:
    """de Bruijn sequence B(K, k) over alphabet [0..K-1], returned linearly (length K**k)."""
    a = [0] * (K * k)
    seq: List[int] = []

    def db(t: int, p: int):
        if t > k:
            if k % p == 0:
                seq.extend(a[1:p + 1])
        else:
            a[t] = a[t - p]
            db(t + 1, p)
            for j in range(a[t - p] + 1, K):
                a[t] = j
                db(t + 1, t)
    db(1, 1)
    return seq  # length K**k


def make_single_plan(n_rooms: int, num_copies: int) -> str:
    """スコア2向け：1本のクエリ列。できるだけ長く（<= 18n）かつ分散よく。
    n>=12 なら k=3 をフル(216) or 繰り返し詰め、n<12 は k=2 を繰り返し。
    """
    if num_copies == 1:
        max_len = 18 * n_rooms
    else:
        max_len = 6 * n_rooms
    if n_rooms >= 12:
        base = de_bruijn_seq(3)  # 216
    else:
        base = de_bruijn_seq(2)  # 36
    plan_digits: List[int] = []
    while len(plan_digits) + len(base) <= max_len:
        plan_digits.extend(base)
    # 端数を足してビット列の“窓”を壊さないよう、先頭から切出す
    remain = max_len - len(plan_digits)
    if remain > 0:
        plan_digits.extend(base[:remain])
    return "".join(str(d) for d in plan_digits)


# -------- CP-SAT 復元 --------
def solve_cpsat_guess_map(n_rooms: int, plan: str, observed: List[int], max_time_in_seconds: float, num_search_workers: int) -> Dict:
    """/explore の戻り (observed labels) に一致する地図を CP-SAT で復元し、GuessMap(JSON相当のdict) を返す。
    observed の長さは len(plan)+1（先頭は開始部屋のラベル）。
    """
    assert len(observed) == len(plan) + 1
    K = 6
    P = n_rooms * K  # port count

    model = cp_model.CpModel()

    # F[p] ∈ [0..P-1]: ポート対応（自己ループ可）/ involution: F[F[p]] = p
    F = [model.NewIntVar(0, P - 1, f"F[{p}]") for p in range(P)]
    # involution: Element(F, F[p]) == p
    for p in range(P):
        model.AddElement(F[p], F, p)

    # 状態列 S[t] ∈ [0..n-1], ドア成分 r_t ∈ [0..5]
    T = len(plan)
    S = [model.NewIntVar(0, n_rooms - 1, f"S[{t}]") for t in range(T + 1)]
    R = [model.NewIntVar(0, K - 1, f"R[{t}]") for t in range(T)]          # 到着側ドア
    model.Add(S[0] == 0)

    # ラベル配列 L[r] ∈ [0..3]
    L = [model.NewIntVar(0, 3, f"L[{r}]") for r in range(n_rooms)]

    # t=0..T: L[S[t]] == observed[t]
    for t in range(T + 1):
        lt = model.NewIntVar(0, 3, f"lt[{t}]")
        model.AddElement(S[t], L, lt)
        model.Add(lt == observed[t])

    # 遷移拘束: p_t = 6*S[t]+a[t]; q_t = F[p_t]; q_t = 6*S[t+1] + R[t]
    for t in range(T):
        a_t = int(plan[t])
        p_t = model.NewIntVar(0, P - 1, f"p[{t}]")
        q_t = model.NewIntVar(0, P - 1, f"q[{t}]")
        # p_t = 6*S[t] + a_t
        model.Add(p_t == S[t] * K + a_t)
        # q_t = F[p_t]
        model.AddElement(p_t, F, q_t)
        # q_t = 6*S[t+1] + R[t]
        model.Add(q_t == S[t + 1] * K + R[t])

    # 目的関数は不要（可行解で十分）
    solver = cp_model.CpSolver()
    # 早めに返すチューニング（必要に応じて調整）
    solver.parameters.max_time_in_seconds = max_time_in_seconds
    solver.parameters.num_search_workers = num_search_workers
    solver.parameters.cp_model_presolve = True

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"CP-SAT no solution: status={solver.StatusName(status)}")

    # 解の復元
    # labels
    labels = [solver.Value(L[r]) for r in range(n_rooms)]
    # to 配列（部屋×6）を Port(r',d') で構成
    def port_of(idx: int) -> Tuple[int, int]:
        return idx // K, idx % K

    connections = []
    seen_pair = set()
    for p in range(P):
        q = solver.Value(F[p])
        a, b = sorted((p, q))
        if (a, b) in seen_pair:
            continue
        seen_pair.add((a, b))
        r1, d1 = port_of(a)
        r2, d2 = port_of(b)
        connections.append({
            "from": {"room": r1, "door": d1},
            "to":   {"room": r2, "door": d2},
        })

    guess_map = {
        "rooms": labels,
        "startingRoom": 0,          # 開始部屋は 0 固定
        "connections": connections, # 片側だけ書けばOK（無向扱い）
    }
    return guess_map



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


class Reconstructor:
    def __init__(self, oracle: ExploreOracle):
        self.o = oracle
        self.last_dfs_output_datetime = datetime.datetime.now()

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

    def reconstruct(self, initial_guess, num_rooms: int, num_copies):
        num_room_groups = num_rooms // num_copies
        # [src部屋グループ番号, src扉番号] -> (dst部屋グループ番号, dst扉番号)
        quotient_graph = [[(-1, -1)] * 6 for _ in range(num_room_groups)]

        for connection in initial_guess["connections"]:
            from_room = connection["from"]["room"]
            from_door = connection["from"]["door"]
            to_room = connection["to"]["room"]
            to_door = connection["to"]["door"]
            quotient_graph[from_room][from_door] = (to_room, to_door)
            quotient_graph[to_room][to_door] = (from_room, from_door)

        graph = dict()
        edges = set()
        past_plan = list()
        original_states = initial_guess["rooms"][:]
        seen_count: List[int] = [0] * num_room_groups
        path = list()
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

        rooms = list()
        for output in initial_guess["rooms"]:
            rooms.extend([output] * num_copies)

        starting_room = initial_guess["startingRoom"] * num_copies

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


# ---------- Core loop ----------

def solve_once(client: AedificiumClient, *, problem_name: str, http_timeout: Optional[int], max_time_in_seconds: float, num_search_workers: int) -> bool:
    """1回の /select → C++ 対話 → /explore → /guess を実行。成功可否を返す。"""

    # /select
    actual_problem = client.select_problem(problem_name)
    print(f'problem_name: {actual_problem}', flush=True)

    assert problem_name in CATALOG
    num_rooms, num_copies = CATALOG[problem_name]

    plan = make_single_plan(num_rooms, num_copies)
    print(f'plan={plan}', flush=True)
    assert not any(ch not in ALPHABET for ch in plan)

    plans = [plan] # single plan
    
    results, qctr = client.explore(plans)
    print(f'results: {results}, qc: {qctr}', flush=True)

    observed = results[0]
    print(f'observed={observed}', flush=True)

    initial_guess = solve_cpsat_guess_map(num_rooms // num_copies, plan, observed, max_time_in_seconds, num_search_workers)
    print(f'initial_guess={initial_guess}', flush=True)

    if num_copies > 1:
        oracle = ExploreOracle(client, num_rooms * 6)
        reconstructor = Reconstructor(oracle)
        guess = reconstructor.reconstruct(initial_guess, num_rooms, num_copies)
        print(f'guess={guess}', flush=True)
    else:
        guess = initial_guess

    result = client.guess(guess)
    print(f'result={result}', flush=True)
    return result


def make_client(args: argparse.Namespace) -> AedificiumClient:
    # server プロファイル
    if args.server == "local":
        base_url = "http://127.0.0.1:8009"
    elif args.server == "proxy":
        base_url = "http://tk2-401-41624.vs.sakura.ne.jp:19384"
    elif args.server == "remote":
        base_url = "https://31pwr5t6ij.execute-api.eu-west-2.amazonaws.com"
    elif args.server == "custom":
        if not args.base_url:
            raise SystemExit("--server custom の場合は --base-url を指定してください")
        base_url = args.base_url
    else:
        raise SystemExit(f"Unknown --server: {args.server}")

    client = AedificiumClient(
        base_url=base_url,
        team_id=os.getenv('ICFP_ID') if args.server in ("proxy", "remote", "custom") else 'ignored',
        timeout_sec=args.timeout_sec,
        agent_name=args.agent_name,   # ← 追加
        agent_id=args.agent_id        # ← 追加
    )
    return client


def main():
    parser = argparse.ArgumentParser(description="ICFP 2025 Aedificium Python<->C++ bridge")
    parser.add_argument("--server", choices=["local", "proxy", "remote", "custom"], default="local",
                        help="server profile (default: local)")
    parser.add_argument("--base-url", type=str, default=None,
                        help="custom base URL when --server custom")
    parser.add_argument("--problem", type=str, default="primus",
                        help="problem name (e.g., probatio|primus|secundus|tertius|quartus|quintus)")
    parser.add_argument("--timeout-sec", type=int, default=None,
                        help="HTTP timeout seconds (overrides env)")
    parser.add_argument("--loop", type=int, default=1,
                        help="number of iterations; 0 means infinite (default: 1)")
    parser.add_argument("--sleep-sec", type=float, default=0.0,
                        help="sleep seconds between iterations (default: 0)")
    parser.add_argument("--agent-name", type=str, default=None,
                        help="override X-Agent-Name header; default uses env X_AGENT_NAME/AGENT_NAME or 'k3+nodchip:solver'")
    parser.add_argument("--agent-id", type=str, default=None,
                        help="override X-Agent-ID header; default uses env X_AGENT_ID/AGENT_ID or current PID")
    parser.add_argument("--max-time-in-seconds", type=float, default=10.0,
                        help="Max time in seconds.")
    parser.add_argument("--num-search-workers", type=int, default=8,
                        help="Num search workers.")
    args = parser.parse_args()

    client = make_client(args)

    start_datetime = datetime.datetime.now()

    # ループ実行
    i = 0
    succeeded = 0
    try:
        while True:
            i += 1
            print(f"\n=== iteration {i} / problem={args.problem} ===", flush=True)
            try:
                ok = solve_once(
                    client,
                    problem_name=args.problem,
                    http_timeout=args.timeout_sec,
                    max_time_in_seconds=args.max_time_in_seconds,
                    num_search_workers=args.num_search_workers,
                )
                if ok:
                    succeeded += 1
            except KeyboardInterrupt:
                print("\nInterrupted by user.", file=sys.stderr, flush=True)
                break
            except Exception as e:
                print(f"[ERROR] iteration {i}: {traceback.format_exception(e)}", file=sys.stderr, flush=True)
            if args.loop != 0 and i >= args.loop:
                break
            if args.sleep_sec > 0:
                time.sleep(args.sleep_sec)
    finally:
        # 共有プロセスを閉じる
        if hasattr(solve_once, "_shared_cpp") and getattr(solve_once, "_shared_cpp") is not None:
            try:
                getattr(solve_once, "_shared_cpp").close()
            except Exception:
                pass

    print(f"\nDone. success={succeeded} / total={i}", flush=True)

    end_datetime = datetime.datetime.now()
    print(end_datetime - start_datetime)


if __name__ == '__main__':
    main()