from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from ortools.sat.python import cp_model
import os
import sys
import json
import time
import argparse
import requests
import subprocess
import threading

ALPHABET = "012345"  # door labels
CATALOG = {
    "probatio": 3, "primus": 6, "secundus": 12, "tertius": 18, "quartus": 24, "quintus": 30,
}

# ---------- API Client ----------

class AedificiumClient:
    def __init__(self, base_url: str, team_id: str = 'ignored', *, agent_name: Optional[str] = None, agent_id: Optional[str] = None, timeout_sec: Optional[int] = None):
        self.base_url = base_url.rstrip("/")
        self.id = team_id
        # Headers (agent identification)
        self.agent_name = 'k3:' + (agent_name or os.getenv('X_AGENT_NAME') or os.getenv('AGENT_NAME') or 'cpsat')
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


def make_single_plan(n_rooms: int) -> str:
    """スコア2向け：1本のクエリ列。できるだけ長く（<= 18n）かつ分散よく。
    n>=12 なら k=3 をフル(216) or 繰り返し詰め、n<12 は k=2 を繰り返し。
    """
    max_len = 18 * n_rooms
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
def solve_cpsat_guess_map(n_rooms: int, plan: str, observed: List[int]) -> Dict:
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
    solver.parameters.max_time_in_seconds = 10.0
    solver.parameters.num_search_workers = 8
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



# ---------- Core loop ----------

def solve_once(client: AedificiumClient, *, problem_name: str, http_timeout: Optional[int]) -> bool:
    """1回の /select → C++ 対話 → /explore → /guess を実行。成功可否を返す。"""

    # /select
    actual_problem = client.select_problem(problem_name)
    print(f'problem_name: {actual_problem}', flush=True)

    assert problem_name in CATALOG
    num_rooms = CATALOG[problem_name]

    plan = make_single_plan(num_rooms)
    print(f'plan={plan}', flush=True)
    assert not any(ch not in ALPHABET for ch in plan)

    plans = [plan] # single plan
    
    results, qctr = client.explore(plans)
    print(f'results: {results}, qc: {qctr}', flush=True)

    observed = results[0]
    print(f'observed={observed}', flush=True)

    guess = solve_cpsat_guess_map(num_rooms, plan, observed)
    print(f'guess={guess}', flush=True)

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
                        help="override X-Agent-Name header; default uses env X_AGENT_NAME/AGENT_NAME or 'k3:solver'")
    parser.add_argument("--agent-id", type=str, default=None,
                        help="override X-Agent-ID header; default uses env X_AGENT_ID/AGENT_ID or current PID")
    args = parser.parse_args()

    client = make_client(args)

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
                    http_timeout=args.timeout_sec
                )
                if ok:
                    succeeded += 1
            except KeyboardInterrupt:
                print("\nInterrupted by user.", file=sys.stderr, flush=True)
                break
            except Exception as e:
                print(f"[ERROR] iteration {i}: {e}", file=sys.stderr, flush=True)
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


if __name__ == '__main__':
    main()