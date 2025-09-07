from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import os
import sys
import json
import time
import argparse
import requests
import subprocess
import threading

ALPHABET = "012345"  # door labels

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 既定: Release ビルド
DEFAULT_CPP_EXE = os.path.join(ROOT_DIR, 'vs', 'solver', 'bin', 'Release', 'solver.exe')
DEBUG_CPP_EXE   = os.path.join(ROOT_DIR, 'vs', 'solver', 'bin', 'Debug',   'solver.exe')


# ---------- API Client ----------

class AedificiumClient:
    def __init__(self, base_url: str, team_id: str = 'ignored', *, agent_name: Optional[str] = None, agent_id: Optional[str] = None, timeout_sec: Optional[int] = None):
        self.base_url = base_url.rstrip("/")
        self.id = team_id
        # Headers (agent identification)
        self.agent_name = 'k3:' + (agent_name or os.getenv('X_AGENT_NAME') or os.getenv('AGENT_NAME') or 'unnamed')
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


# ---------- C++ Solver Subprocess ----------

class CppSolverProcess:
    def __init__(self, exe_path: str, *, mirror_stderr: bool = True, stderr_prefix: str = "[CPP] "):
        self.proc = subprocess.Popen(
            [exe_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,   # Pipeにして TEE
            text=True,
            encoding="utf-8",
            bufsize=1
        )
        self._mirror_stderr = mirror_stderr
        self._stderr_prefix = stderr_prefix
        self._stderr_buf: List[str] = []
        self._stderr_thread = threading.Thread(target=self._pump_stderr, daemon=True)
        self._stderr_thread.start()

    def _pump_stderr(self):
        try:
            assert self.proc.stderr is not None
            for line in self.proc.stderr:
                self._stderr_buf.append(line)
                if self._mirror_stderr:
                    print(f"{self._stderr_prefix}{line}", file=sys.stderr, end="", flush=True)
        except Exception:
            pass

    def write_line(self, s: str):
        assert self.proc.stdin is not None
        self.proc.stdin.write(s.rstrip("\r\n") + "\n")
        self.proc.stdin.flush()

    def read_line(self, timeout: Optional[float] = None) -> str:
        assert self.proc.stdout is not None
        if timeout is None:
            line = self.proc.stdout.readline()
            if line == "":
                raise RuntimeError("CPP solver closed stdout unexpectedly.\n" + "".join(self._stderr_buf))
            return line.rstrip("\r\n")
        # with timeout
        buf = {"line": None}
        def _t():
            buf["line"] = self.proc.stdout.readline()
        th = threading.Thread(target=_t)
        th.start()
        th.join(timeout)
        if th.is_alive():
            raise TimeoutError("Timed out waiting for line from CPP solver.")
        line = buf["line"]
        if line == "":
            raise RuntimeError("CPP solver closed stdout unexpectedly.\n" + "".join(self._stderr_buf))
        return line.rstrip("\r\n")

    def read_json_object(self, timeout: Optional[float] = None) -> Dict:
        """Read until a full JSON object {} is balanced. Multi-line supported."""
        chunks: List[str] = []
        depth = 0
        started = False

        while True:
            line = self.read_line(timeout=timeout)
            chunks.append(line)
            for ch in line:
                if ch == '{':
                    depth += 1
                    started = True
                elif ch == '}':
                    depth -= 1
            if started and depth <= 0:
                txt = "\n".join(chunks)
                first = txt.find('{')
                last = txt.rfind('}')
                if first == -1 or last == -1 or last < first:
                    raise ValueError("Malformed JSON block from CPP solver:\n" + txt)
                obj_txt = txt[first:last+1]
                return json.loads(obj_txt)

    def close(self):
        try:
            if self.proc and self.proc.poll() is None:
                if self.proc.stdin:
                    try:
                        self.proc.stdin.close()
                    except Exception:
                        pass
                self.proc.terminate()
        except Exception:
            pass


# ---------- Core loop ----------

def solve_once(client: AedificiumClient, *, cpp_exe: str, problem_name: str, mirror_stderr: bool, reuse_solver: bool, http_timeout: Optional[int]) -> bool:
    """1回の /select → C++ 対話 → /explore → /guess を実行。成功可否を返す。"""

    # /select
    actual_problem = client.select_problem(problem_name)
    print(f'problem_name: {actual_problem}', flush=True)

    cpp: Optional[CppSolverProcess] = None
    try:
        # C++ solver 起動
        if not reuse_solver:
            cpp = CppSolverProcess(cpp_exe, mirror_stderr=mirror_stderr)
        else:
            # 使い回し指定だが、最初の呼び出し時は作る
            if not hasattr(solve_once, "_shared_cpp") or getattr(solve_once, "_shared_cpp") is None:
                setattr(solve_once, "_shared_cpp", CppSolverProcess(cpp_exe, mirror_stderr=mirror_stderr))
            cpp = getattr(solve_once, "_shared_cpp")

        # 問題名送信
        cpp.write_line(actual_problem)

        # プラン受領
        try:
            n_plans_line = cpp.read_line()
            n_plans = int(n_plans_line.strip())
        except Exception as e:
            raise RuntimeError(f"Failed to read number of plans from C++: {e}")

        plans: List[str] = []
        for _ in range(n_plans):
            plan = cpp.read_line().strip()
            if any(ch not in ALPHABET for ch in plan):
                raise ValueError(f"Plan contains non-door char: {plan}")
            plans.append(plan)

        # /explore
        results, qctr = client.explore(plans)
        print(f'results: {results}, qc: {qctr}', flush=True)

        # ラベル列送信
        for rec in results:
            line = "".join(str(x) for x in rec)
            cpp.write_line(line)

        # 解(JSON)受領
        sol = cpp.read_json_object()
        print(f'sol={sol}', flush=True)

    finally:
        # 使い回ししない場合は毎回閉じる
        if cpp is not None and not reuse_solver:
            cpp.close()

    # /guess
    result = client.guess(sol)
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


def resolve_exe_path(args: argparse.Namespace) -> str:
    if args.exe:
        return args.exe
    return DEBUG_CPP_EXE if args.debug else DEFAULT_CPP_EXE


def main():
    parser = argparse.ArgumentParser(description="ICFP 2025 Aedificium Python<->C++ bridge")
    parser.add_argument("--server", choices=["local", "proxy", "remote", "custom"], default="local",
                        help="server profile (default: local)")
    parser.add_argument("--base-url", type=str, default=None,
                        help="custom base URL when --server custom")
    parser.add_argument("--problem", type=str, default="primus",
                        help="problem name (e.g., probatio|primus|secundus|tertius|quartus|quintus)")
    parser.add_argument("--exe", type=str, default=None,
                        help="path to C++ solver exe (overrides --debug)")
    parser.add_argument("--debug", action="store_true",
                        help="use Debug build exe instead of Release")
    parser.add_argument("--timeout-sec", type=int, default=None,
                        help="HTTP timeout seconds (overrides env)")
    parser.add_argument("--loop", type=int, default=1,
                        help="number of iterations; 0 means infinite (default: 1)")
    parser.add_argument("--mirror-stderr", dest="mirror_stderr", action="store_true", default=True,
                        help="mirror C++ stderr to Python stderr (default: on)")
    parser.add_argument("--no-mirror-stderr", dest="mirror_stderr", action="store_false",
                        help="disable mirroring C++ stderr")
    parser.add_argument("--reuse-solver", action="store_true",
                        help="reuse a single C++ solver process across iterations")
    parser.add_argument("--sleep-sec", type=float, default=0.0,
                        help="sleep seconds between iterations (default: 0)")
    parser.add_argument("--agent-name", type=str, default=None,
                        help="override X-Agent-Name header; default uses env X_AGENT_NAME/AGENT_NAME or 'k3:solver'")
    parser.add_argument("--agent-id", type=str, default=None,
                        help="override X-Agent-ID header; default uses env X_AGENT_ID/AGENT_ID or current PID")
    args = parser.parse_args()

    cpp_exe = resolve_exe_path(args)
    if not os.path.isfile(cpp_exe):
        raise SystemExit(f"solver exe not found: {cpp_exe}")

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
                    cpp_exe=cpp_exe,
                    problem_name=args.problem,
                    mirror_stderr=args.mirror_stderr,
                    reuse_solver=args.reuse_solver,
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