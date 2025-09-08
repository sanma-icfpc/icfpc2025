#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Dict, List, Optional
import os
import sys
import json
import time
import argparse
import platform
import requests

# Best-effort ANSI support on Windows consoles
try:
    import colorama  # type: ignore

    try:
        colorama.just_fix_windows_console()
    except Exception:
        colorama.init()
except Exception:
    colorama = None  # type: ignore

from ctree import ExploreOracle, ClassTreeMooreLearner, build_guess_from_hypothesis


# URLs and problems
LOCAL_URL = "http://127.0.0.1:8009"
MINOTAUR_URL = "http://tk2-401-41624.vs.sakura.ne.jp:19384"

PROBLEM_SIZES = {
    "probatio": 3,
    "primus": 6,
    "secundus": 12,
    "tertius": 18,
    "quartus": 24,
    "quintus": 30,
    "aleph": 12,
    "beth": 24,
    "gimel": 36,
    "daleth": 48,
    "he": 60,
    "vau": 18,
    "zain": 36,
    "hhet": 54,
    "teth": 72,
    "iod": 90,
}


class StatusBoard:
    def __init__(self, config: Dict[str, str]):
        self.cfg = config
        self.t0: Optional[float] = None  # starts after /select
        self.requests_total = 0
        self.req_counts = {"/select": 0, "/explore": 0, "/guess": 0, "/minotaur_giveup": 0}
        self.last_endpoint = None
        self.last_status = None
        self.last_time = None
        self.explore_commits = 0
        self.plans_total = 0
        self.last_plans = 0
        self.last_doors_min = 0
        self.last_doors_max = 0
        self.last_doors_avg = 0.0
        self.qc_last = 0
        self.pending = 0
        self.deferred = 0
        self.states = 0
        self.exps = 0
        self.leaves = 0
        self.refinements = 0
        self.hyp_states_last = 0
        self._rendered_lines = 0
        self.extra: List[str] = []
        self.extra_lines: int = int(config.get("extra_lines", 8))
        self._solver_renderer = None  # optional callable returning list[str]
        # psutil (optional)
        try:
            import psutil  # type: ignore
            self._psutil = psutil
            self._proc = psutil.Process(os.getpid())
            # prime cpu percent measurement
            try:
                self._proc.cpu_percent(None)
            except Exception:
                pass
        except Exception:
            self._psutil = None
            self._proc = None
        self._cleared = False
        # Fallback CPU% via process_time delta
        self._last_proc_time: Optional[float] = None
        self._last_wall_time: Optional[float] = None

    def _fmt_dur(self, secs: float) -> str:
        secs = int(secs)
        h = secs // 3600
        m = (secs % 3600) // 60
        s = secs % 60
        if h:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    def update_request(self, path: str, status_code: int):
        self.requests_total += 1
        self.req_counts[path] = self.req_counts.get(path, 0) + 1
        self.last_endpoint = path
        self.last_status = status_code
        self.last_time = time.strftime("%H:%M:%S")

    def update_explore(self, plans_count: int, doors_min: int, doors_max: int, doors_avg: float, qc_last: int, pending: int, deferred: int):
        self.explore_commits += 1
        self.plans_total += plans_count
        self.last_plans = plans_count
        self.last_doors_min = doors_min
        self.last_doors_max = doors_max
        self.last_doors_avg = doors_avg
        self.qc_last = qc_last
        self.pending = pending
        self.deferred = deferred

    def update_solver(self, states: int = None, exps: int = None, leaves: int = None):
        if states is not None:
            self.states = states
        if exps is not None:
            self.exps = exps
        if leaves is not None:
            self.leaves = leaves

    def bump_refinements(self):
        self.refinements += 1

    def set_hyp_states(self, n: int):
        self.hyp_states_last = n

    def add_free(self, msg: str):
        self.extra.append(msg)
        self.render()

    def set_solver_renderer(self, renderer):
        """Register a callable that returns lines to render in the solver pane."""
        self._solver_renderer = renderer

    def _fmt_bytes_mb(self, b: int) -> str:
        try:
            return f"{b/1024/1024:.1f} MB"
        except Exception:
            return str(b)

    def _proc_stats(self):
        import threading
        # Defaults
        cpu = 0.0
        rss = None
        avail = None
        total_threads = None
        active_threads = threading.active_count()
        # psutil path
        if self._proc is not None and self._psutil is not None:
            try:
                cpu = self._proc.cpu_percent(0.0)
            except Exception:
                pass
            try:
                rss = self._proc.memory_info().rss
            except Exception:
                pass
            try:
                avail = self._psutil.virtual_memory().available
            except Exception:
                pass
            try:
                total_threads = self._proc.num_threads()
            except Exception:
                pass
        else:
            # Fallbacks (cross-platform)
            try:
                # CPU% approx via process_time delta over wall time and CPU count
                now_pt = time.process_time()
                now_wt = time.monotonic()
                if self._last_proc_time is not None and self._last_wall_time is not None:
                    dpt = max(0.0, now_pt - self._last_proc_time)
                    dwt = max(1e-6, now_wt - self._last_wall_time)
                    ncpu = max(1, os.cpu_count() or 1)
                    cpu = (dpt / (dwt * ncpu)) * 100.0
                self._last_proc_time = now_pt
                self._last_wall_time = now_wt
            except Exception:
                pass
            # Memory
            try:
                if os.name == 'nt':
                    # Windows: use ctypes to query process working set and available memory
                    import ctypes, ctypes.wintypes as wt
                    psapi = ctypes.WinDLL('Psapi.dll')
                    kernel32 = ctypes.WinDLL('kernel32.dll')
                    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                        _fields_ = [
                            ('cb', wt.DWORD),
                            ('PageFaultCount', wt.DWORD),
                            ('PeakWorkingSetSize', wt.SIZE_T),
                            ('WorkingSetSize', wt.SIZE_T),
                            ('QuotaPeakPagedPoolUsage', wt.SIZE_T),
                            ('QuotaPagedPoolUsage', wt.SIZE_T),
                            ('QuotaPeakNonPagedPoolUsage', wt.SIZE_T),
                            ('QuotaNonPagedPoolUsage', wt.SIZE_T),
                            ('PagefileUsage', wt.SIZE_T),
                            ('PeakPagefileUsage', wt.SIZE_T),
                        ]
                    pmc = PROCESS_MEMORY_COUNTERS()
                    pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
                    hproc = kernel32.GetCurrentProcess()
                    if psapi.GetProcessMemoryInfo(hproc, ctypes.byref(pmc), pmc.cb):
                        rss = int(pmc.WorkingSetSize)
                    class MEMORYSTATUSEX(ctypes.Structure):
                        _fields_ = [
                            ('dwLength', wt.DWORD),
                            ('dwMemoryLoad', wt.DWORD),
                            ('ullTotalPhys', ctypes.c_uint64),
                            ('ullAvailPhys', ctypes.c_uint64),
                            ('ullTotalPageFile', ctypes.c_uint64),
                            ('ullAvailPageFile', ctypes.c_uint64),
                            ('ullTotalVirtual', ctypes.c_uint64),
                            ('ullAvailVirtual', ctypes.c_uint64),
                            ('sullAvailExtendedVirtual', ctypes.c_uint64),
                        ]
                    gm = MEMORYSTATUSEX()
                    gm.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                    if kernel32.GlobalMemoryStatusEx(ctypes.byref(gm)):
                        avail = int(gm.ullAvailPhys)
                else:
                    # Linux/Unix: read /proc
                    try:
                        with open('/proc/self/statm', 'r') as f:
                            parts = f.read().strip().split()
                            if len(parts) >= 2:
                                rss_pages = int(parts[1])
                                page_size = os.sysconf('SC_PAGE_SIZE')
                                rss = rss_pages * page_size
                    except Exception:
                        pass
                    try:
                        with open('/proc/meminfo', 'r') as f:
                            for line in f:
                                if line.startswith('MemAvailable:'):
                                    avail = int(line.split()[1]) * 1024  # kB -> bytes
                                    break
                    except Exception:
                        pass
            except Exception:
                pass
            # Threads
            try:
                total_threads = len(__import__('threading').enumerate())
            except Exception:
                total_threads = active_threads
        return cpu, rss, avail, active_threads, (total_threads or active_threads)

    def render(self):
        # Position to top-left and clear on first render
        if not self._cleared:
            print("\x1b[2J\x1b[H", end="")
            self._cleared = True
        else:
            print("\x1b[H", end="")
        lines = []
        W = 18
        def L(k, v):
            lines.append(f"{k:<{W}}: {v}")
        lines.append("=== Aedificium Status ===")
        # Common
        lines.append("-- Common --")
        L("Server", f"{self.cfg.get('target')} -> {self.cfg.get('base_url')}")
        L("Solver", f"{self.cfg.get('solver')} (algorithm)")
        L("Agent", f"{self.cfg.get('agent')} (X-Agent-Name, id: {self.cfg.get('agent_id')})")
        L("Problem", f"{self.cfg.get('problem')} (room count n={self.cfg.get('size')})")
        L("Plan Budget", f"{self.cfg.get('plan_mode')} (max door-steps per plan)")
        elapsed_str = self._fmt_dur(time.monotonic() - self.t0) if self.t0 is not None else "00:00"
        L("Elapsed (/select)", elapsed_str)
        # System & process one-liner (cross-platform)
        try:
            cpu, rss, avail, act, total = self._proc_stats()
            cpu_str = f"{cpu:4.1f}%" if cpu is not None else " n/a"
            rss_str = self._fmt_bytes_mb(rss) if rss is not None else "n/a"
            avail_str = self._fmt_bytes_mb(avail) if avail is not None else "n/a"
            inactive = max(0, total - act)
            # Clarify that 'py' counts live Python threads; 'os' is process thread count
            L("Sys/Proc", f"{cpu_str} CPU, {rss_str}/{avail_str}, threads py={act}, os={total} (inactive={inactive})")
        except Exception:
            L("Sys/Proc", "n/a")
        # Network
        lines.append("-- Network --")
        L(
            "Requests total",
            f"{self.requests_total} (select:{self.req_counts.get('/select',0)}, explore:{self.req_counts.get('/explore',0)}, guess:{self.req_counts.get('/guess',0)}, giveup:{self.req_counts.get('/minotaur_giveup',0)})",
        )
        last_ep = self.last_endpoint if self.last_endpoint is not None else "-"
        last_st = self.last_status if self.last_status is not None else "-"
        last_tm = self.last_time if self.last_time is not None else "-"
        L("Last POST", f"{last_ep} -> HTTP {last_st} at {last_tm}")
        # Explore
        lines.append("-- Explore --")
        L("Explore commits", f"{self.explore_commits}")
        L("Last batch", f"{self.last_plans} plans (doors min/max/avg = {self.last_doors_min}/{self.last_doors_max}/{self.last_doors_avg:.2f})")
        L("Plans sent", f"{self.plans_total}")
        L("queryCount (srv)", f"{self.qc_last}")
        L("Pending/Deferred", f"{self.pending} queued / {self.deferred} over-budget")
        # Solver
        lines.append("-- Solver --")
        if self._solver_renderer is not None:
            try:
                for ln in self._solver_renderer():
                    lines.append(ln)
            except Exception:
                L("States", f"{self.states} (hyp:{self.hyp_states_last})")
                L("Experiments", f"{self.exps}")
                L("Leaves", f"{self.leaves}")
                L("Refinements", f"{self.refinements}")
        else:
            L("States", f"{self.states} (hyp:{self.hyp_states_last})")
            L("Experiments", f"{self.exps}")
            L("Leaves", f"{self.leaves}")
            L("Refinements", f"{self.refinements}")
        # Output area
        lines.append("-- Output (free area) --")
        recent = self.extra[-self.extra_lines:]
        pad = [""] * (self.extra_lines - len(recent))
        recent = pad + recent
        for msg in recent:
            lines.append(msg)
        # Print and clear remainder of previous render if any
        total = max(self._rendered_lines, len(lines))
        for i in range(total):
            sys.stdout.write("\x1b[2K\r")
            if i < len(lines):
                print(lines[i])
            else:
                print("")
        self._rendered_lines = len(lines)


class AedificiumClient:
    def __init__(
        self,
        base_url: str,
        team_id: Optional[str] = None,
        verbose: bool = False,
        agent_name: str = "tsuzuki:ctree",
        agent_id: Optional[str] = None,
        timeout_sec: Optional[float] = 6000,
        status: Optional[StatusBoard] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.id = team_id
        self.verbose = verbose
        self.agent_name = agent_name
        self.agent_id = agent_id or str(os.getpid())
        self.timeout_sec = timeout_sec
        self.status = status

    def _post(self, path: str, payload: Dict) -> Dict:
        url = f"{self.base_url}{path}"
        headers = {
            "Content-Type": "application/json",
            "X-Agent-Name": self.agent_name,
            "X-Name-Agent": self.agent_name,
            "X-Agent-ID": self.agent_id,
        }
        if self.verbose and self.status is None:
            print(f"POST {url}")
            print("Headers:", headers)
            print("Payload:", json.dumps(payload, ensure_ascii=False))
        r = requests.post(url, json=payload, headers=headers, timeout=self.timeout_sec)
        if self.status is not None:
            try:
                self.status.update_request(path, r.status_code)
            except Exception:
                pass
        if self.verbose and self.status is None:
            try:
                print("Response:", r.status_code, r.json())
            except Exception:
                print("Response:", r.status_code, r.text)
        r.raise_for_status()
        if self.status is not None and path == "/select":
            self.status.t0 = time.monotonic()
            self.status.render()
        try:
            return r.json()
        except Exception:
            return {"_raw": r.text}

    def select_problem(self, problem_name: str) -> str:
        payload = {"problemName": problem_name}
        if self.id is not None:
            payload["id"] = self.id
        j = self._post("/select", payload)
        return j.get("problemName", problem_name)

    def explore(self, plans: List[str]):
        payload = {"plans": plans}
        if self.id is not None:
            payload["id"] = self.id
        j = self._post("/explore", payload)
        return j["results"], j["queryCount"]

    def guess(self, guess_map: Dict) -> bool:
        payload = {"map": guess_map}
        if self.id is not None:
            payload["id"] = self.id
        j = self._post("/guess", payload)
        return bool(j.get("correct", False))

    def minotaur_giveup(self) -> Dict:
        try:
            return self._post("/minotaur_giveup", {})
        except Exception:
            return {"ok": False}


def main():
    parser = argparse.ArgumentParser(description="ICFP 2025 runner")
    parser.add_argument("--target", choices=["local", "minotaur"], default="local")
    parser.add_argument("--agent-name", default=None)
    parser.add_argument("--plan-length", choices=["6n", "18n"], default="6n")
    parser.add_argument("--problem", choices=list(PROBLEM_SIZES.keys()))
    parser.add_argument("-y", "--yes", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--giveup", action="store_true")
    parser.add_argument("--solver", choices=["ctree"], default="ctree")
    args = parser.parse_args()

    base_url = MINOTAUR_URL if args.target == "minotaur" else LOCAL_URL
    # Agent name default suggested by solver
    if not args.agent_name:
        solver_agent_defaults = {"ctree": "tsuzuki:ctree"}
        args.agent_name = solver_agent_defaults.get(args.solver, "anonymous:agent")
    team_id = os.getenv("ICFP_ID")

    # Resolve problem
    problem = args.problem or os.getenv("ICFP_PROBLEM", "probatio")
    if problem not in PROBLEM_SIZES:
        print(f"Error: unknown problem '{problem}'. Choose one of {list(PROBLEM_SIZES.keys())}.", file=sys.stderr)
        sys.exit(2)
    size = PROBLEM_SIZES[problem]

    # Clear screen (Windows/Linux) and show startup info
    try:
        # ANSI clear + home
        sys.stdout.write("\x1b[2J\x1b[H")
        sys.stdout.flush()
    except Exception:
        try:
            os.system("cls" if os.name == "nt" else "clear")
        except Exception:
            pass
    print("=== Aedificium Runner ===")
    W = 16
    def P(k, v):
        print(f"{k:<{W}}: {v}")
    P("Target", f"{args.target} -> {base_url}")
    P("Solver", args.solver)
    P("Agent", f"{args.agent_name}")
    P("Team ID set", str(bool(team_id)))
    P("Problem", f"{problem} (n={size})")
    P("Plan", f"{args.plan_length} (budget={(6 if args.plan_length=='6n' else 18)}*n; n={size} -> {(6 if args.plan_length=='6n' else 18)*size})")
    P("Python", platform.python_version())
    P("Requests", requests.__version__)

    if not args.yes:
        try:
            resp = input("Proceed with these settings? [y/N] ").strip().lower()
        except KeyboardInterrupt:
            print("\nAborted.")
            sys.exit(130)
        if resp not in ("y", "yes"):
            print("Aborted.")
            sys.exit(0)

    # Initialize status board
    status = StatusBoard(
        {
            "target": args.target,
            "base_url": base_url,
            "agent": args.agent_name,
            "agent_id": str(os.getpid()),
            "problem": problem,
            "size": size,
            "plan_mode": f"{args.plan_length} (budget={(6 if args.plan_length=='6n' else 18)}*n)",
            "solver": args.solver,
            "extra_lines": 8,
        }
    )
    status.render()

    client = AedificiumClient(
        base_url=base_url,
        team_id=team_id,
        verbose=args.verbose,
        agent_name=args.agent_name,
        agent_id=str(os.getpid()),
        timeout_sec=6000,
        status=status,
    )

    if args.giveup:
        res = client.minotaur_giveup()
        try:
            status.add_free(f"minotaur_giveup: {res}")
        except Exception:
            pass
        return

    chosen = client.select_problem(problem)
    try:
        status.add_free(f"Selected: {chosen}")
    except Exception:
        pass

    # Solver selection
    if args.solver == "ctree":
        oracle = ExploreOracle(client, plan_length_mode=args.plan_length, fixed_n=size, status=status)
        learner = ClassTreeMooreLearner(oracle, use_graffiti=True, status=status)
    else:
        print("Unknown solver", args.solver)
        sys.exit(2)

    hyp = learner.learn()
    guess_map = build_guess_from_hypothesis(hyp)

    try:
        ok = client.guess(guess_map)
        try:
            status.add_free(f"Guess correct? {ok}")
        except Exception:
            pass
    except requests.HTTPError as e:
        try:
            status.add_free(f"HTTP error during /guess: {e}")
        except Exception:
            pass
        sys.exit(1)
    except Exception as ex:
        try:
            status.add_free(f"Error during /guess: {type(ex).__name__}: {ex}")
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
