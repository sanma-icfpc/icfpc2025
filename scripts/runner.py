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
import shutil

# Best-effort ANSI support on Windows consoles
try:
    import colorama  # type: ignore

    try:
        colorama.just_fix_windows_console()
    except Exception:
        colorama.init()
except Exception:
    colorama = None  # type: ignore

import importlib


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
        self.state: str = "idle"  # pending | solving | done | cancelled | terminated
        self._t_pending_start: Optional[float] = None
        # Trial counters (for repeated runs)
        self.trial_index: Optional[int] = None
        self.trial_total: Optional[int] = None  # 0 or None => ∞
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

    def set_state(self, state: str):
        if state not in ("pending", "solving", "done", "cancelled", "terminated", "idle"):
            return
        self.state = state
        if state == "pending":
            self._t_pending_start = time.monotonic()
        # re-render to reflect state bar
        self.render()

    def _status_bar(self) -> str:
        cols = shutil.get_terminal_size(fallback=(100, 25)).columns
        # Elapsed policy:
        # - pending: time since we entered pending (before issuing /select)
        # - solving: time since /select responded (t0)
        # - done: since t0 if available, else 0
        # - cancelled: if cancelled while pending, show pending duration; else since t0
        now = time.monotonic()
        if self.state == "pending":
            base = self._t_pending_start or now
        elif self.state == "solving":
            base = self.t0 or now
        elif self.state == "cancelled":
            base = (self.t0 if self.t0 is not None else (self._t_pending_start or now))
        else:  # done/idle
            base = self.t0 or now
        elapsed = int(now - base)
        h = elapsed // 3600
        m = (elapsed % 3600) // 60
        s = elapsed % 60
        elapsed_str = f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
        # Trial string (e.g., "2 / 10" or "13 / ∞")
        trial_str = ""
        try:
            if self.trial_index is not None:
                tot_disp = "∞" if (self.trial_total is None or self.trial_total == 0) else str(self.trial_total)
                trial_str = f" • {self.trial_index} / {tot_disp}"
        except Exception:
            pass
        label = f" STATUS: {self.state.upper()} • {elapsed_str}{trial_str} "
        color = {
            "pending": "30;43",  # black on yellow
            "solving": "97;44",  # bright white on blue
            "done": "30;42",     # black on green
            "cancelled": "97;41", # bright white on red
            "terminated": "97;45", # bright white on magenta
            "idle": "30;47",     # black on white
        }.get(self.state, "30;47")
        return f"\x1b[{color}m" + label.ljust(cols) + "\x1b[0m"

    def set_trial(self, index: int, total: Optional[int]):
        """Set current trial number and total (0/None => ∞)."""
        try:
            self.trial_index = index
            self.trial_total = total
        except Exception:
            pass
        self.render()

    def prompt_yes_no(self, question: str, default_yes: bool = True) -> Optional[bool]:
        """Prompt the user for a Y/n answer below the board.

        Returns True for yes, False for no, or None if input not possible.
        Ctrl-C during prompt will propagate to caller.
        """
        try:
            # Move cursor to the line after the board and ask
            sys.stdout.write("\n" + question)
            sys.stdout.flush()
            resp = input(" ").strip().lower()
        except KeyboardInterrupt:
            # propagate to allow force termination
            raise
        except Exception:
            # re-render to clean any partial output
            self.render()
            return None
        # Clean: re-render to erase prompt line
        self.render()
        if resp == "":
            return True if default_yes else False
        if resp in ("y", "yes"):
            return True
        if resp in ("n", "no"):
            return False
        # Unrecognized; use default
        return True if default_yes else False

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
            except BaseException:
                pass
            try:
                rss = self._proc.memory_info().rss
            except BaseException:
                pass
            try:
                avail = self._psutil.virtual_memory().available
            except BaseException:
                pass
            try:
                total_threads = self._proc.num_threads()
            except BaseException:
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
            except BaseException:
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
                    except BaseException:
                        pass
                    try:
                        with open('/proc/meminfo', 'r') as f:
                            for line in f:
                                if line.startswith('MemAvailable:'):
                                    avail = int(line.split()[1]) * 1024  # kB -> bytes
                                    break
                    except BaseException:
                        pass
            except BaseException:
                pass
            # Threads
            try:
                total_threads = len(__import__('threading').enumerate())
            except BaseException:
                total_threads = active_threads
        return cpu, rss, avail, active_threads, (total_threads or active_threads)

    def render(self):
        try:
            # Position to top-left and clear on first render
            if not self._cleared:
                print("\x1b[2J\x1b[H", end="")
                self._cleared = True
            else:
                print("\x1b[H", end="")
            lines = []
            # Top status bar with colored background
            lines.append(self._status_bar())
            W = 18
            def L(k, v):
                lines.append(f"{k:<{W}}: {v}")
            # Common
            lines.append("-- Common --")
            L("Server", f"{self.cfg.get('target')} -> {self.cfg.get('base_url')}")
            L("Solver", f"{self.cfg.get('solver')}")
            L("Agent", f"{self.cfg.get('agent')} (ID: {self.cfg.get('agent_id')})")
            L("Problem", f"{self.cfg.get('problem')} (room count n={self.cfg.get('size')})")
            L("Plan Budget", f"{self.cfg.get('plan_mode')} (max door-steps per plan)")
            # System & process one-liner (cross-platform)
            try:
                cpu, rss, avail, act, total = self._proc_stats()
                cpu_str = f"{cpu:4.1f}%" if cpu is not None else " n/a"
                rss_str = self._fmt_bytes_mb(rss) if rss is not None else "n/a"
                avail_str = self._fmt_bytes_mb(avail) if avail is not None else "n/a"
                inactive = max(0, total - act)
                # Clarify: py = live Python threads; os = process-level threads
                L("Sys/Proc", f"{cpu_str} CPU, {rss_str}/{avail_str}, threads active={act}, inactive={inactive}, total={total}")
            except BaseException:
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
                except BaseException:
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
        except KeyboardInterrupt:
            # Swallow Ctrl-C during rendering so main flow can manage interrupts
            return
        except BaseException:
            return


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
        pause_event: Optional["_threading.Event"] = None,
        cancel_event: Optional["_threading.Event"] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.id = team_id
        self.verbose = verbose
        self.agent_name = agent_name
        self.agent_id = agent_id or str(os.getpid())
        self.timeout_sec = timeout_sec
        self.status = status
        self.pause_event = pause_event
        self.cancel_event = cancel_event

    def _post(self, path: str, payload: Dict) -> Dict:
        url = f"{self.base_url}{path}"
        headers = {
            "Content-Type": "application/json",
            "X-Agent-Name": self.agent_name,
            "X-Agent-ID": self.agent_id,
        }
        if self.verbose and self.status is None:
            print(f"POST {url}")
            print("Headers:", headers)
            print("Payload:", json.dumps(payload, ensure_ascii=False))
        # Honor pause (e.g., while user is answering a prompt)
        # Gate on pause and cancel
        try:
            if self.pause_event is not None:
                self.pause_event.wait()
            if self.cancel_event is not None and self.cancel_event.is_set():
                raise RuntimeError("cancelled")
        except BaseException:
            pass
        # Use chunked timeouts for non-/select endpoints so we can react to cancel_event
        r = None
        if path == "/select":
            r = requests.post(url, json=payload, headers=headers, timeout=self.timeout_sec)
        else:
            import math
            chunk = 10.0
            total = float(self.timeout_sec or 600)
            deadline = time.monotonic() + total
            last_err = None
            while True:
                if self.cancel_event is not None and self.cancel_event.is_set():
                    raise RuntimeError("cancelled")
                if self.pause_event is not None:
                    self.pause_event.wait()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    remaining = chunk
                try:
                    r = requests.post(
                        url,
                        json=payload,
                        headers=headers,
                        timeout=min(chunk, remaining),
                    )
                    break
                except requests.exceptions.Timeout as e:
                    last_err = e
                    continue
            if r is None and last_err is not None:
                raise last_err
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
        # If error, dump payload/response for debugging before raising
        if r.status_code >= 400:
            try:
                body_txt = None
                try:
                    body_txt = json.dumps(r.json())
                except Exception:
                    body_txt = r.text
                msg = f"HTTP {r.status_code} on {path}; payload={payload}; response={body_txt}"
                if self.status is not None:
                    # Keep it concise in the free area
                    self.status.add_free(msg[:500])
                else:
                    sys.stderr.write(msg + "\n")
            except Exception:
                pass
            r.raise_for_status()
        # Non-error
        if self.status is not None and path == "/select":
            self.status.t0 = time.monotonic()
            try:
                self.status.set_state("solving")
            except Exception:
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
    parser.add_argument("--problem", choices=list(PROBLEM_SIZES.keys()))
    parser.add_argument("-y", "--yes", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--giveup", action="store_true")
    parser.add_argument(
        "--n-trial",
        type=int,
        default=1,
        help="Number of trials to repeat (0 = infinite)",
    )
    parser.add_argument(
        "--solver",
        default="ctree",
        help="Solver name; imports module 'runner_<name>'",
    )
    args = parser.parse_args()

    base_url = MINOTAUR_URL if args.target == "minotaur" else LOCAL_URL
    # Dynamically import solver module: runner_<solver>
    module_name = f"runner_{args.solver}"
    try:
        solver_module = importlib.import_module(module_name)
    except Exception as e:
        print(f"Error: failed to import solver module '{module_name}': {e}", file=sys.stderr)
        sys.exit(2)
    # Agent name default suggested by solver
    if not args.agent_name:
        args.agent_name = getattr(solver_module, "DEFAULT_AGENT_NAME", "anonymous:agent")
    team_id = os.getenv("ICFP_ID")
    if team_id is None and args.target == "local":
        team_id = "local"

    # Resolve problem
    problem = args.problem or os.getenv("ICFP_PROBLEM", "probatio")
    if problem not in PROBLEM_SIZES:
        print(f"Error: unknown problem '{problem}'. Choose one of {list(PROBLEM_SIZES.keys())}.", file=sys.stderr)
        sys.exit(2)
    size = PROBLEM_SIZES[problem]
    # Determine plan mode (server-aware):
    # - local target: enforce 6n to match local_judge_server limits
    # - minotaur target: lightning set uses 18n, aleph and after use 6n
    lightning_set = {"probatio", "primus", "secundus", "tertius", "quartus", "quintus"}
    plan_mode = "18n" if problem in lightning_set else "6n"

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
    mult = 6 if plan_mode == "6n" else 18
    P("Plan", f"{plan_mode} (budget={mult}*n; n={size} -> {mult*size})")
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
            "plan_mode": f"{plan_mode} (budget={(6 if plan_mode=='6n' else 18)}*n)",
            "solver": args.solver,
            "extra_lines": 8,
        }
    )
    status.render()

    # Pause/cancel controllers to gate HTTP during prompts and cancel background work
    import threading as _threading
    _pause_event = _threading.Event()
    _pause_event.set()
    _cancel_event = _threading.Event()

    client = AedificiumClient(
        base_url=base_url,
        team_id=team_id,
        verbose=args.verbose,
        agent_name=args.agent_name,
        agent_id=str(os.getpid()),
        timeout_sec=6000,
        status=status,
        pause_event=_pause_event,
        cancel_event=_cancel_event,
    )

    if args.giveup:
        res = client.minotaur_giveup()
        try:
            status.add_free(f"minotaur_giveup: {res}")
            status.set_state("cancelled")
        except Exception:
            pass
        return

    # Helper to handle Ctrl-C prompts consistently
    def _handle_interrupt(q_minotaur: str, q_other: str):
        try:
            _pause_event.clear()
        except BaseException:
            pass
        try:
            ans = status.prompt_yes_no(q_minotaur if args.target == "minotaur" else q_other, default_yes=True)
        except KeyboardInterrupt:
            sys.exit(130)
        if ans is True:
            try:
                status.set_state("terminated")
            except Exception:
                pass
            if args.target == "minotaur":
                _cancel_event.set()
                _res = client.minotaur_giveup()
                try:
                    status.add_free(f"minotaur_giveup: {_res}")
                except Exception:
                    pass
            sys.exit(0)
        else:
            try:
                status.render()
            except Exception:
                pass
            try:
                _pause_event.set()
            except BaseException:
                pass

    # Create solver via module factory (runner remains solver-agnostic)
    solve_fn = getattr(solver_module, "solve", None)
    if solve_fn is None:
        # Backward-compatibility fallback
        create_solver = getattr(solver_module, "create_solver", None)
        build_guess = getattr(solver_module, "build_guess_from_hypothesis", None)
        if create_solver is None or build_guess is None:
            print(
                f"Error: solver '{args.solver}' does not expose required API (solve) or (create_solver + build_guess_from_hypothesis)",
                file=sys.stderr,
            )
            sys.exit(2)
        def _compat_solve(client, problem, size, plan_mode, status):
            learner = create_solver(client=client, problem=problem, size=size, plan_mode=plan_mode, status=status)
            return build_guess(learner.learn())
        solve_fn = _compat_solve

    # Trial loop: repeat select -> solve -> guess
    total_trials = args.n_trial
    i = 1
    while True:
        try:
            _cancel_event.clear()
        except BaseException:
            pass
        # Update trial display (0 or None => ∞)
        try:
            status.set_trial(i, total_trials)
        except Exception:
            pass

        # Indicate waiting/pending for /select
        try:
            status.set_state("pending")
            status.add_free("Waiting for our turn… Press Ctrl-C to open prompt")
        except Exception:
            pass
        # Run /select in a background thread to keep main thread responsive to Ctrl-C
        import threading as _threading
        _select_done = {"done": False, "value": None, "error": None}
        def _do_select():
            try:
                _select_done["value"] = client.select_problem(problem)
            except Exception as e:
                _select_done["error"] = e
            finally:
                _select_done["done"] = True
        _t = _threading.Thread(target=_do_select, daemon=True)
        _t.start()
        while _t.is_alive():
            try:
                time.sleep(0.1)
            except KeyboardInterrupt:
                _handle_interrupt(
                    "Give up? (sends /minotaur_giveup to step aside) [Y/n]",
                    "Abort? [Y/n]",
                )
        # Thread finished: check result
        if _select_done["error"] is not None:
            # Re-raise to use existing error handling path
            raise _select_done["error"]
        chosen = _select_done["value"]
        try:
            status.add_free(f"Selected: {chosen}")
        except Exception:
            pass

        # Run solve in background to keep Ctrl-C responsive on Windows/Linux
        import threading as _threading
        _solve_done = {"done": False, "value": None, "error": None}
        def _do_solve():
            try:
                _solve_done["value"] = solve_fn(
                    client=client,
                    problem=problem,
                    size=size,
                    plan_mode=plan_mode,
                    status=status,
                    cancel_event=_cancel_event,
                    use_graffiti=(args.target == "minotaur"),
                )
            except BaseException as e:
                _solve_done["error"] = e
            finally:
                _solve_done["done"] = True
        _ts = _threading.Thread(target=_do_solve, daemon=True)
        _ts.start()
        try:
            status.add_free("Solving… Press Ctrl-C to open prompt")
        except Exception:
            pass
        while _ts.is_alive():
            try:
                time.sleep(0.1)
            except KeyboardInterrupt:
                _handle_interrupt(
                    "Give up solving? (sends /minotaur_giveup and exits) [Y/n]",
                    "Abort solving? [Y/n]",
                )
        if _solve_done["error"] is not None:
            raise _solve_done["error"]
        guess_map = _solve_done["value"]

        # Run guess in background to avoid Ctrl-C killing request
        _guess_done = {"done": False, "value": None, "error": None}
        def _do_guess():
            try:
                _guess_done["value"] = client.guess(guess_map)
            except BaseException as e:
                _guess_done["error"] = e
            finally:
                _guess_done["done"] = True
        _tg = _threading.Thread(target=_do_guess, daemon=True)
        _tg.start()
        try:
            status.add_free("Submitting… Press Ctrl-C to open prompt")
        except Exception:
            pass
        while _tg.is_alive():
            try:
                time.sleep(0.1)
            except KeyboardInterrupt:
                _handle_interrupt(
                    "Give up while submitting? (sends /minotaur_giveup) [Y/n]",
                    "Abort submitting? [Y/n]",
                )
        if _guess_done["error"] is not None:
            ex = _guess_done["error"]
            if isinstance(ex, requests.HTTPError):
                try:
                    status.add_free(f"HTTP error during /guess: {ex}")
                    status.set_state("done")
                except Exception:
                    pass
                sys.exit(1)
            else:
                try:
                    status.add_free(f"Error during /guess: {type(ex).__name__}: {ex}")
                    status.set_state("done")
                except Exception:
                    pass
                sys.exit(1)
        ok = _guess_done["value"]
        try:
            status.add_free(f"Guess correct? {ok}")
            status.set_state("done")
        except Exception:
            pass

        # Exit or continue to next trial
        if bool(ok):
            break
        if total_trials != 0 and i >= total_trials:
            break
        i += 1
        try:
            status.add_free("--- Next trial starting ---")
        except Exception:
            pass


if __name__ == "__main__":
    main()
