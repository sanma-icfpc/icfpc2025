#!/usr/bin/env python3
"""
written by Codex, based on k3/vs/solver/src/solver.cpp

Local Judge HTTP Server (ICFPC 2025)

Endpoints (JSON over HTTP POST):
  - /register  : {"name": str, "pl": str, "email": str} -> {"id": str}
  - /select    : {"id": str, "problemName": str, ["seed": int]} -> {"problemName": str}
  - /explore   : {"id": str, "plans": [str]} -> {"results": [[int]], "queryCount": int}
  - /guess     : {"id": str, "map": {"rooms": [int], "startingRoom": int, "connections": [{"from":{"room":int,"door":int}, "to":{"room":int,"door":int}}]}}
                 -> {"correct": bool}

Behavior mirrors the C++ LocalJudge in k3/vs/solver/src/solver.cpp, with one usability tweak:
  - Problems: {probatio:3, primus:6, secundus:12, tertius:18, quartus:24, quintus:30}
  - explore(plan): returns labels AFTER each step (length == len(plan))
  - queryCount increases by len(plans) + 1 per /explore request (batching penalty)
  - Note: unlike the official judge, this server does NOT reset the selected problem on /guess; the selection persists until another /select.

Pure stdlib implementation for portability (Linux/macOS/Windows/WSL2).
"""

from __future__ import annotations

import json
import random
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import argparse
import datetime
import webbrowser
from typing import Dict, List, Optional, Tuple


# ==================== Core Structures ====================
Door = int
Room = int
Port = Tuple[Room, Door]  # (room, door)


class Labyrinth:
    def __init__(self, labels: List[int], start: Room, to: List[List[Port]]):
        self.labels = labels
        self.start = start
        self.to = to  # to[r][d] = (room, door)

    def step(self, r: Room, d: Door) -> Room:
        nr, _nd = self.to[r][d]
        return nr

    def explore_plan(self, plan: str) -> List[int]:
        # Record label AFTER each move (length == len(plan))
        cur = self.start
        out: List[int] = []
        out.append(self.labels[cur])
        for ch in plan:
            d = ord(ch) - 48  # fast int(ch) for '0'..'9'
            if d < 0 or d > 5:
                raise ValueError("plan contains non-door digit")
            cur = self.step(cur, d)
            out.append(self.labels[cur])
        return out


def generate_random_labyrinth(n_rooms: int, seed: Optional[int]) -> Labyrinth:
    rng = random.Random(seed if seed is not None else time.time_ns())
    labels = [rng.randrange(4) for _ in range(n_rooms)]
    start = 0

    # Enumerate all ports and shuffle
    ports: List[Port] = [(r, d) for r in range(n_rooms) for d in range(6)]
    rng.shuffle(ports)

    # Pairing with small chance of self-loop
    p_self = 0.04
    pair_of: Dict[Port, Port] = {}
    i = 0
    while i < len(ports):
        p = ports[i]
        i += 1
        if p in pair_of:
            continue
        do_self = (rng.random() < p_self)
        if do_self:
            pair_of[p] = p
            continue
        # find next unpaired port q
        j = i
        while j < len(ports) and ports[j] in pair_of:
            j += 1
        if j >= len(ports):
            pair_of[p] = p  # residue self-loop
        else:
            q = ports[j]
            pair_of[p] = q
            pair_of[q] = p
            # skip q by swapping with i-1 (like C++ implementation comment)
            ports[j], ports[i - 1] = ports[i - 1], ports[j]

    # Build adjacency table
    to: List[List[Port]] = [[(-1, -1) for _ in range(6)] for _ in range(n_rooms)]
    for (r, d), (r2, d2) in pair_of.items():
        to[r][d] = (r2, d2)

    # Sanity check: all defined
    for r in range(n_rooms):
        for d in range(6):
            if to[r][d] == (-1, -1):
                raise RuntimeError("internal generator error: incomplete port")

    return Labyrinth(labels=labels, start=start, to=to)


def equivalent(a: Labyrinth, b: Labyrinth) -> bool:
    if len(a.labels) != len(b.labels):
        return False
    from collections import deque

    def enc(u: int, v: int) -> int:
        return (u << 32) ^ (v & 0xFFFFFFFF)

    q = deque()
    seen = set()
    q.append((a.start, b.start))
    seen.add(enc(a.start, b.start))

    while q:
        u, v = q.popleft()
        for d in range(6):
            u2, _ = a.to[u][d]
            v2, _ = b.to[v][d]
            if a.labels[u2] != b.labels[v2]:
                return False
            key = enc(u2, v2)
            if key not in seen:
                seen.add(key)
                q.append((u2, v2))
    return True


# ==================== Judge State ====================
CATALOG = {
    "probatio": 3,
    "primus": 6,
    "secundus": 12,
    "tertius": 18,
    "quartus": 24,
    "quintus": 30,
    "superdumb": 1,
}


class TeamState:
    def __init__(self) -> None:
        self.active: Optional[Labyrinth] = None
        self.problem_name: Optional[str] = None
        self.query_count: int = 0
        self.seed: Optional[int] = None

    def reset_problem(self) -> None:
        self.active = None
        self.problem_name = None
        self.query_count = 0
        self.seed = None


TEAMS: Dict[str, TeamState] = {}
TEAMS_LOCK = threading.Lock()
REQ_COUNTER = 0
# Logging controls
# - JUDGE_LOG: high-level judge process logs (on by default)
# - HTTP_VERBOSE: raw HTTP request/response payload logs (off by default)
JUDGE_LOG = True
HTTP_VERBOSE = False
# RNG controls (default: fixed seed for reproducibility)
RNG_FIXED_DEFAULT = True
RNG_DEFAULT_SEED = 2025
VISUALIZE = True
VIZ_PORT_DEFAULT = 8002

# Visualizer shared state
VIS_LOG_BUFFER: List[str] = []
VIS_EVENTS: List[Dict] = []
VIS_NEXT_EVENT_ID = 1
SSE_CLIENTS: List[Tuple[BaseHTTPRequestHandler, threading.Lock]] = []
VIS_LOCK = threading.Lock()

# Sessions: each begins at /select and aggregates subsequent explores/guesses until next /select
SESSIONS: List[Dict] = []
CURRENT_SESSION_ID: Optional[int] = None


def now_ts() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]


def vlog(msg: str) -> None:
    if JUDGE_LOG:
        line = f"judge: {msg}"
        sys.stderr.write(line + "\n")
        with VIS_LOCK:
            VIS_LOG_BUFFER.append(line)
            # limit buffer
            if len(VIS_LOG_BUFFER) > 2000:
                del VIS_LOG_BUFFER[:1000]
            # broadcast to SSE clients
            for handler, lock in list(SSE_CLIENTS):
                try:
                    with lock:
                        handler.wfile.write(b"event: log\n")
                        handler.wfile.write(b"data: ")
                        handler.wfile.write(json.dumps({"line": line}).encode("utf-8"))
                        handler.wfile.write(b"\n\n")
                        handler.wfile.flush()
                except Exception:
                    SSE_CLIENTS.remove((handler, lock))


def json_response(handler: BaseHTTPRequestHandler, code: int, payload: Dict) -> None:
    data = json.dumps(payload).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)
    if HTTP_VERBOSE:
        sys.stderr.write(f"[http {now_ts()}] -> {handler.path} {code} bytes={len(data)} resp={payload}\n")


def error(handler: BaseHTTPRequestHandler, code: int, message: str) -> None:
    json_response(handler, code, {"error": message})


def build_labyrinth_from_guess(map_obj: Dict) -> Labyrinth:
    rooms = map_obj.get("rooms")
    starting = map_obj.get("startingRoom")
    conns = map_obj.get("connections")
    if not isinstance(rooms, list) or not rooms:
        raise ValueError("rooms must be a non-empty list")
    if not isinstance(starting, int):
        raise ValueError("startingRoom must be int")
    if not isinstance(conns, list):
        raise ValueError("connections must be a list")

    n = len(rooms)
    to: List[List[Port]] = [[(-1, -1) for _ in range(6)] for _ in range(n)]

    def in_range_room(r: int) -> bool:
        return 0 <= r < n

    def in_range_door(d: int) -> bool:
        return 0 <= d < 6

    for c in conns:
        fr = c.get("from", {})
        to_ = c.get("to", {})
        r1, d1 = fr.get("room"), fr.get("door")
        r2, d2 = to_.get("room"), to_.get("door")
        if not (isinstance(r1, int) and isinstance(d1, int) and isinstance(r2, int) and isinstance(d2, int)):
            raise ValueError("connection endpoints must be integers")
        if not (in_range_room(r1) and in_range_room(r2) and in_range_door(d1) and in_range_door(d2)):
            raise ValueError("connection out of range")
        if to[r1][d1] != (-1, -1) or to[r2][d2] != (-1, -1):
            raise ValueError("duplicate assignment on a door")
        # Undirected: wire both sides; self-loop allowed
        to[r1][d1] = (r2, d2)
        to[r2][d2] = (r1, d1)

    # Ensure all doors are defined
    for r in range(n):
        for d in range(6):
            if to[r][d] == (-1, -1):
                raise ValueError(f"door undefined: room={r} door={d}")

    return Labyrinth(labels=list(rooms), start=starting, to=to)


def labyrinth_to_wire(L: Optional[Labyrinth], name: Optional[str], seed: Optional[int]) -> Dict:
    if L is None:
        return {"active": False}
    return {
        "active": True,
        "problemName": name,
        "seed": seed,
        "rooms": L.labels,
        "start": L.start,
        "to": L.to,
        "numRooms": len(L.labels),
    }


def add_event(ev: Dict) -> None:
    global VIS_NEXT_EVENT_ID
    with VIS_LOCK:
        ev = dict(ev)
        ev.setdefault("id", VIS_NEXT_EVENT_ID)
        VIS_NEXT_EVENT_ID += 1
        VIS_EVENTS.append(ev)
        if len(VIS_EVENTS) > 1000:
            del VIS_EVENTS[:500]
        # broadcast SSE
        for handler, lock in list(SSE_CLIENTS):
            try:
                with lock:
                    handler.wfile.write(b"event: event\n")
                    handler.wfile.write(b"data: ")
                    handler.wfile.write(json.dumps(ev).encode("utf-8"))
                    handler.wfile.write(b"\n\n")
                    handler.wfile.flush()
            except Exception:
                SSE_CLIENTS.remove((handler, lock))


# ==================== HTTP Handler ====================
class JudgeHandler(BaseHTTPRequestHandler):
    server_version = "LocalJudge/1.0"

    def do_POST(self) -> None:  # noqa: N802
        global REQ_COUNTER
        REQ_COUNTER += 1
        req_id = REQ_COUNTER
        length = int(self.headers.get("Content-Length", "0"))
        try:
            body = self.rfile.read(length)
            payload = json.loads(body.decode("utf-8")) if body else {}
        except Exception:
            return error(self, 400, "Invalid JSON body")

        if HTTP_VERBOSE:
            sys.stderr.write(f"[http {now_ts()}] <-#{req_id} {self.client_address[0]} POST {self.path} body={payload}\n")
        elif JUDGE_LOG:
            sys.stderr.write(f"[judge {now_ts()}] POST {self.path}\n")

        if self.path == "/register":
            return self.handle_register(payload)
        if self.path == "/select":
            return self.handle_select(payload)
        if self.path == "/explore":
            return self.handle_explore(payload)
        if self.path == "/guess":
            return self.handle_guess(payload)

        return error(self, 404, "Unknown path")

    # Silence default logs (keep minimal noise)
    def log_message(self, fmt: str, *args) -> None:  # noqa: A003 - keep signature
        if HTTP_VERBOSE:
            sys.stderr.write(f"[http {now_ts()}] " + (fmt % args) + "\n")

    # -------- Handlers --------
    def handle_register(self, payload: Dict) -> None:
        name = payload.get("name")
        pl = payload.get("pl")
        email = payload.get("email")
        if not (isinstance(name, str) and isinstance(pl, str) and isinstance(email, str)):
            return error(self, 400, "name, pl, email are required")

        # Create a pseudo-random team id; keep simple and local
        tid = f"local-{int(time.time()*1000)}-{random.randrange(1<<30):08x}"
        with TEAMS_LOCK:
            TEAMS[tid] = TeamState()
        vlog(f"registered id={tid} name={name} pl={pl}")
        return json_response(self, 200, {"id": tid})

    def handle_select(self, payload: Dict) -> None:
        tid = payload.get("id")  # optional and ignored for routing
        pname = payload.get("problemName")
        seed = payload.get("seed")  # optional int
        use_random = payload.get("random")  # optional bool
        if not isinstance(pname, str):
            return error(self, 400, "problemName is required")
        if pname not in CATALOG:
            return error(self, 400, "unknown problemName")
        if seed is not None and not isinstance(seed, int):
            return error(self, 400, "seed must be int if provided")

        with TEAMS_LOCK:
            # Ignore provided id; use a shared default state so users don't worry about ids.
            ts = TEAMS.setdefault("_default", TeamState())
            n = CATALOG[pname]
            # Choose seed: explicit seed > explicit random flag > server default
            if seed is not None:
                used_seed = seed
                mode = "fixed(seed)"
            elif isinstance(use_random, bool):
                if use_random:
                    used_seed = random.SystemRandom().randrange(1 << 63)
                    mode = "random"
                else:
                    used_seed = RNG_DEFAULT_SEED
                    mode = "fixed(default)"
            else:
                if RNG_FIXED_DEFAULT:
                    used_seed = RNG_DEFAULT_SEED
                    mode = "fixed(default)"
                else:
                    used_seed = random.SystemRandom().randrange(1 << 63)
                    mode = "random(default)"

            ts.active = generate_random_labyrinth(n, used_seed)
            ts.problem_name = pname
            ts.query_count = 0
            ts.seed = used_seed
        vlog(f"select problem={pname} rooms={CATALOG[pname]} seed={used_seed} mode={mode}")
        # Start a new session snapshot
        global CURRENT_SESSION_ID
        snap = {
            "labels": ts.active.labels[:],
            "start": ts.active.start,
            "to": ts.active.to,
        }
        with VIS_LOCK:
            sid = len(SESSIONS) + 1
            CURRENT_SESSION_ID = sid
            SESSIONS.append({
                "id": sid,
                "problemName": pname,
                "rooms": CATALOG[pname],
                "seed": used_seed,
                "mode": mode,
                "labyrinth": snap,
                "explores": [],
                "guesses": [],
            })
        add_event({"type": "select", "sessionId": sid, "problemName": pname, "rooms": CATALOG[pname], "seed": used_seed, "mode": mode})
        return json_response(self, 200, {"problemName": pname})

    def handle_explore(self, payload: Dict) -> None:
        tid = payload.get("id")  # optional and ignored
        plans = payload.get("plans")
        if not isinstance(plans, list):
            return error(self, 400, "plans are required")
        with TEAMS_LOCK:
            ts = TEAMS.setdefault("_default", TeamState())
            if ts.active is None:
                return error(self, 400, "select a problem first")

            results: List[List[int]] = []
            try:
                for p in plans:
                    if not isinstance(p, str) or any((c < '0' or c > '5') for c in p):
                        return error(self, 400, "plans must be strings of digits 0..5")
                    res = ts.active.explore_plan(p)
                    results.append(res)
                    vlog(f"explore '{p}' -> {res}")
            finally:
                # Update queryCount even on malformed plans? C++ increments only when explore succeeds.
                # We mirror C++: increment after processing the list once.
                pass

            before = ts.query_count
            ts.query_count += len(plans) + 1
            out = {"results": results, "queryCount": ts.query_count}
        if HTTP_VERBOSE:
            plan_logs = ", ".join(f"{p}->{res}" for p, res in zip(plans, results))
            sys.stderr.write(f"[http {now_ts()}] explore details=[{plan_logs}]\n")
        vlog(f"explore plans={len(plans)} qc:{before}->{out['queryCount']}")
        with VIS_LOCK:
            sid = CURRENT_SESSION_ID
            if sid is not None and 1 <= sid <= len(SESSIONS):
                SESSIONS[sid-1]["explores"].append({"plans": plans, "results": results, "queryCount": out["queryCount"]})
        add_event({"type": "explore", "sessionId": CURRENT_SESSION_ID, "plans": plans, "results": results, "queryCount": out["queryCount"]})
        return json_response(self, 200, out)

    def handle_guess(self, payload: Dict) -> None:
        tid = payload.get("id")  # optional and ignored
        map_obj = payload.get("map")
        if not isinstance(map_obj, dict):
            return error(self, 400, "map is required")
        with TEAMS_LOCK:
            ts = TEAMS.setdefault("_default", TeamState())
            if ts.active is None:
                return error(self, 400, "select a problem first")
            ok = False
            try:
                cand = build_labyrinth_from_guess(map_obj)
                ok = equivalent(ts.active, cand)
                vlog("guess map accepted structurally; checking equivalence done")
            except Exception as e:
                ok = False
                vlog(f"guess error: {e}")
            # Keep current selection for usability (no reset on guess)
            # Build diff for visualization
            diff = {}
            try:
                # labels diff
                truth_labels = ts.active.labels
                guess_labels = map_obj.get("rooms", []) if isinstance(map_obj, dict) else []
                label_mismatch = [i for i, (a, b) in enumerate(zip(truth_labels, guess_labels)) if a != b]
                # edges diff (compare undirected port pairs)
                def edge_set_from_to(to_tbl):
                    S = set()
                    for r in range(len(to_tbl)):
                        for d in range(6):
                            r2, d2 = to_tbl[r][d]
                            a = (r, d)
                            b = (r2, d2)
                            key = tuple(sorted([a, b]))
                            S.add(key)
                    return S

                truth_edges = edge_set_from_to(ts.active.to)
                # build to from guess
                try:
                    cand_full = build_labyrinth_from_guess(map_obj)
                    guess_edges = edge_set_from_to(cand_full.to)
                except Exception:
                    guess_edges = set()
                missing = sorted(list(truth_edges - guess_edges))
                extra = sorted(list(guess_edges - truth_edges))
                diff = {"labelMismatch": label_mismatch, "missingEdges": missing, "extraEdges": extra}
            except Exception:
                diff = {"error": "diff failed"}
            with VIS_LOCK:
                sid = CURRENT_SESSION_ID
                if sid is not None and 1 <= sid <= len(SESSIONS):
                    SESSIONS[sid-1]["guesses"].append({"correct": bool(ok), "diff": diff})
            add_event({"type": "guess", "sessionId": CURRENT_SESSION_ID, "correct": bool(ok), "diff": diff})
        vlog(f"guess correct={ok}")
        return json_response(self, 200, {"correct": bool(ok)})


def run(host: str = "0.0.0.0", port: int = 8009) -> None:
    server = ThreadingHTTPServer((host, port), JudgeHandler)
    print(f"LocalJudge listening on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server.server_close()


if __name__ == "__main__":
    def parse_host_port(s: str) -> Tuple[str, int]:
        if ":" in s:
            host, port_s = s.rsplit(":", 1)
            return host, int(port_s)
        return s, 8009

    parser = argparse.ArgumentParser(description="ICFPC 2025 Local Judge Server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8009, help="Bind port (default: 8009)")
    parser.add_argument("--quiet", action="store_true", help="Hide judge process logs (shown by default)")
    parser.add_argument("--verbose", action="store_true", help="Enable HTTP payload logs (off by default)")
    parser.add_argument("--client-test", nargs="?", const="127.0.0.1:8009", metavar="HOST:PORT", help="Run as client tester (default: 127.0.0.1:8009)")
    parser.add_argument("--delay-ms", type=int, default=0, help="Delay between client-test requests in milliseconds (default: 0)")
    parser.add_argument("--timeout-ms", type=int, default=600_000, help="Client-test HTTP timeout in milliseconds (default: 600000 = 10 minutes)")
    rng_group = parser.add_mutually_exclusive_group()
    rng_group.add_argument("--rng-fixed", action="store_true", help="Use fixed RNG seed by default (default)")
    rng_group.add_argument("--rng-random", action="store_true", help="Use true randomness by default")
    parser.add_argument("--seed", type=int, default=RNG_DEFAULT_SEED, help=f"Default seed when using fixed RNG (default: {RNG_DEFAULT_SEED})")
    parser.add_argument("--no-visualize", action="store_true", help="Disable visualizer UI server")
    parser.add_argument("--visualizer", type=int, default=VIZ_PORT_DEFAULT, metavar="PORT", help=f"Visualizer port (default: {VIZ_PORT_DEFAULT})")
    args = parser.parse_args()

    JUDGE_LOG = not bool(args.quiet)
    HTTP_VERBOSE = bool(args.verbose)
    # RNG defaults
    RNG_FIXED_DEFAULT = not bool(args.rng_random)
    RNG_DEFAULT_SEED = int(args.seed)
    if args.rng_fixed:
        RNG_FIXED_DEFAULT = True

    VISUALIZE = not bool(args.no_visualize)

    if args.client_test:
        target_host, target_port = parse_host_port(args.client_test)

        import http.client
        import os as _os
        _AGENT_ID = str(_os.getpid())
        _AGENT_NAME = "local_judge_server_client"
        _DELAY_SEC = max(0, int(args.delay_ms)) / 1000.0
        _TIMEOUT_SEC = max(1, int(args.timeout_ms)) / 1000.0

        def post_json(path: str, obj: Dict):
            body = json.dumps(obj)
            conn = http.client.HTTPConnection(target_host, target_port, timeout=_TIMEOUT_SEC)
            conn.request(
                "POST",
                path,
                body=body,
                headers={
                    "Content-Type": "application/json",
                    "X-Agent-ID": _AGENT_ID,
                    "X-Agent-Name": _AGENT_NAME,
                },
            )
            resp = conn.getresponse()
            raw = resp.read()
            text = raw.decode("utf-8", errors="replace")
            try:
                data = json.loads(text)
            except Exception:
                data = {"_raw": text}
            print(f"POST {path} -> {resp.status}\nreq: {obj}\nres: {data}\n")
            conn.close()
            if _DELAY_SEC > 0:
                time.sleep(_DELAY_SEC)
            return resp.status, data

        print(f"Client test against http://{target_host}:{target_port}\n")
        # 1) select with deterministic seed (id ignored server-side)
        _, sel = post_json("/select", {"id": "ignored", "problemName": "probatio", "seed": 42})
        # 2) explore a few plans
        _, ex1 = post_json("/explore", {"id": "ignored", "plans": ["0", "123", "505"]})
        # 3) submit a deliberately invalid/partial guess
        bad_guess = {
            "rooms": [0, 1, 2],
            "startingRoom": 0,
            "connections": [
                {"from": {"room": 0, "door": 0}, "to": {"room": 1, "door": 1}},
                {"from": {"room": 0, "door": 1}, "to": {"room": 2, "door": 2}},
                {"from": {"room": 0, "door": 2}, "to": {"room": 0, "door": 2}},
            ],
        }
        _, g1 = post_json("/guess", {"id": "ignored", "map": bad_guess})
        # 4) ensure reset works
        _, sel2 = post_json("/select", {"id": "ignored", "problemName": "primus", "seed": 7})
        _, ex2 = post_json("/explore", {"id": "ignored", "plans": ["012345"]})

        # 5) Solve the superdumb problem automatically
        print("\nSolving 'superdumb' problem (1 room) automatically...\n")
        _, sel3 = post_json("/select", {"id": "ignored", "problemName": "superdumb", "seed": 1})
        status, ex3 = post_json("/explore", {"id": "ignored", "plans": ["0"]})
        label = ex3.get("results", [[0]])[0][0]
        guess_superdumb = {
            "rooms": [label],
            "startingRoom": 0,
            "connections": [
                {"from": {"room": 0, "door": d}, "to": {"room": 0, "door": d}} for d in range(6)
            ],
        }
        _, g2 = post_json("/guess", {"id": "ignored", "map": guess_superdumb})
        print("Expected correct=true for superdumb.")
        print("Client test completed.")
        sys.exit(0)

    # Start judge server
    judge_thread = threading.Thread(target=run, args=(args.host, args.port), daemon=True)
    judge_thread.start()

    # Optionally start visualizer server
    if VISUALIZE:
        class VizHandler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                if self.path == "/" or self.path == "/index.html":
                    return self.serve_index_file()
                if self.path == "/state":
                    with TEAMS_LOCK:
                        ts = TEAMS.get("_default")
                        st = labyrinth_to_wire(ts.active if ts else None, ts.problem_name if ts else None, ts.seed if ts else None)
                    return json_response(self, 200, st)
                if self.path == "/sessions":
                    with VIS_LOCK:
                        summ = [
                            {"id": s["id"], "problemName": s["problemName"], "rooms": s["rooms"], "seed": s["seed"], "mode": s["mode"],
                             "explores": len(s["explores"]), "guesses": len(s["guesses"]) }
                            for s in SESSIONS
                        ]
                    return json_response(self, 200, {"sessions": summ})
                if self.path.startswith("/session"):
                    from urllib.parse import urlparse, parse_qs
                    q = parse_qs(urlparse(self.path).query)
                    sid = int(q.get("id", [0])[0]) if q.get("id") else 0
                    with VIS_LOCK:
                        if 1 <= sid <= len(SESSIONS):
                            s = SESSIONS[sid-1]
                            detail = {
                                "id": s["id"],
                                "problemName": s["problemName"],
                                "rooms": s["rooms"],
                                "seed": s["seed"],
                                "mode": s["mode"],
                                "labyrinth": s["labyrinth"],
                                "explores": s["explores"],
                                "guesses": s["guesses"],
                            }
                        else:
                            detail = {"error": "unknown session id"}
                    return json_response(self, 200, detail)
                if self.path == "/events":
                    with VIS_LOCK:
                        evs = list(VIS_EVENTS)
                    return json_response(self, 200, {"events": evs, "logs": list(VIS_LOG_BUFFER[-200:])})
                if self.path == "/stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "keep-alive")
                    self.end_headers()
                    lock = threading.Lock()
                    with VIS_LOCK:
                        SSE_CLIENTS.append((self, lock))
                    try:
                        while True:
                            time.sleep(60)
                    except Exception:
                        pass
                    finally:
                        with VIS_LOCK:
                            try:
                                SSE_CLIENTS.remove((self, lock))
                            except ValueError:
                                pass
                    return
                return error(self, 404, "Not Found")

            def log_message(self, fmt: str, *args) -> None:  # keep quiet
                if HTTP_VERBOSE:
                    sys.stderr.write("[viz] " + (fmt % args) + "\n")

            def serve_index_file(self):
                import os
                here = os.path.dirname(os.path.abspath(__file__))
                path = os.path.join(here, "local_judge_server.html")
                try:
                    with open(path, "rb") as f:
                        data = f.read()
                except Exception:
                    # Fallback to inline template if the external file is missing
                    return error(self, 500, "HTML file not found")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        viz_host = args.host
        viz_port = int(args.visualizer)
        viz_server = ThreadingHTTPServer((viz_host, viz_port), VizHandler)
        print(f"Visualizer at http://{viz_host}:{viz_port}")
        webbrowser.open(f"http://{viz_host}:{viz_port}")
        try:
            viz_server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            viz_server.server_close()
    else:
        # Block main thread on judge
        judge_thread.join()
