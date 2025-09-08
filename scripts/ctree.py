# -*- coding: utf-8 -*-
"""
ICFP Programming Contest 2025 / Ædificium map learner (Classification Tree variant)
- Batches all /explore queries to minimize queryCount.
- Builds an equivalent Moore machine (rooms, doors 0..5) and emits /guess JSON.
- Supports the post-lightning addendum (charcoal marks "[d]" with d in 0..3).

Usage:
  1) Set ICFP_ID in env if you already registered, or call client.register(...)
  2) Run main() to solve a selected problem (see PROBLEM_NAME).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set
import os
import json
import random
import requests
import sys
import datetime
import argparse

# Default base URLs per target
LOCAL_URL = "http://127.0.0.1:8009"
MINOTAUR_URL = "http://tk2-401-41624.vs.sakura.ne.jp:19384"
ALPHABET = "012345"  # door labels
GRAFFITI_DIGITS = "0123"  # allowed label overrides in [d]

# ---------- API Client ----------


class AedificiumClient:
    """Ædificium contest API client."""

    def __init__(
        self,
        base_url: str,
        team_id: Optional[str] = None,
        verbose: bool = False,
        agent_name: str = "tsuzuki:ctree",
        agent_id: Optional[str] = None,
        timeout_sec: Optional[float] = 6000,
    ):
        self.base_url = base_url.rstrip("/")
        self.id = team_id
        self.verbose = verbose
        # Per instructions: include Minotaur headers on all POSTs.
        self.agent_name = agent_name
        self.agent_id = agent_id or str(os.getpid())
        # Long timeout for potentially blocking endpoints like /select
        self.timeout_sec = timeout_sec

    def _post(self, path: str, payload: Dict) -> Dict:
        url = f"{self.base_url}{path}"
        headers = {
            "Content-Type": "application/json",
            "X-Agent-Name": self.agent_name,
            "X-Agent-ID": self.agent_id,
        }
        if self.verbose:
            print(f"POST {url}")
            print("Headers:", headers)
            print("Payload:", json.dumps(payload, ensure_ascii=False))
        r = requests.post(url, json=payload, headers=headers, timeout=self.timeout_sec)
        if self.verbose:
            try:
                print("Response:", r.status_code, r.json())
            except Exception:
                print("Response:", r.status_code, r.text)
        r.raise_for_status()
        try:
            return r.json()
        except Exception:
            return {"_raw": r.text}

    def register(self, name: str, pl: str, email: str) -> str:
        data = {"name": name, "pl": pl, "email": email}
        j = self._post("/register", data)
        self.id = j["id"]
        return self.id

    def select_problem(self, problem_name: str) -> str:
        payload = {"problemName": problem_name}
        if self.id is not None:
            payload["id"] = self.id
        j = self._post("/select", payload)
        return j["problemName"]

    def explore(self, plans: List[str]) -> Tuple[List[List[int]], int]:
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
        """Minotaur-only auxiliary endpoint to voluntarily give up a run.
        Absent on the official server; safe to ignore failures.
        """
        try:
            return self._post("/minotaur_giveup", {})
        except Exception as _:
            return {"ok": False}


# ---------- Membership Oracle with batching & caching ----------


class ExploreOracle:
    """
    Batches /explore and caches full traces per plan (digits 0..5 plus optional "[d]" tokens).

    For a plan w that contains 0..5 and optional graffiti tokens like "[3]":
    - The server returns a trace of labels of length (#door-steps + 1).
    - We take the last element as g(w) (the Moore output after w).
    - The initial state's label is the 0-th element of any returned trace.
    """

    def __init__(self, client: AedificiumClient):
        self.client = client
        self.trace_cache: Dict[str, List[int]] = {}  # plan -> full trace (len = |doors|+1)
        self.last_label: Dict[str, int] = {}  # plan -> final label
        self.pending: List[str] = []
        self.init_label: Optional[int] = None
        self.last_query_count: int = 0
        self.last_query_count_output_datetime = datetime.datetime.now()
        self.verbose: bool = getattr(client, "verbose", False)

    @staticmethod
    def _doors_len(plan: str) -> int:
        """Count door-steps (digits 0..5) in the plan (graffiti does not count)."""
        return sum(1 for ch in plan if ch in ALPHABET)

    def ensure(self, words: Set[str]) -> None:
        """Queue all plans to batch request via /explore."""
        for w in words:
            if w == "" or self._doors_len(w) == 0:
                # Never submit a plan with zero door-steps; treat as initial.
                if w not in self.last_label:
                    self.last_label[w] = self.g("")  # may raise until prewarmed
                continue
            if w not in self.trace_cache and w not in self.pending:
                self.pending.append(w)

    def commit(self) -> None:
        if not self.pending:
            return
        plans = list(dict.fromkeys(self.pending))  # dedupe, keep order
        self.pending.clear()
        if self.verbose:
            door_lens = [self._doors_len(p) for p in plans]
            print(
                f"/explore commit: {len(plans)} plans, door-steps min/max/avg = "
                f"{min(door_lens) if door_lens else 0}/"
                f"{max(door_lens) if door_lens else 0}/"
                f"{(sum(door_lens)/len(door_lens)) if door_lens else 0:.2f}"
            )
            if len(plans) <= 10:
                print("Plans:", plans)
        results, _qc = self.client.explore(plans)
        # soft-throttle printing
        if (
            self.last_query_count_output_datetime + datetime.timedelta(seconds=1.0)
            < datetime.datetime.now()
        ):
            self.last_query_count_output_datetime = (
                datetime.datetime.now() + datetime.timedelta(seconds=1.0)
            )
            print(f"{_qc=}")
        self.last_query_count = _qc
        assert len(results) == len(plans)
        for w, trace in zip(plans, results):
            self.trace_cache[w] = trace
            if len(trace) == 0:
                raise RuntimeError("Server returned empty trace for plan: %r" % w)
            self.last_label[w] = trace[-1]
            if self.init_label is None and len(trace) >= 1:
                self.init_label = trace[0]

    def g(self, w: str) -> int:
        """Return final label for plan w ("" denotes initial label)."""
        if w == "":
            if self.init_label is not None:
                return self.init_label
            # derive from any existing cached trace or prime the oracle
            for trace in self.trace_cache.values():
                if trace:
                    self.init_label = trace[0]
                    break
            if self.init_label is None:
                self.ensure({"0"})
                self.commit()
                if self.init_label is None:
                    for trace in self.trace_cache.values():
                        if trace:
                            self.init_label = trace[0]
                            break
                if self.init_label is None:
                    raise RuntimeError("Initial label is not available yet.")
            return self.init_label
        if w not in self.last_label:
            self.ensure({w})
            self.commit()
        return self.last_label[w]

    def trace(self, w: str) -> List[int]:
        """Return full trace for plan w (batches if necessary)."""
        if w == "":
            return [self.g("")]
        if w not in self.trace_cache:
            self.ensure({w})
            self.commit()
        return self.trace_cache[w]


# ---------- Classification Tree (multi-ary outcomes 0..3) ----------


class CTNode:
    __slots__ = ("exp", "children", "parent", "rep_word", "state_index")

    def __init__(self, exp: Optional[str], parent: Optional["CTNode"] = None):
        # exp is the suffix experiment; None denotes a leaf
        self.exp: Optional[str] = exp
        # mapping from observation (0..3) to child node
        self.children: Dict[int, "CTNode"] = {}
        self.parent: Optional["CTNode"] = parent
        # leaf-only fields
        self.rep_word: Optional[str] = None
        self.state_index: Optional[int] = None

    def is_leaf(self) -> bool:
        return self.exp is None

    def child(self, obs: int) -> "CTNode":
        if obs not in self.children:
            self.children[obs] = CTNode(None, parent=self)
        return self.children[obs]

    def replace_with_test(self, exp: str) -> "CTNode":
        """Convert this leaf into an internal node testing 'exp'."""
        assert self.is_leaf(), "replace_with_test must be called on a leaf"
        self.exp = exp
        self.rep_word = None
        self.state_index = None
        self.children = {}
        return self


@dataclass
class Hypothesis:
    rep_for_state: List[str]           # representative access word for each state
    outputs: List[int]                 # Moore output per state (label 0..3)
    delta: List[List[int]]             # transitions [state][door] -> state


class ClassTreeMooreLearner:
    """
    Moore machine learner using a basic classification tree (Rivest-Schapire style),
    extended with simple graffiti-based discriminators for the addendum problems.
    """

    def __init__(
        self,
        oracle: ExploreOracle,
        random_tests: int = 200,
        max_test_len: int = 8,
        use_graffiti: bool = True,
    ):
        self.oracle = oracle
        self.root = CTNode(exp="")  # test "" (i.e. observe current label)
        self.states: List[CTNode] = []  # leaves that have representatives
        self.random_tests = random_tests
        self.max_test_len = max_test_len
        self.use_graffiti = use_graffiti
        # Warm-up to get initial label and server wake-up
        self.oracle.ensure({"0"})
        self.oracle.commit()

    # ---- classification ----

    def _observe(self, u: str, exp: str) -> int:
        """Observation outcome for word u w.r.t. experiment exp (0..3)."""
        return self.oracle.g(u + exp)

    def classify(self, u: str) -> CTNode:
        """Traverse the classification tree using observations for u; returns a leaf."""
        node = self.root
        while not node.is_leaf():
            assert node.exp is not None
            o = self._observe(u, node.exp)
            node = node.child(o)
        return node

    def ensure_state(self, u: str) -> int:
        """
        Ensure the leaf reached by u has a representative; return its state index.
        If the leaf has no rep yet, assign u as the rep and create a new state.
        """
        leaf = self.classify(u)
        if leaf.state_index is None:
            leaf.rep_word = u
            leaf.state_index = len(self.states)
            self.states.append(leaf)
        return leaf.state_index

    # ---- utilities over the tree ----

    def _path_to_leaf(self, u: str) -> List[Tuple[CTNode, int]]:
        """Return list of (internal_node, obs) pairs on the path when classifying u."""
        path: List[Tuple[CTNode, int]] = []
        node = self.root
        while not node.is_leaf():
            assert node.exp is not None
            o = self._observe(u, node.exp)
            path.append((node, o))
            node = node.child(o)
        return path

    def _discriminator_between_leaves(self, leaf1: CTNode, leaf2: CTNode) -> str:
        """Return the experiment exp at the LCA where leaf1 and leaf2 diverge."""
        assert leaf1.rep_word is not None and leaf2.rep_word is not None
        path1 = self._path_to_leaf(leaf1.rep_word)
        path2 = self._path_to_leaf(leaf2.rep_word)
        L = min(len(path1), len(path2))
        last_same_idx = -1
        for i in range(L):
            if path1[i][0] is path2[i][0] and path1[i][1] == path2[i][1]:
                last_same_idx = i
            else:
                break
        next_idx = last_same_idx + 1
        node = path1[next_idx][0] if next_idx < len(path1) else path2[next_idx][0]
        assert node.exp is not None
        return node.exp

    def _insert_test_at_leaf(self, leaf: CTNode, exp: str, words_to_redistribute: List[str]) -> None:
        """
        Replace the given leaf by an internal test node 'exp' and redistribute
        the given words (which previously classified to this leaf) into its children.
        """
        if self.oracle.verbose:
            print(f"Refine: insert test exp='{exp}' (doors={ExploreOracle._doors_len(exp)})")
        leaf.replace_with_test(exp)
        buckets: Dict[int, List[str]] = {}
        for w in words_to_redistribute:
            o = self._observe(w, exp)
            buckets.setdefault(o, []).append(w)
        for o, group in buckets.items():
            child = leaf.child(o)
            reps = [w for w in group if any(s.rep_word == w for s in self.states)]
            chosen = reps[0] if reps else group[0]
            child.rep_word = chosen
            child.state_index = None  # indices will be (re)assigned as needed

        def _invalidate(node: CTNode):
            if node.is_leaf():
                node.state_index = None
            else:
                for ch in node.children.values():
                    _invalidate(ch)
        _invalidate(leaf)

    # ---- main learning loop ----

    def learn(self) -> Hypothesis:
        # Initialize with start state as representative
        self.ensure_state("")
        while True:
            # Expand transitions and discover new states
            self._expand_states_once()
            # Refine until consistent
            if self._refine_once():
                continue
            # Consistent: make hypothesis
            hyp = self._make_hypothesis()
            # Validate with random tests; strengthen if needed
            if self._strengthen_with_random_counterexample(hyp):
                continue
            return hyp

    def _expand_states_once(self) -> None:
        """For each known state, ensure successors exist (may create new states)."""
        need: Set[str] = set()
        all_exps = self._collect_all_experiments()
        for leaf in self.states:
            if leaf.rep_word is None:
                continue
            for a in ALPHABET:
                u = leaf.rep_word + a
                for e in all_exps:
                    plan = u + e
                    if ExploreOracle._doors_len(plan) > 0:
                        need.add(plan)
        self.oracle.ensure(need)
        self.oracle.commit()
        for leaf in self.states:
            if leaf.rep_word is None:
                continue
            for a in ALPHABET:
                self.ensure_state(leaf.rep_word + a)

    def _collect_all_experiments(self) -> List[str]:
        """Collect all exp strings from internal nodes; ensure "" present."""
        exps: List[str] = []
        stack = [self.root]
        seen = set()
        while stack:
            node = stack.pop()
            if node.exp is not None and node.exp not in seen:
                exps.append(node.exp)
                seen.add(node.exp)
            for ch in node.children.values():
                stack.append(ch)
        if "" not in seen:
            exps.append("")
        return exps

    def _words_in_same_leaf(self) -> Dict[CTNode, List[str]]:
        """Partition interesting words (S ∪ SΣ) by the leaf they classify to."""
        words: List[str] = []
        for leaf in self.states:
            if leaf.rep_word is None:
                continue
            words.append(leaf.rep_word)
            for a in ALPHABET:
                words.append(leaf.rep_word + a)
        # Batch observations for all words across all experiments
        all_exps = self._collect_all_experiments()
        need: Set[str] = set()
        for u in words:
            for e in all_exps:
                plan = u + e
                if ExploreOracle._doors_len(plan) > 0:
                    need.add(plan)
        self.oracle.ensure(need)
        self.oracle.commit()
        # Classify
        buckets: Dict[CTNode, List[str]] = {}
        for u in words:
            leaf = self.classify(u)
            buckets.setdefault(leaf, []).append(u)
        return buckets

    def _refine_once(self) -> bool:
        """
        Check for an inconsistency. If found, refine by inserting a new test at a leaf.
        Returns True if refined.
        """
        buckets = self._words_in_same_leaf()
        for leaf, members in buckets.items():
            if len(members) <= 1:
                continue
            # Look for single-letter successor mismatch
            for a in ALPHABET:
                classes = {}
                for u in members:
                    leaf_succ = self.classify(u + a)
                    classes.setdefault(leaf_succ, []).append(u)
                if len(classes) > 1:
                    (leafA, groupA), (leafB, groupB) = list(classes.items())[:2]
                    u, v = groupA[0], groupB[0]
                    exp0 = self._discriminator_between_leaves(leafA, leafB)
                    new_exp = a + exp0
                    self._insert_test_at_leaf(leaf, new_exp, members)
                    return True
            # No single-letter mismatch; try a short graffiti-based discriminator
            if self.use_graffiti:
                for i in range(len(members)):
                    for j in range(i + 1, len(members)):
                        u, v = members[i], members[j]
                        new_exp = self._synthesize_graffiti_discriminator(u, v)
                        if new_exp is not None:
                            self._insert_test_at_leaf(leaf, new_exp, members)
                            return True
        return False

    def _synthesize_graffiti_discriminator(self, u: str, v: str) -> Optional[str]:
        """
        Try to find e such that g(u+e) != g(v+e) using short graffiti patterns.
        Returns such an e if found, otherwise None.

        Heuristics:
          1) length-1 and length-2 pure door suffixes (cheap sanity check)
          2) "[d]ab" for d in 0..3 and a,b in 0..5 (mark & bounce)
          3) "[d]a" (rarely useful but cheap)
        """
        candidates: List[str] = []
        # 1) Pure door suffixes length 1 and 2
        for a in ALPHABET:
            candidates.append(a)
        for a in ALPHABET:
            for b in ALPHABET:
                candidates.append(a + b)
        # 2) Mark & bounce
        for d in GRAFFITI_DIGITS:
            for a in ALPHABET:
                for b in ALPHABET:
                    candidates.append(f"[{d}]{a}{b}")
        # 3) Mark-only then move one step
        for d in GRAFFITI_DIGITS:
            for a in ALPHABET:
                candidates.append(f"[{d}]{a}")
        # Batch and test
        need: Set[str] = set()
        for e in candidates:
            if ExploreOracle._doors_len(e) == 0:
                continue
            need.add(u + e)
            need.add(v + e)
        self.oracle.ensure(need)
        self.oracle.commit()
        for e in candidates:
            if ExploreOracle._doors_len(e) == 0:
                continue
            gu = self.oracle.g(u + e)
            gv = self.oracle.g(v + e)
            if gu != gv:
                return e
        return None

    def _make_hypothesis(self) -> Hypothesis:
        # Assign indices (ensure all reps are current leaves)
        for leaf in self.states:
            if leaf.rep_word is not None:
                self.ensure_state(leaf.rep_word)
        # Build outputs & reps in index order
        idx_to_rep: Dict[int, str] = {}
        idx_to_out: Dict[int, int] = {}
        for leaf in self.states:
            if leaf.state_index is not None and leaf.rep_word is not None:
                idx_to_rep[leaf.state_index] = leaf.rep_word
                idx_to_out[leaf.state_index] = self.oracle.g(leaf.rep_word)
        if not idx_to_rep:
            raise RuntimeError("No states learned.")
        max_idx = max(idx_to_rep)
        rep_for_state = [idx_to_rep[i] for i in range(max_idx + 1)]
        outputs = [idx_to_out[i] for i in range(max_idx + 1)]
        # Build delta
        n = len(rep_for_state)
        delta: List[List[int]] = [[0] * 6 for _ in range(n)]
        for i, rep in enumerate(rep_for_state):
            for d in range(6):
                u = rep + str(d)
                j = self.ensure_state(u)
                leaf_u = self.classify(u)
                if leaf_u.state_index is None:
                    leaf_u.state_index = j
                delta[i][d] = leaf_u.state_index
        hyp = Hypothesis(rep_for_state=rep_for_state, outputs=outputs, delta=delta)
        if self.oracle.verbose:
            print(f"Hypothesis built: states={len(rep_for_state)}")
        return hyp

    def _strengthen_with_random_counterexample(self, hyp: Hypothesis) -> bool:
        """
        Run random tests; if any prediction mismatches the oracle,
        refine the tree using a synthesized discriminator at the offending leaf.
        """
        def predict(word: str) -> int:
            s = 0
            for ch in word:
                s = hyp.delta[s][int(ch)]
            return hyp.outputs[s]

        # Candidate tests: small BFS + random
        words: Set[str] = set()
        frontier = [""]
        for _ in range(4):
            new_frontier = []
            for u in frontier:
                for a in ALPHABET:
                    w = u + a
                    words.add(w)
                    new_frontier.append(w)
            frontier = new_frontier
        for _ in range(200):
            L = random.randint(1, 8)
            words.add("".join(random.choice(ALPHABET) for _ in range(L)))
        self.oracle.ensure(words)
        self.oracle.commit()

        for w in words:
            pred = predict(w)
            real = self.oracle.g(w)
            if pred != real:
                # Try to refine using the earliest step in w
                buckets = self._words_in_same_leaf()
                for i in range(len(w)):
                    prefix = w[:i]
                    a = w[i]
                    leaf = self.classify(prefix)
                    members = buckets.get(leaf, [prefix])
                    # Prefer discriminator from successor disagreement if available
                    classes = {}
                    for u in members:
                        classes.setdefault(self.classify(u + a), []).append(u)
                    if len(classes) > 1:
                        (leafA, groupA), (leafB, groupB) = list(classes.items())[:2]
                        exp0 = self._discriminator_between_leaves(leafA, leafB)
                        self._insert_test_at_leaf(leaf, a + exp0, members)
                        return True
                    # Else try graffiti separator
                    if self.use_graffiti and len(members) >= 2:
                        new_exp = self._synthesize_graffiti_discriminator(members[0], members[1])
                        if new_exp is not None:
                            self._insert_test_at_leaf(leaf, new_exp, members)
                            return True
                return False
        return False


# ---------- Build /guess JSON from hypothesis ----------


def build_guess_from_hypothesis(hyp: Hypothesis) -> Dict:
    """
    Construct the map structure for /guess from a learned hypothesis.
    """
    n = len(hyp.outputs)
    rooms = hyp.outputs[:]  # 2-bit labels per room
    starting_room = 0  # state of "" by construction

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
            if partner is None:
                for b in range(6):
                    if hyp.delta[r2][b] == r:
                        partner = b
                        break
            if partner is None:
                partner = a if r == r2 else 0

            connections.append(
                {"from": {"room": r, "door": a}, "to": {"room": r2, "door": partner}}
            )
            used[r][a] = True
            used[r2][partner] = True

    return {"rooms": rooms, "startingRoom": starting_room, "connections": connections}


# ---------- Example runner ----------


def main():
    """
    End-to-end: select problem, learn, and submit guess.
    """
    parser = argparse.ArgumentParser(description="ICFP 2025 Classification Tree solver")
    parser.add_argument(
        "--target",
        choices=["local", "minotaur"],
        default="local",
        help="API target to use (default: local)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print HTTP request/response details",
    )
    parser.add_argument(
        "--agent-name",
        default="anonymous:ctree",
        help="Sets X-Agent-Name (default: anonymous:ctree)",
    )
    parser.add_argument(
        "--plan-length",
        choices=["6n", "18n"],
        default="6n",
        help="Constraint mode for door-steps in plans (default: 6n)",
    )
    parser.add_argument(
        "--giveup",
        action="store_true",
        help="When targeting Minotaur, call /minotaur_giveup and exit",
    )
    args = parser.parse_args()

    # Choose base URL by target
    if args.target == "minotaur":
        base_url = MINOTAUR_URL
    else:
        base_url = LOCAL_URL

    # Get ID from env or register once (then set ICFP_ID env for later runs)
    team_id = os.getenv("ICFP_ID")
    client = AedificiumClient(
        base_url=base_url,
        team_id=team_id,
        verbose=args.verbose,
        agent_name=args.agent_name,
        agent_id=str(os.getpid()),
        timeout_sec=6000,
    )

    # Startup diagnostics
    print("=== Aedificium DT Runner ===")
    print("Target:", args.target, "->", client.base_url)
    print("Agent:", client.agent_name, "(id:", client.agent_id + ")")
    print("Team ID set:", bool(client.id))
    print("Plan Length Mode:", args.plan_length)
    print("Verbose:", args.verbose)
    try:
        import platform
        import requests as _rq
        print("Python:", platform.python_version())
        print("Requests:", _rq.__version__)
    except Exception:
        pass

    # Optional: allow manual giveup against Minotaur
    if args.giveup:
        res = client.minotaur_giveup()
        print("minotaur_giveup:", res)
        return

    # Select a problem (post-lightning addendum enabled)
    # Try a small problem first when debugging:
    # PROBLEM_NAME = "probatio"
    # Lightning set: primus..quintus
    # Post-lightning set: aleph, beth, gimel, ...
    PROBLEM_NAME = os.getenv("ICFP_PROBLEM", "primus")
    chosen = client.select_problem(PROBLEM_NAME)
    print("Selected:", chosen)

    oracle = ExploreOracle(client)
    learner = ClassTreeMooreLearner(oracle, use_graffiti=True)

    hyp = learner.learn()
    guess_map = build_guess_from_hypothesis(hyp)

    print(json.dumps(guess_map, indent=2))

    try:
        ok = client.guess(guess_map)
        print("Guess correct?", ok)
    except requests.HTTPError as e:
        print("HTTP error during /guess:", e)
    except Exception as ex:
        print("Error during /guess:", ex)


if __name__ == "__main__":
    try:
        main()
    except requests.HTTPError as e:
        print("HTTP error:", e)
    except Exception as ex:
        print("Error:", ex)
