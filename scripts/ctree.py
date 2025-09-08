# -*- coding: utf-8 -*-
"""
ICFP Programming Contest 2025 / Ædificium map learner (Classification Tree variant)
- Batches all /explore queries to minimize queryCount.
- Builds an equivalent Moore machine (rooms, doors 0..5) and emits /guess JSON.
- Supports the post-lightning addendum (charcoal marks "[d]" with d in 0..3).

"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set, Any
import os
import random
import datetime
import sys
ALPHABET = "012345"  # door labels
GRAFFITI_DIGITS = "0123"  # allowed label overrides in [d]

# ---------- Solver-facing Oracle and Learner ----------

# ---------- Membership Oracle with batching & caching ----------


class ExploreOracle:
    """
    Batches /explore and caches full traces per plan (digits 0..5 plus optional "[d]" tokens).

    For a plan w that contains 0..5 and optional graffiti tokens like "[3]":
    - The server returns a trace of labels of length (#door-steps + 1).
    - We take the last element as g(w) (the Moore output after w).
    - The initial state's label is the 0-th element of any returned trace.
    """

    def __init__(self, client: Any, plan_length_mode: str = "6n", fixed_n: Optional[int] = None, status: Optional[Any] = None):
        self.client = client
        self.trace_cache: Dict[str, List[int]] = {}  # plan -> full trace (len = |doors|+1)
        self.last_label: Dict[str, int] = {}  # plan -> final label
        self.pending: List[str] = []
        self.init_label: Optional[int] = None
        self.last_query_count: int = 0
        self.last_query_count_output_datetime = datetime.datetime.now()
        self.verbose: bool = getattr(client, "verbose", False)
        # plan-length enforcement
        self.plan_length_mode: str = plan_length_mode  # "6n" or "18n"
        self._state_count: int = 1
        self._fixed_n: Optional[int] = fixed_n
        self._deferred: Set[str] = set()
        self.status = status

    def set_state_count(self, n: int) -> None:
        # If fixed_n is provided (problem size known), ignore dynamic updates
        if self._fixed_n is not None:
            return
        n = max(1, int(n))
        if n != self._state_count:
            self._state_count = n
            if self.verbose:
                print(f"[budget] n={n}, budget={self._budget()} (mode={self.plan_length_mode})")
            # Try to re-queue deferred plans now within budget
            if self._deferred:
                now_ok = [p for p in list(self._deferred) if self._doors_len(p) <= self._budget()]
                if now_ok:
                    for p in now_ok:
                        self._deferred.discard(p)
                    if self.verbose:
                        print(f"[budget] resurrect {len(now_ok)} deferred plans")
                    # Put them at the front to be executed soon
                    self.pending[:0] = now_ok

    def _budget(self) -> int:
        mult = 6 if self.plan_length_mode == "6n" else 18
        n = self._fixed_n if self._fixed_n is not None else self._state_count
        return mult * max(1, n)

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
            if w in self.trace_cache or w in self.pending:
                continue
            if self._doors_len(w) <= self._budget():
                self.pending.append(w)
            else:
                # Defer overlong plan; try again when n grows
                self._deferred.add(w)
        if self.verbose and self._deferred:
            print(f"[budget] deferred plans: {len(self._deferred)} (budget={self._budget()})")

    def commit(self) -> None:
        if not self.pending:
            return
        plans = list(dict.fromkeys(self.pending))  # dedupe, keep order
        self.pending.clear()
        if self.verbose and self.status is None:
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
        # Update status board with explore summary if available
        if self.status is not None:
            try:
                door_lens = [self._doors_len(p) for p in plans]
                doors_min = min(door_lens) if door_lens else 0
                doors_max = max(door_lens) if door_lens else 0
                doors_avg = (sum(door_lens)/len(door_lens)) if door_lens else 0.0
                self.status.update_explore(
                    plans_count=len(plans),
                    doors_min=doors_min,
                    doors_max=doors_max,
                    doors_avg=doors_avg,
                    qc_last=_qc,
                    pending=len(self.pending),
                    deferred=len(self._deferred),
                )
                self.status.render()
            except Exception:
                pass
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
        status: Optional[Any] = None,
    ):
        self.oracle = oracle
        self.root = CTNode(exp="")  # test "" (i.e. observe current label)
        self.states: List[CTNode] = []  # leaves that have representatives
        self.random_tests = random_tests
        self.max_test_len = max_test_len
        self.use_graffiti = use_graffiti
        self.status = status
        # Warm-up to get initial label and server wake-up
        self.oracle.ensure({"0"})
        self.oracle.set_state_count(len(self.states) if self.states else 1)
        self.oracle.commit()
        # Register a solver-specific renderer with the status board if available
        if self.status is not None and hasattr(self.status, "set_solver_renderer"):
            def _solver_lines() -> List[str]:
                lines: List[str] = []
                try:
                    states = len(self.states)
                    leaves = sum(1 for s in self.states if s.is_leaf())
                    exps = len(self._collect_all_experiments())
                    lines.append(f"Learned states   : {states} (leaf reps {leaves})")
                    lines.append(f"Active tests     : {exps} discriminator experiments")
                    if hasattr(self.oracle, "_budget"):
                        try:
                            budget = self.oracle._budget()
                            lines.append(f"Plan budget      : {budget} max door-steps")
                        except Exception:
                            pass
                    if hasattr(self.oracle, "_deferred"):
                        try:
                            deferred = len(self.oracle._deferred)
                            pend = len(self.oracle.pending)
                            lines.append(f"Pending/Deferred : {pend} queued / {deferred} over-budget")
                        except Exception:
                            pass
                    if hasattr(self, "last_hyp_states"):
                        lines.append(f"Hypothesis size  : {getattr(self, 'last_hyp_states')} states")
                except Exception:
                    pass
                return lines
            try:
                self.status.set_solver_renderer(_solver_lines)
            except Exception:
                pass

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
            # Update budget estimate when a new state is confirmed
            self.oracle.set_state_count(len(self.states))
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
        if self.status is not None:
            try:
                self.status.bump_refinements()
                self.status.render()
            except Exception:
                pass
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
        if self.status is not None:
            try:
                self.status.update_solver(states=len(self.states), exps=len(self._collect_all_experiments()), leaves=sum(1 for s in self.states if s.is_leaf()))
                self.status.render()
            except Exception:
                pass
        while True:
            # Expand transitions and discover new states
            self._expand_states_once()
            if self.status is not None:
                try:
                    self.status.update_solver(states=len(self.states), exps=len(self._collect_all_experiments()), leaves=sum(1 for s in self.states if s.is_leaf()))
                    self.status.render()
                except Exception:
                    pass
            # Refine until consistent
            if self._refine_once():
                if self.status is not None:
                    try:
                        self.status.update_solver(states=len(self.states), exps=len(self._collect_all_experiments()), leaves=sum(1 for s in self.states if s.is_leaf()))
                        self.status.render()
                    except Exception:
                        pass
                continue
            # Consistent: make hypothesis
            hyp = self._make_hypothesis()
            if self.status is not None:
                try:
                    self.status.set_hyp_states(len(hyp.outputs))
                    self.status.render()
                except Exception:
                    pass
            # If problem size is known and we already have exactly that many states,
            # skip random strengthening to save queries on small problems.
            try:
                fixed_n = getattr(self.oracle, "_fixed_n", None)
                if fixed_n is not None and len(hyp.outputs) >= fixed_n:
                    # no console print here; runner/status board handles output
                    return hyp
            except Exception:
                pass
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
        # update budget before queueing
        self.oracle.set_state_count(len(self.states) if self.states else 1)
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
        self.oracle.set_state_count(len(self.states) if self.states else 1)
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

        Heuristics (staged to keep query volume low):
          Stage 1) a in 0..5 and "[d]a"
          Stage 2) ab in 0..5^2
          Stage 3) "[d]ab" (mark & bounce)
        """
        # Build stages
        stages: List[List[str]] = []
        s1: List[str] = []
        s1.extend([a for a in ALPHABET])
        for d in GRAFFITI_DIGITS:
            for a in ALPHABET:
                s1.append(f"[{d}]{a}")
        stages.append(s1)
        s2: List[str] = []
        for a in ALPHABET:
            for b in ALPHABET:
                s2.append(a + b)
        stages.append(s2)
        s3: List[str] = []
        for d in GRAFFITI_DIGITS:
            for a in ALPHABET:
                for b in ALPHABET:
                    s3.append(f"[{d}]{a}{b}")
        stages.append(s3)

        for cand in stages:
            need: Set[str] = set()
            for e in cand:
                if ExploreOracle._doors_len(e) == 0:
                    continue
                if ExploreOracle._doors_len(u + e) <= self.oracle._budget():
                    need.add(u + e)
                if ExploreOracle._doors_len(v + e) <= self.oracle._budget():
                    need.add(v + e)
            if not need:
                continue
            self.oracle.ensure(need)
            self.oracle.commit()
            for e in cand:
                if ExploreOracle._doors_len(e) == 0:
                    continue
                if ExploreOracle._doors_len(u + e) > self.oracle._budget():
                    continue
                if ExploreOracle._doors_len(v + e) > self.oracle._budget():
                    continue
                gu = self.oracle.g(u + e)
                gv = self.oracle.g(v + e)
                if gu != gv:
                    return e
        return None

    def _make_hypothesis(self) -> Hypothesis:
        # Reindex states from current leaves to ensure contiguous indices starting at 0.
        # 1) Collect all current leaves that have representatives
        rep_leaves: List[CTNode] = []
        seen_nodes: Set[int] = set()
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.is_leaf():
                if node.rep_word is not None and id(node) not in seen_nodes:
                    rep_leaves.append(node)
                    seen_nodes.add(id(node))
            else:
                for ch in node.children.values():
                    stack.append(ch)

        # Ensure the start state's leaf exists and is first
        start_leaf = self.classify("")
        if start_leaf.rep_word is None:
            start_leaf.rep_word = ""
        # Clear all indices
        for ln in rep_leaves:
            ln.state_index = None
        start_leaf.state_index = 0

        ordered_nodes: List[CTNode] = [start_leaf]
        for ln in rep_leaves:
            if ln is start_leaf:
                continue
            ln.state_index = len(ordered_nodes)
            ordered_nodes.append(ln)

        # Build reps and outputs dynamically and guarantee closure under Σ
        rep_for_state: List[str] = [node.rep_word or "" for node in ordered_nodes]
        outputs: List[int] = [self.oracle.g(word) for word in rep_for_state]
        delta: List[List[int]] = [[0] * 6 for _ in range(len(ordered_nodes))]

        i = 0
        while i < len(ordered_nodes):
            rep = ordered_nodes[i].rep_word or ""
            # Prefetch successors for this representative across all current experiments
            all_exps = self._collect_all_experiments()
            need: Set[str] = set()
            for d in range(6):
                u = rep + str(d)
                for e in all_exps:
                    plan = u + e
                    if ExploreOracle._doors_len(plan) > 0:
                        need.add(plan)
            if need:
                self.oracle.ensure(need)
                self.oracle.commit()
            for d in range(6):
                u = rep + str(d)
                leaf_u = self.classify(u)
                if leaf_u.state_index is None:
                    # New state discovered; assign next index
                    leaf_u.state_index = len(ordered_nodes)
                    ordered_nodes.append(leaf_u)
                    rep_for_state.append(leaf_u.rep_word or u)
                    outputs.append(self.oracle.g(rep_for_state[-1]))
                    # expand delta table for the new state
                    delta.append([0] * 6)
                delta[i][d] = leaf_u.state_index
            i += 1

        hyp = Hypothesis(rep_for_state=rep_for_state, outputs=outputs, delta=delta)
        # Track for status renderer
        try:
            self.last_hyp_states = len(rep_for_state)
        except Exception:
            pass
        if self.oracle.verbose:
            print(f"Hypothesis built: states={len(rep_for_state)}")
        # Keep self.states in sync with the current indexed leaves
        self.states = ordered_nodes
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
                    if ExploreOracle._doors_len(w) <= self.oracle._budget():
                        words.add(w)
                    new_frontier.append(w)
            frontier = new_frontier
        for _ in range(200):
            L = random.randint(1, 8)
            w = "".join(random.choice(ALPHABET) for _ in range(L))
            if ExploreOracle._doors_len(w) <= self.oracle._budget():
                words.add(w)
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


# Runner/CLI moved to runner.py; this module contains only solver logic.
