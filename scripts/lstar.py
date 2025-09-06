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
import datetime

BASE_URL = "https://31pwr5t6ij.execute-api.eu-west-2.amazonaws.com"
# BASE_URL = "http://127.0.0.1:8009/"
ALPHABET = "012345"  # door labels

# ---------- API Client ----------


class AedificiumClient:
    """Ædificium コンテストの各種 API を扱うクライアントクラス."""

    def __init__(self, base_url: str = BASE_URL, team_id: Optional[str] = None):
        """
        Args:
            base_url (str): API のベース URL。
            team_id (Optional[str]): チームの識別子（登録後に取得）。
        """
        self.base_url = base_url.rstrip("/")
        self.id = team_id

    def _post(self, path: str, payload: Dict) -> Dict:
        """
        内部用：HTTP POST リクエストを送信し JSON レスポンスを返す。

        Args:
            path (str): エンドポイント（/register, /select, /explore, /guess）。
            payload (Dict): 送信する JSON データ。

        Returns:
            Dict: サーバーからの JSON 応答。

        Raises:
            HTTPError: リクエストが失敗した場合。
        """
        url = f"{self.base_url}{path}"
        r = requests.post(url, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()

    def register(self, name: str, pl: str, email: str) -> str:
        """
        チーム登録を行い、チーム ID（秘密）を取得する。

        Args:
            name (str): チーム名。
            pl (str): 使用言語（例："Python"）。
            email (str): 連絡用メールアドレス。

        Returns:
            str: 発行された team_id。
        """
        data = {"name": name, "pl": pl, "email": email}
        j = self._post("/register", data)
        self.id = j["id"]
        return self.id

    def select_problem(self, problem_name: str) -> str:
        """
        解きたい問題を選択する。

        Args:
            problem_name (str): 問題名（例："probatio"）。

        Returns:
            str: 選択された問題名（確認用）。
        """
        assert self.id, "team id is required. call register() or set client.id"
        j = self._post("/select", {"id": self.id, "problemName": problem_name})
        return j["problemName"]

    def explore(self, plans: List[str]) -> Tuple[List[List[int]], int]:
        """
        扉プラン文字列リストを送信し、各探索結果（ラベル列）と累計 queryCount を取得。

        Args:
            plans (List[str]): 0–5 の文字列（例："012"）を複数。

        Returns:
            Tuple[List[List[int]], int]:
                - 各プランの入室ごとのラベル列のリスト（計画長 + 1 個）。
                - 累計クエリ数（queryCount）。
        """
        assert self.id, "team id is required"
        j = self._post("/explore", {"id": self.id, "plans": plans})
        return j["results"], j["queryCount"]

    def guess(self, guess_map: Dict) -> bool:
        """
        推測地図を送信して正否を確認する。

        Args:
            guess_map (Dict): "rooms", "startingRoom", "connections" を含む地図情報。

        Returns:
            bool: 正解なら True、そうでなければ False。
        """
        assert self.id, "team id is required"
        j = self._post("/guess", {"id": self.id, "map": guess_map})
        return bool(j.get("correct", False))


# ---------- Membership Oracle with batching & caching ----------


class ExploreOracle:
    """
    /explore をバッチ化し、語に対する観測結果（ラベル）をキャッシュするユーティリティ。

    For a plan w of length x, server returns a list of length x+1; we take the last
    element as g(w) (Moore state's label after executing w). The initial state's
    label is the 0-th element of any returned trace.
    """

    def __init__(self, client: AedificiumClient):
        """
        Args:
            client (AedificiumClient): API クライアントインスタンス。
        """
        self.client = client
        self.trace_cache: Dict[str, List[int]] = {}  # word -> full trace (len = |w|+1)
        self.last_label: Dict[str, int] = {}  # word -> final label
        self.pending: List[str] = []
        self.init_label: Optional[int] = None
        self.last_query_count: int = 0
        self.last_query_count_output_datetime = datetime.datetime.now()

    def ensure(self, words: Set[str]) -> None:
        """
        要求されたすべての語を /explore のバッチ対象として追加（重複除去済み）。

        Args:
            words (Set[str]): クエリ対象の語集合。
        """
        for w in words:
            if w == "":
                # don't push empty; derive init label from any non-empty query
                continue
            if w not in self.trace_cache and w not in self.pending:
                self.pending.append(w)

    def commit(self) -> None:
        """
        バッチに溜まった語を /explore に投げて結果をキャッシュに登録。
        """
        if not self.pending:
            return
        plans = list(dict.fromkeys(self.pending))  # dedupe, keep order
        self.pending.clear()
        results, _qc = self.client.explore(plans)
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
            # trace length must be len(w)+1
            self.trace_cache[w] = trace
            self.last_label[w] = trace[-1]
            if self.init_label is None and len(trace) >= 1:
                self.init_label = trace[0]

    def g(self, w: str) -> int:
        """
        語 w 実行後の状態に対応する 2bit ラベルを返す。

        Args:
            w (str): 扉プラン文字列（"" は初期ラベル）。

        Returns:
            int: 対応するラベル（0–3）。
        """
        if w == "":
            # initial label
            if self.init_label is not None:
                return self.init_label
            # ❌ 余計な /explore は発行しない。
            # 最初の観測表バッチ（非空語）と同じ応答から trace[0] を拾う。
            for trace in self.trace_cache.values():
                if trace:
                    self.init_label = trace[0]
                    break
            if self.init_label is None:
                raise RuntimeError(
                    "Initial label is not available yet. "
                    "Batch some non-empty words with ensure(...); commit() first."
                )
            return self.init_label
        if w not in self.last_label:
            self.ensure({w})
            self.commit()
        return self.last_label[w]

    def ensure_rows(self, S: List[str], E: List[str]) -> None:
        """
        L* アルゴリズム用の観測表行列を作成するために必要な全語をバッチで問い合わせ。

        Args:
            S (List[str]): アクセス語リスト。
            E (List[str]): 識別接尾辞リスト。
        """
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

    def predict_trace_end(
        self, delta, rep_for_state, start_rep: str, word: str, outputs
    ) -> int:
        """
        仮説機械に基づいて、語 word の実行後に到達する状態のラベルを予測する。

        Args:
            delta (List[List[int]]): 遷移関数 [state][door]→state。
            rep_for_state (List[str]): 各状態を代表する語のリスト。
            start_rep (str): 開始状態を表す語。
            word (str): テストする語。
            outputs (List[int]): 各状態の Moore 出力ラベル。

        Returns:
            int: 予測された最終ラベル値。
        """
        # Simulate from start state 0
        s = 0
        for ch in word:
            s = delta[s][int(ch)]
        return outputs[s]


# ---------- L* for Moore machines ----------


@dataclass
class Hypothesis:
    """仮説オートマトンを表現するデータ構造."""

    rows: Dict[str, Tuple[int, ...]]  # row signature for reps in S
    state_index_of_row: Dict[Tuple[int, ...], int]
    rep_for_state: List[str]  # representative word for each state
    outputs: List[int]  # Moore output per state
    delta: List[List[int]]  # transitions [state][door] -> state


class LStarMooreLearner:
    """
    Moore 型オートマトン用 L* 学習器。閉性／一貫性／反例検出で仮説を修正。
    """

    def __init__(
        self,
        oracle: ExploreOracle,
        max_loops: int = 200,
        bfs_depth: int = 4,
        bfs_adoption_propbability: float = 1.0,
        max_random_len: int = 8,
        num_trials: int = 200,
    ):
        """
        Args:
            oracle (ExploreOracle): 観測を収集するオラクル。
            max_loops (int): 学習ループの上限回数。
        """
        self.oracle = oracle
        self.max_loops = max_loops
        self.bfs_depth = bfs_depth
        self.bfs_adoption_propbability = bfs_adoption_propbability
        self.max_random_len = max_random_len
        self.num_trials = num_trials
        self.S: List[str] = [""]  # access prefixes
        self.E: List[str] = [""]  # distinguishing suffixes (must contain "")
        # Pre-warm a tiny query to get initial label (and server wake-up)
        self.oracle.ensure({"0"})
        self.oracle.commit()

    def row(self, u: str) -> Tuple[int, ...]:
        """
        語 u に基づいた観測表の行を返す（各識別接尾辞に対する出力ラベル）。

        Args:
            u (str): アクセス語。

        Returns:
            Tuple[int, ...]: E に対応する出力ラベル列。
        """
        return tuple(self.oracle.g(u + e) for e in self.E)

    def build_table(self):
        """現在の S, E に基づいて観測表データを収集する。"""
        self.oracle.ensure_rows(self.S, self.E)

    def is_closed(self, rowS: Dict[str, Tuple[int, ...]]) -> Optional[str]:
        """
        閉性を満たしていない場合、必要な語を返す。

        Args:
            rowS (Dict[str, Tuple[int, ...]]): 各アクセス語の行シグネチャ。

        Returns:
            Optional[str]: 欠けているアクセス語、なければ None。
        """
        rows_set = {rowS[s] for s in self.S}
        for u in list(self.S):
            for a in ALPHABET:
                ua = u + a
                r = self.row(ua)
                if r not in rows_set:
                    return ua
        return None

    def find_inconsistency(
        self, rowS: Dict[str, Tuple[int, ...]]
    ) -> Optional[Tuple[str, str, str]]:
        """
        一貫性違反を検出する。

        Args:
            rowS (Dict[str, Tuple[int, ...]]): 各アクセス語の行シグネチャ。

        Returns:
            Optional[Tuple[str, str, str]]:
                (s1, s2, a) の形で、一貫性違反を示す例。なければ None。
        """
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
        """
        観測表に基づいて仮説オートマトンを構造化し返す。

        Returns:
            Hypothesis: 構築された仮説オートマトン。
        """
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

        return Hypothesis(
            rows=rows_map,
            state_index_of_row=idx_of_row,
            rep_for_state=rep_for_state,
            outputs=outputs,
            delta=delta,
        )

    def strengthen_with_counterexample(self, hyp: Hypothesis) -> bool:
        """
        仮説に対する反例があれば S を拡張する。

        Args:
            hyp (Hypothesis): 現在の仮説。
            max_len (int): ランダム語の最大長。
            trials (int): テスト語生成数（ランダム）。

        Returns:
            bool: 反例が見つかり S を拡張した場合 True。
        """
        # Build some test words (systematic + random)
        tests: Set[str] = set()
        # small BFS up to length self.bfs_depth
        frontier = [""]
        for _ in range(self.bfs_depth):
            new_frontier = []
            for u in frontier:
                for a in ALPHABET:
                    if random.random() > self.bfs_adoption_propbability:
                        continue
                    w = u + a
                    tests.add(w)
                    new_frontier.append(w)
            frontier = new_frontier
        # random longer
        for _ in range(self.num_trials):
            L = random.randint(1, self.max_random_len)
            w = "".join(random.choice(ALPHABET) for _ in range(L))
            tests.add(w)

        # ensure queries
        self.oracle.ensure(set(tests))
        self.oracle.commit()

        for w in tests:
            pred = self.oracle.predict_trace_end(
                hyp.delta, hyp.rep_for_state, hyp.rep_for_state[0], w, hyp.outputs
            )
            real = self.oracle.g(w)
            if pred != real:
                # refine: add all prefixes of w to S
                for k in range(1, len(w) + 1):
                    p = w[:k]
                    if p not in self.S:
                        self.S.append(p)
                    # print(f"Found counterexample.  Added '{p}' to S.  Continue.")
                return True
        return False

    def learn(self) -> Hypothesis:
        """
        L* フローに従って学習を進め、最終的に仮説オートマトンを返す。

        Returns:
            Hypothesis: 学習完了時の仮説機械。
        """
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
                # print(f"Not closed.  Added '{need}' to S.  Continue.")
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
                # print(f"Not consistent.  Added '{new_e}' to E.  Continue.")
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
    """
    仮説オートマトンから /guess 提出用のマップ構造を生成する。

    Args:
        hyp (Hypothesis): 学習された仮説機械。

    Returns:
        Dict: /guess API に渡す地図情報。
    """
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

            connections.append(
                {"from": {"room": r, "door": a}, "to": {"room": r2, "door": partner}}
            )
            used[r][a] = True
            used[r2][partner] = True

    return {"rooms": rooms, "startingRoom": starting_room, "connections": connections}


# ---------- Example runner ----------


def main():
    """
    登録・問題選択・学習・推測・提出までの一連の流れを実行するメイン関数。
    """
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
    # PROBLEM_NAME = "probatio"
    # PROBLEM_NAME = "primus"
    # PROBLEM_NAME = "secundus"
    # PROBLEM_NAME = "tertius"
    # PROBLEM_NAME = "quartus"
    PROBLEM_NAME = "quintus"
    chosen = client.select_problem(PROBLEM_NAME)
    print("Selected:", chosen)

    oracle = ExploreOracle(client)
    learner = LStarMooreLearner(oracle)

    hyp = learner.learn()
    guess_map = build_guess_from_hypothesis(hyp)

    # Optionally pretty print the guess before submitting
    print(json.dumps(guess_map, indent=2))

    ok = client.guess(guess_map)
    print("Guess correct?", ok)


if __name__ == "__main__":
    # For safety in an example script; remove in production
    try:
        main()
    except requests.HTTPError as e:
        print("HTTP error:", e)
    except Exception as ex:
        print("Error:", ex)
