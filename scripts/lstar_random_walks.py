import lstar


class LStarMooreRandomWalkLearner(lstar.LStarMooreLearner):
    """
    Moore 型オートマトン用 L* 学習器。閉性／一貫性／反例検出で仮説を修正。
    """

    def __init__(
        self,
        oracle: lstar.ExploreOracle,
        max_loops: int = 200,
        num_random_walks: int = 4,
        max_random_walk_length: int = 16,
    ):
        super().__init__(oracle, max_loops)
        self.num_random_walks = num_random_walks
        self.max_random_walk_length = max_random_walk_length

    # --- ランダムウォーク生成ユーティリティ -----------------------------

    def _gen_random_walks(
        self, num: int = 4, max_len: int = 16, seed: int | None = None
    ) -> list[str]:
        """
        数本のランダムウォーク語を生成する。
        - 1 本の“最長語”に偏らず、複数本（長さは中～やや長）で分岐を広く叩く。
        - 生成した語は strengthen_with_counterexample() で一括問い合わせる。
        参考: L* は観測表で仮説を作り、反例で修正する枠組み。反例候補として
            ランダム語を用いるのは実務で一般的（理論背景は Angluin 系）。
        """
        import random as _rnd

        rnd = _rnd.Random(seed)
        walks = []
        for _ in range(max(1, num)):
            L = rnd.randint(max(3, max_len // 2), max_len)
            walks.append("".join(rnd.choice(lstar.ALPHABET) for _ in range(L)))
        return walks

    # --- (b) RW のみを用いる strengthen_with_counterexample ---------------

    def strengthen_with_counterexample(
        self,
        hyp: lstar.Hypothesis,
        rw_num: int = 4,
        rw_max_len: int = 16,
        seed: int | None = None,
    ) -> bool:
        """
        置き換え版：ランダムウォーク語のみで反例探索を行う。
        1) RW 語を複数生成
        2) Oracle に一括投入して実測ラベルを取得（/explore はバッチで 1 回に抑制）
        3) 予測と不一致の語を見つけたら、それを反例として
            その全プレフィックスを S に追加（E は変更しない）

        戻り値:
            bool: 反例が見つかって S を拡張したら True、なければ False
        """
        rw_num = self.num_random_walks
        rw_max_len = self.max_random_walk_length

        # 1) 候補語の生成
        tests = set(self._gen_random_walks(num=rw_num, max_len=rw_max_len, seed=seed))

        # 2) 一括問い合わせ（Oracle 側で /explore をバッチ化）
        self.oracle.ensure(tests)
        self.oracle.commit()

        # 3) 反例探索：仮説の予測と実測の最終ラベルを比較
        def predict(word: str) -> int:
            s = 0
            for ch in word:
                s = hyp.delta[s][int(ch)]
            return hyp.outputs[s]

        for w in tests:
            pred = predict(w)
            real = self.oracle.g(w)
            if pred != real:
                # 反例 w の全プレフィックスを S に昇格（L* の定石）
                for k in range(1, len(w) + 1):
                    pfx = w[:k]
                    if pfx not in self.S:
                        self.S.append(pfx)
                return True

        # 反例なし
        return False
