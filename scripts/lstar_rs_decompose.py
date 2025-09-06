import lstar
import lstar_random_walks


class LStarMooreRSDecomposeLearner(lstar_random_walks.LStarMooreRandomWalkLearner):
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
        super().__init__(oracle, max_loops, num_random_walks, max_random_walk_length)

    # -----------------------------------------
    # (b) 数本の RW で反例探索 → (c) RS 分解で E 拡張
    # 依存: self.oracle.ensure/commit/g, ALPHABET, self.S, self.E, Hypothesis
    # -----------------------------------------

    def _predict_label(self, hyp: lstar.Hypothesis, word: str) -> int:
        """仮説 Moore オートマトン hyp 上で語 word の最終出力（終状態ラベル）を返す。"""
        s = 0
        for ch in word:
            s = hyp.delta[s][int(ch)]
        return hyp.outputs[s]


    def _gen_random_walks(self, num: int = 6, max_len: int = 14, seed: int | None = None) -> list[str]:
        """
        (b) 反例候補のランダム語を複数生成（1 本の最長語ではなく“数本”）。
        文献上、反例は「短い分岐＋短い接尾辞」に集約されがちで、長大 1 本より
        複数本の中～長 RW を投げる方が効率的なことが多い。:contentReference[oaicite:1]{index=1}
        """
        import random as _rnd
        rnd = _rnd.Random(seed)
        walks = []
        for _ in range(max(1, num)):
            L = rnd.randint(max(3, max_len // 2), max_len)
            walks.append("".join(rnd.choice(lstar.ALPHABET) for _ in range(L)))
        return walks


    def _rs_decompose_and_add_suffix(self, hyp: lstar.Hypothesis, w: str) -> bool:
        """
        (c) Rivest–Schapire 風の反例分解:
        - w を二分探索で w = u a v に分解（最初に不一致が生じる境界）
        - 候補 {v, a+v} のうち短い方を識別接尾辞として self.E に 1 語だけ追加
        - さらに w の全プレフィックスを self.S に昇格（アクセス拡張）
        参考: L* と RS の反例解析。:contentReference[oaicite:2]{index=2}
        """
        # 本当に反例か確認
        real_last = self.oracle.g(w)
        pred_last = self._predict_label(hyp, w)
        if real_last == pred_last:
            return False

        # どの接頭辞から不一致になるか：二分探索
        lo, hi = 0, len(w)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._predict_label(hyp, w[:mid]) == self.oracle.g(w[:mid]):
                lo = mid + 1
            else:
                hi = mid
        split = lo  # w[:split-1] まで一致、w[:split] で初不一致

        # w = u a v
        u = w[:max(0, split - 1)]
        a = w[split - 1:split]
        v = w[split:]

        # 短い識別接尾辞を優先
        for e in sorted({v, (a + v) if a else v}, key=len):
            if e not in self.E:
                self.E.append(e)
                break

        # 反例の全プレフィックスを S に追加（L* 定石）:contentReference[oaicite:3]{index=3}
        for k in range(1, len(w) + 1):
            pfx = w[:k]
            if pfx not in self.S:
                self.S.append(pfx)

        return True


    def strengthen_with_counterexample(self, hyp: lstar.Hypothesis,
                                    rw_num: int = 6,
                                    rw_max_len: int = 14,
                                    seed: int | None = None) -> bool:
        """
        置き換え版:
        1) 数本の RW 語を生成（(b)）
        2) まとめて Oracle に投入（/explore はバッチ化で回数節約）
        3) 予測と実測が食い違う語 w を見つけたら、RS 分解で self.E を 1 語だけ拡張し、
            w の全プレフィックスを self.S に追加して True を返す（(c)）
        4) 見つからなければ False
        背景: L*（Angluin 1987）と RS 反例解析、近年の反例処理の抽象枠組み。:contentReference[oaicite:4]{index=4}
        """
        # (1) RW 候補を作る
        rw_num = self.num_random_walks
        rw_max_len = self.max_random_walk_length
        tests = set(self._gen_random_walks(num=rw_num, max_len=rw_max_len, seed=seed))

        # (2) 一括問い合わせ（/explore の回数は増やさない）
        self.oracle.ensure(tests)
        self.oracle.commit()

        # (3) 反例を探し、見つけたら RS で E を拡張
        for w in tests:
            if self._predict_label(hyp, w) != self.oracle.g(w):
                self._rs_decompose_and_add_suffix(hyp, w)
                return True

        # (4) 反例なし
        return False
