#include <iostream>
#include <vector>
#include <queue>
#include <array>
#include <unordered_map>
#include <unordered_set>
#include <optional>
#include <random>
#include <chrono>


namespace NJudge {

    //using namespace std;

    using Door = int;
    using Room = int;
    using Port = std::pair<Room, Door>; // (room, door)

    //======================= 実マップ（地図の真値） =======================
    struct Labyrinth {
        std::vector<int> labels;                 // 各部屋の2bitラベル（0..3）
        Room start = 0;                     // 開始部屋
        std::vector<std::array<Port, 6>> to;          // to[r][d] = (r', d') 扉dで移動後の(部屋, 扉)

        Room step(Room r, Door d) const {
            auto [nr, nd] = to[r][d];
            return nr;
        }

        // 仕様通り：各ステップで「入室後」の部屋ラベルを記録
        std::vector<int> explore_plan(const std::string& plan) const {
            Room cur = start;
            std::vector<int> out; out.reserve(plan.size());
            for (char ch : plan) {
                int d = ch - '0';
                if (d < 0 || d > 5) throw std::runtime_error("plan contains non-door digit");
                cur = step(cur, d);
                out.push_back(labels[cur]);
            }
            return out;
        }
    };

    //======================= 提出形式（/guess の map をモデル化） =======================
    struct GuessConnectionEnd { Room room; Door door; };
    struct GuessConnection { GuessConnectionEnd from; GuessConnectionEnd to; };

    struct GuessMap {
        std::vector<int> rooms;                   // 2bit ラベル列
        Room startingRoom = 0;
        std::vector<GuessConnection> connections; // 片側書けば自動で両方向を張る

        Labyrinth to_labyrinth() const {
            int n = (int)rooms.size();
            if (n <= 0) throw std::runtime_error("rooms empty");
            std::vector<std::array<Port, 6>> to(n);
            // 未定義を (-1,-1) で初期化
            for (int r = 0; r < n; ++r) for (int d = 0; d < 6; ++d) to[r][d] = Port{ -1,-1 };

            auto in_range_room = [&](int r) { return 0 <= r && r < n; };
            auto in_range_door = [&](int d) { return 0 <= d && d < 6; };

            for (const auto& c : connections) {
                int r1 = c.from.room, d1 = c.from.door;
                int r2 = c.to.room, d2 = c.to.door;
                if (!in_range_room(r1) || !in_range_room(r2) || !in_range_door(d1) || !in_range_door(d2))
                    throw std::runtime_error("connection out of range");
                if (to[r1][d1] != Port{ -1,-1 } || to[r2][d2] != Port{ -1,-1 })
                    throw std::runtime_error("duplicate assignment on a door");
                // 無向：両側を張る。自己ループ (r,d)->(r,d) も許容
                to[r1][d1] = Port{ r2, d2 };
                to[r2][d2] = Port{ r1, d1 };
            }

            // 全扉が定義済みか確認
            for (int r = 0; r < n; ++r)
                for (int d = 0; d < 6; ++d)
                    if (to[r][d] == Port{ -1,-1 })
                        throw std::runtime_error("door undefined: room=" + std::to_string(r) + " door=" + std::to_string(d));

            Labyrinth L;
            L.labels = rooms;
            L.start = startingRoom;
            L.to = std::move(to);
            return L;
        }
    };

    //======================= 等価性判定（挙動同値：積グラフBFS） =======================
    bool equivalent(const Labyrinth& a, const Labyrinth& b) {
        if (a.labels.size() != b.labels.size()) return false; // 部屋数一致が必要条件

        std::queue<std::pair<Room, Room>> q;
        std::unordered_set<long long> seen; // (u,v) の組をエンコード
        auto enc = [](int u, int v)->long long { return ((long long)u << 32) ^ (unsigned long long)v; };

        q.emplace(a.start, b.start);
        seen.insert(enc(a.start, b.start));

        while (!q.empty()) {
            auto [u, v] = q.front(); q.pop();
            for (int d = 0; d < 6; ++d) {
                auto [u2, _1] = a.to[u][d];
                auto [v2, _2] = b.to[v][d];
                // 入室後の部屋ラベルの一致を確認（Moore出力）
                if (a.labels[u2] != b.labels[v2]) return false;
                long long key = enc(u2, v2);
                if (!seen.count(key)) {
                    seen.insert(key);
                    q.emplace(u2, v2);
                }
            }
        }
        return true;
    }

    //======================= 乱択生成器 =======================
    Labyrinth generate_random_labyrinth(int n_rooms, std::optional<uint64_t> seed) {
        std::mt19937_64 rng(seed.has_value() ? *seed : (uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count());
        Labyrinth L;
        L.labels.resize(n_rooms);
        for (int i = 0; i < n_rooms; ++i) L.labels[i] = (int)(rng() % 4);
        L.start = 0;

        // すべてのポート (r,d) を列挙
        std::vector<Port> ports; ports.reserve((size_t)n_rooms * 6);
        for (int r = 0; r < n_rooms; ++r) for (int d = 0; d < 6; ++d) ports.emplace_back(r, d);
        std::shuffle(ports.begin(), ports.end(), rng);

        // 双方向マッチング。少数の自己固定点 (p->p) を許容
        const double p_self = 0.04;
        std::vector<std::array<Port, 6>> to(n_rooms);
        for (int r = 0; r < n_rooms; ++r) for (int d = 0; d < 6; ++d) to[r][d] = Port{ -1,-1 };

        // 割当済みか判定するための map
        struct KeyHash { size_t operator()(const Port& p) const { return ((size_t)p.first << 3) ^ (size_t)p.second; } };
        std::unordered_map<Port, Port, KeyHash> pair_of;

        size_t i = 0;
        while (i < ports.size()) {
            Port p = ports[i++];
            if (pair_of.find(p) != pair_of.end()) continue;

            bool do_self = (double)(rng() % 1000000) / 1000000.0 < p_self;
            if (do_self) {
                pair_of[p] = p;
                continue;
            }
            // 相手 q を未割当から探す
            size_t j = i;
            while (j < ports.size() && pair_of.find(ports[j]) != pair_of.end()) ++j;
            if (j >= ports.size()) {
                pair_of[p] = p; // 残渣は自己固定点
            }
            else {
                Port q = ports[j];
                pair_of[p] = q;
                pair_of[q] = p;
                // q を i-1 と入替（スキップ）
                std::swap(ports[j], ports[i - 1]);
            }
        }

        for (auto& kv : pair_of) {
            auto [r, d] = kv.first;
            auto [r2, d2] = kv.second;
            to[r][d] = Port{ r2, d2 };
        }

        // 完全性保証
        for (int r = 0; r < n_rooms; ++r)
            for (int d = 0; d < 6; ++d)
                if (to[r][d] == Port{ -1,-1 })
                    throw std::runtime_error("internal generator error: incomplete port");

        L.to = std::move(to);
        return L;
    }

    //======================= ジャッジ本体 =======================
    struct ExploreResult {
        std::vector<std::vector<int>> results;
        int queryCount = 0;
    };

    struct LocalJudge {

        std::unordered_map<std::string, int> catalog = {
            {"probatio", 3},
            {"primus",   6},
            {"secundus",12},
            {"tertius", 18},
            {"quartus", 24},
            {"quintus", 30}
        };

        std::optional<Labyrinth> active;
        int queryCount = 0;
        std::optional<uint64_t> rng_seed;
        std::optional<std::string> problem_name;

        std::string select_problem(const std::string& name, std::optional<uint64_t> seed = std::nullopt) {
            auto it = catalog.find(name);
            if (it == catalog.end()) throw std::runtime_error("unknown problemName");
            int n = it->second;
            rng_seed = seed;
            active = generate_random_labyrinth(n, seed);
            problem_name = name;
            queryCount = 0;
            return name;
        }

        ExploreResult explore(const std::vector<std::string>& plans) {
            if (!active.has_value()) throw std::runtime_error("select a problem first");
            ExploreResult er;
            for (auto& p : plans) er.results.push_back(active->explore_plan(p));
            // 「探索回数 = 送ったルート数 + リクエスト1のペナルティ」
            queryCount += (int)plans.size() + 1;
            er.queryCount = queryCount;
            return er;
        }

        bool guess(const GuessMap& gm) {
            if (!active.has_value()) throw std::runtime_error("select a problem first");
            bool ok = false;
            try {
                Labyrinth cand = gm.to_labyrinth();
                ok = equivalent(*active, cand);
            }
            catch (...) {
                // 例外は不正解扱い
                ok = false;
            }
            // /guess は常に問題をクリア（成功/失敗に関わらずデセレクト）
            active.reset();
            rng_seed.reset();
            problem_name.reset();
            queryCount = 0;
            return ok;
        }
    };

    //======================= 簡易デモ（雰囲気用） =======================
    int demo() {
        std::ios::sync_with_stdio(false);
        std::cin.tie(nullptr);

        std::cout << "== demo ==\n";
        LocalJudge J;
        J.select_problem("probatio", /*seed*/ 42);

        auto er = J.explore({ "0", "123", "505" });
        std::cout << "queryCount=" << er.queryCount << "\n";
        for (auto& rec : er.results) {
            for (size_t i = 0; i < rec.size(); ++i) {
                if (i) std::cout << ' ';
                std::cout << rec[i];
            }
            std::cout << "\n";
        }

        // いい加減な誤答（全扉未定義のため to_labyrinth() で例外→false になる想定）
        GuessMap bad;
        bad.rooms = { 0,1,2 };
        bad.startingRoom = 0;
        bad.connections = {
            {{0,0},{1,1}},
            {{0,1},{2,2}},
            {{0,2},{0,2}}, // 自己ループ例
            // …本来は全ポート(部屋×6)を埋める必要あり
        };
        bool ok = J.guess(bad);
        std::cout << "guess(correct?)=" << (ok ? "true" : "false") << "\n";

        // 以降、本気で使う場合はライブラリとして import 想定：
        // - select_problem(name, seed)
        // - explore(std::vector<std::string>)
        // - guess(GuessMap)
        return 0;
    }

}


int main() {

    NJudge::demo();
    
	return 0;
}