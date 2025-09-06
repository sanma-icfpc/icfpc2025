#define _CRT_NONSTDC_NO_WARNINGS
#define _SILENCE_CXX17_ITERATOR_BASE_CLASS_DEPRECATION_WARNING
#include <bits/stdc++.h>
#include <random>
#include <unordered_set>
#include <array>
#include <optional>
//#include <format>
#ifdef _MSC_VER
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <conio.h>
#include <ppl.h>
#include <omp.h>
#include <filesystem>
#include <intrin.h>

/* g++ functions */
int __builtin_clz(unsigned int n) { unsigned long index; _BitScanReverse(&index, n); return 31 - index; }
int __builtin_ctz(unsigned int n) { unsigned long index; _BitScanForward(&index, n); return index; }
namespace std { inline int __lg(int __n) { return sizeof(int) * 8 - 1 - __builtin_clz(__n); } }
int __builtin_popcount(int bits) {
    bits = (bits & 0x55555555) + (bits >> 1 & 0x55555555);
    bits = (bits & 0x33333333) + (bits >> 2 & 0x33333333);
    bits = (bits & 0x0f0f0f0f) + (bits >> 4 & 0x0f0f0f0f);
    bits = (bits & 0x00ff00ff) + (bits >> 8 & 0x00ff00ff);
    return (bits & 0x0000ffff) + (bits >> 16 & 0x0000ffff);
}
/* enable __uint128_t in MSVC */
//#include <boost/multiprecision/cpp_int.hpp>
//using __uint128_t = boost::multiprecision::uint128_t;
#include <windows.h>
#include <psapi.h>
template<typename Pred>
static bool WaitFor(Pred pred, DWORD timeout_ms = INFINITE) {
    std::cerr << __FUNCTION__ << ": waiting...\n";
    DWORD wait = 500;
    bool done = false;
    while (!(done = pred()) && timeout_ms > 0) {
        if (timeout_ms != INFINITE) {
            if (timeout_ms < wait) wait = timeout_ms;
            else timeout_ms -= wait;
        }
        Sleep(wait);
        std::cerr << __FUNCTION__ << ": ...\n";
    }
    if (done) {
        std::cerr << __FUNCTION__ << ": leaving (succeeded)\n";
        Sleep(100);
        return true;
    }
    std::cerr << __FUNCTION__ << ": leaving (timeout)\n";
    return false;
}
bool WaitDebugger(DWORD timeout_ms) {
    return WaitFor([] {return IsDebuggerPresent() != FALSE; }, timeout_ms);
}
#else
#pragma GCC target("avx2")
#pragma GCC optimize("O3")
#pragma GCC optimize("unroll-loops")
#endif

/** compro io **/
namespace aux {
    template<typename T, unsigned N, unsigned L> struct tp { static void output(std::ostream& os, const T& v) { os << std::get<N>(v) << ", "; tp<T, N + 1, L>::output(os, v); } };
    template<typename T, unsigned N> struct tp<T, N, N> { static void output(std::ostream& os, const T& v) { os << std::get<N>(v); } };
}
template<typename... Ts> std::ostream& operator<<(std::ostream& os, const std::tuple<Ts...>& t) { os << '{'; aux::tp<std::tuple<Ts...>, 0, sizeof...(Ts) - 1>::output(os, t); return os << '}'; } // tuple out
template<class Ch, class Tr, class Container> std::basic_ostream<Ch, Tr>& operator<<(std::basic_ostream<Ch, Tr>& os, const Container& x); // container out (fwd decl)
template<class S, class T> std::ostream& operator<<(std::ostream& os, const std::pair<S, T>& p) { return os << '{' << p.first << ", " << p.second << '}'; } // pair out
template<class S, class T> std::istream& operator>>(std::istream& is, std::pair<S, T>& p) { return is >> p.first >> p.second; } // pair in
std::ostream& operator<<(std::ostream& os, const std::vector<bool>::reference& v) { os << (v ? '1' : '0'); return os; } // bool (vector) out
std::ostream& operator<<(std::ostream& os, const std::vector<bool>& v) { bool f = true; os << '{'; for (const auto& x : v) { os << (f ? "" : ", ") << x; f = false; } os << '}'; return os; } // vector<bool> out
template<class Ch, class Tr, class Container> std::basic_ostream<Ch, Tr>& operator<<(std::basic_ostream<Ch, Tr>& os, const Container& x) { bool f = true; os << '{'; for (auto& y : x) { os << (f ? "" : ", ") << y; f = false; } return os << '}'; } // container out
template<class T, class = decltype(std::begin(std::declval<T&>())), class = typename std::enable_if<!std::is_same<T, std::string>::value>::type> std::istream& operator>>(std::istream& is, T& a) { for (auto& x : a) is >> x; return is; } // container in
template<typename T> auto operator<<(std::ostream& out, const T& t) -> decltype(out << t.str()) { out << t.str(); return out; } // struct (has stringify() func) out
/** io setup **/
struct IOSetup { IOSetup(bool f) { if (f) { std::cin.tie(nullptr); std::ios::sync_with_stdio(false); } std::cout << std::fixed << std::setprecision(15); } }
iosetup(true); // set false when solving interective problems
/** string formatter **/
template<typename... Ts> std::string format(const std::string& f, Ts... t) { size_t l = std::snprintf(nullptr, 0, f.c_str(), t...); std::vector<char> b(l + 1); std::snprintf(&b[0], l + 1, f.c_str(), t...); return std::string(&b[0], &b[0] + l); }
/** dump **/
#define ENABLE_DUMP
#ifdef ENABLE_DUMP
std::ofstream ofs("log.txt");
#define DUMPOUT ofs
#define debug(...) do{DUMPOUT<<"  ";DUMPOUT<<#__VA_ARGS__<<" :[DUMP - "<<__LINE__<<":"<<__FUNCTION__<<"]"<<std::endl;DUMPOUT<<"    ";dump_func(__VA_ARGS__);}while(0);
void dump_func() { DUMPOUT << std::endl; }
template <class Head, class... Tail> void dump_func(Head&& head, Tail&&... tail) { DUMPOUT << head; if (sizeof...(Tail) == 0) { DUMPOUT << " "; } else { DUMPOUT << ", "; } dump_func(std::move(tail)...); }
#else
#define debug(...) void(0);
#endif
/** timer **/
class Timer {
    double t = 0, paused = 0, tmp;
public:
    Timer() { reset(); }
    static double time() {
#ifdef _MSC_VER
        return __rdtsc() / 2.9e9;
#else
        unsigned long long a, d;
        __asm__ volatile("rdtsc"
            : "=a"(a), "=d"(d));
        return (d << 32 | a) / 2.9e9;
#endif
    }
    void reset() { t = time(); }
    void pause() { tmp = time(); }
    void restart() { paused += time() - tmp; }
    double elapsed_ms() const { return (time() - t - paused) * 1000.0; }
} g_timer;
/** rand **/
struct Xorshift {
    Xorshift() {}
    Xorshift(uint64_t seed) { reseed(seed); }
    inline void reseed(uint64_t seed) { x = 0x498b3bc5 ^ seed; for (int i = 0; i < 20; i++) next_u64(); }
    inline uint64_t next_u64() { x ^= x << 7; return x ^= x >> 9; }
    inline uint32_t next_u32() { return next_u64() >> 32; }
    inline uint32_t next_u32(uint32_t mod) { return ((uint64_t)next_u32() * mod) >> 32; }
    inline uint32_t next_u32(uint32_t l, uint32_t r) { return l + next_u32(r - l + 1); }
    inline double next_double() { return next_u32() * e; }
    inline double next_double(double c) { return next_double() * c; }
    inline double next_double(double l, double r) { return next_double(r - l) + l; }
private:
    static constexpr uint32_t M = UINT_MAX;
    static constexpr double e = 1.0 / M;
    uint64_t x = 88172645463325252LL;
};
/** shuffle **/
template<typename T> void shuffle_vector(std::vector<T>& v, Xorshift& rnd) { int n = v.size(); for (int i = n - 1; i >= 1; i--) { int r = rnd.next_u32(i); std::swap(v[i], v[r]); } }
/** split **/
std::vector<std::string> split(const std::string& str, const std::string& delim) {
    std::vector<std::string> res;
    std::string buf;
    for (const auto& c : str) {
        if (delim.find(c) != std::string::npos) {
            if (!buf.empty()) res.push_back(buf);
            buf.clear();
        }
        else buf += c;
    }
    if (!buf.empty()) res.push_back(buf);
    return res;
}
std::string join(const std::string& delim, const std::vector<std::string>& elems) {
    if (elems.empty()) return "";
    std::string res = elems[0];
    for (int i = 1; i < (int)elems.size(); i++) {
        res += delim + elems[i];
    }
    return res;
}
/** misc **/
template<typename A, size_t N, typename T> inline void Fill(A(&array)[N], const T& val) { std::fill((T*)array, (T*)(array + N), val); } // fill array
template<typename T, typename ...Args> auto make_vector(T x, int arg, Args ...args) { if constexpr (sizeof...(args) == 0)return std::vector<T>(arg, x); else return std::vector(arg, make_vector<T>(x, args...)); }
template<typename T> bool chmax(T& a, const T& b) { if (a < b) { a = b; return true; } return false; }
template<typename T> bool chmin(T& a, const T& b) { if (a > b) { a = b; return true; } return false; }
/** hash **/
namespace aux { template<typename T> inline void hash(std::size_t& s, const T& v) { s ^= std::hash<T>()(v) + 0x9e3779b9 + (s << 6) + (s >> 2); } }
namespace std { template<typename F, typename S> struct hash<std::pair<F, S>> { size_t operator()(const std::pair<F, S>& s) const noexcept { size_t seed = 0; aux::hash(seed, s.first); aux::hash(seed, s.second); return seed; } }; }

/* fast queue */
class FastQueue {
    int front = 0;
    int back = 0;
    int v[4096];
public:
    inline bool empty() { return front == back; }
    inline void push(int x) { v[front++] = x; }
    inline int pop() { return v[back++]; }
    inline void reset() { front = back = 0; }
    inline int size() { return front - back; }
};

class RandomQueue {
    int sz = 0;
    int v[4096];
public:
    inline bool empty() const { return !sz; }
    inline int size() const { return sz; }
    inline void push(int x) { v[sz++] = x; }
    inline void reset() { sz = 0; }
    inline int pop(int i) {
        std::swap(v[i], v[sz - 1]);
        return v[--sz];
    }
    inline int pop(Xorshift& rnd) {
        return pop(rnd.next_u32(sz));
    }
};

#if 1
inline double get_temp(double stemp, double etemp, double t, double T) {
    return etemp + (stemp - etemp) * (T - t) / T;
};
#else
inline double get_temp(double stemp, double etemp, double t, double T) {
    return stemp * pow(etemp / stemp, t / T);
};
#endif

struct LogTable {
    static constexpr int M = 65536;
    static constexpr int mask = M - 1;
    double l[M];
    LogTable() : l() {
        unsigned long long x = 88172645463325252ULL;
        double log_u64max = log(2) * 64;
        for (int i = 0; i < M; i++) {
            x = x ^ (x << 7);
            x = x ^ (x >> 9);
            l[i] = log(double(x)) - log_u64max;
        }
    }
    inline double operator[](int i) const { return l[i & mask]; }
} log_table;

struct Perf {
    Timer t;
    const char* const func;
    Perf(const char* func_) : func(func_) {}
    ~Perf() {
        //std::format_to(std::ostream_iterator<char>(std::cerr), "[{:>50}] {:7.2f} / {:7.2f} [ms]\n", func, t.elapsed_ms(), g_timer.elapsed_ms());
        fprintf(stderr, "%s: %4.2f / %4.2f [ms]\n", func, t.elapsed_ms(), g_timer.elapsed_ms());
    }
};

#ifdef _DEBUG
#undef HAVE_OPENCV_HIGHGUI
#endif


const std::unordered_map<std::string, int> catalog = {
    {"probatio", 3},
    {"primus",   6},
    {"secundus",12},
    {"tertius", 18},
    {"quartus", 24},
    {"quintus", 30}
};

using Door = int;
using Room = int;

struct Port {

    int room;
    int door;

    Port(int room_ = -1, int door_ = -1) : room(room_), door(door_) {}

    std::string str() const { return format("Port [room=%d, door=%d]", room, door); }

    bool operator==(const Port& rhs) const { return room == rhs.room && door == rhs.door; }
    bool operator<(const Port& rhs) const { return room == rhs.room ? door < rhs.door : room < rhs.room; }
    bool operator>(const Port& rhs) const { return room == rhs.room ? door > rhs.door : room > rhs.room; }

    inline int to_int() const { return room * 6 + door; }
    static Port from_int(int p) { return Port(p / 6, p % 6); }

};

//======================= 実マップ（地図の真値） =======================
struct Labyrinth {

    // NOTE: start index は 0 固定
    std::vector<int> labels;                 // 各部屋の2bitラベル (0..3)
    std::vector<std::array<Port, 6>> to;     // to[r][d] = (r', d') 扉dで移動後の(部屋, 扉)

    Room step(Room r, Door d) const {
        auto [nr, nd] = to[r][d];
        return nr;
    }

    // 仕様通り：各ステップで「入室後」の部屋ラベルを記録
    std::vector<int> explore_plan(const std::string& plan) const {
        Room cur = 0; // start index
        std::vector<int> out; out.reserve(plan.size() + 1);
        out.push_back(labels[cur]);
        for (char ch : plan) {
            int d = ch - '0';
            if (d < 0 || d > 5) throw std::runtime_error("plan contains non-door digit");
            cur = step(cur, d);
            out.push_back(labels[cur]);
        }
        return out;
    }

    std::string str() const {
        auto to_str = [](const std::array<Port, 6>& dest) {
            std::string s;
            for (int i = 0; i < 6; i++) {
                if (i) s += ",";
                s += "(" + std::to_string(dest[i].room) + "," + std::to_string(dest[i].door) + ")";
            }
            return s;
            };
        std::ostringstream oss;
        oss << "Labyrinth [\n";
        assert(labels.size() == to.size());
        for (size_t i = 0; i < labels.size(); i++) {
            oss << "    " << format("Room %3zd: ", i) << "label(" << labels[i] << "), dest=[" << to_str(to[i]) << "]\n";
        }
        oss << ']';
        return oss.str();
    }
};

//======================= 提出形式（/guess の map をモデル化） =======================
struct Connection {
    Port from, to;
    bool operator==(const Connection& rhs) const { return from == rhs.from && to == rhs.to; }
    bool operator<(const Connection& rhs) const { return from == rhs.from ? to < rhs.to : from < rhs.from; }
    bool operator>(const Connection& rhs) const { return from == rhs.from ? to > rhs.to : from > rhs.from; }
};
    
struct GuessMap {

    std::vector<int> rooms;                   // 2bit ラベル列
    Room startingRoom = 0;
    std::vector<Connection> connections; // 片側書けば自動で両方向を張る

    GuessMap() = default;

    GuessMap(const Labyrinth& L) {
        rooms = L.labels;
        startingRoom = 0; // fixed
        std::set<Connection> edge_pairs; // (r1, d1, r2, d2)
        const int n = (int)rooms.size();
        for (int r1 = 0; r1 < n; r1++) {
            for (int d1 = 0; d1 < 6; d1++) {
                Port p1(r1, d1), p2(L.to[r1][d1]);
                if (p1 > p2) continue;
                Connection conn(p1, p2);
                assert(!edge_pairs.count(conn));
                edge_pairs.insert(conn);
            }
        }
        connections.assign(edge_pairs.begin(), edge_pairs.end());
    }

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
        L.to = std::move(to);
        return L;
    }

    std::string to_json() const {
        // id を除いた json 化
        nlohmann::json j;
        j["rooms"] = rooms;
        j["startingRoom"] = startingRoom;
        j["connections"] = nlohmann::json::array();

        for (const auto& c : connections) {
            nlohmann::json jc;
            jc["from"] = { {"room", c.from.room}, {"door", c.from.door} };
            jc["to"] = { {"room", c.to.room},   {"door", c.to.door} };
            j["connections"].push_back(std::move(jc));
        }

        return j.dump();
    }

};

//======================= 等価性判定（挙動同値：積グラフBFS） =======================
bool equivalent(const Labyrinth& a, const Labyrinth& b) {
    if (a.labels.size() != b.labels.size()) return false; // 部屋数一致が必要条件

    std::queue<std::pair<Room, Room>> q;
    std::unordered_set<long long> seen; // (u,v) の組をエンコード
    auto enc = [](int u, int v)->long long { return ((long long)u << 32) ^ (unsigned long long)v; };

    q.emplace(0, 0);
    seen.insert(enc(0, 0));

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
Labyrinth generate_random_labyrinth(int n_rooms, std::optional<uint64_t> seed = std::nullopt) {

    std::mt19937_64 engine(seed.has_value() ? *seed : (uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count());
    Labyrinth L;

    L.labels.resize(n_rooms);
    for (int i = 0; i < n_rooms; ++i) L.labels[i] = (int)(engine() % 4); 
    //std::shuffle(L.labels.begin(), L.labels.end(), engine); // 均等ラベル

    // すべてのポート (r,d) を列挙
    std::vector<Port> ports; ports.reserve((size_t)n_rooms * 6);
    for (int r = 0; r < n_rooms; ++r) for (int d = 0; d < 6; ++d) ports.emplace_back(r, d);
    std::shuffle(ports.begin(), ports.end(), engine);

    // 双方向マッチング。少数の自己固定点 (p->p) を許容
    const double p_self = 0.04;
    std::vector<std::array<Port, 6>> to(n_rooms);
    for (int r = 0; r < n_rooms; ++r) for (int d = 0; d < 6; ++d) to[r][d] = Port{ -1,-1 };

    // 割当済みか判定するための map
    struct KeyHash { size_t operator()(const Port& p) const { return ((size_t)p.room << 3) ^ (size_t)p.door; } };
    std::unordered_map<Port, Port, KeyHash> pair_of;

    size_t i = 0;
    while (i < ports.size()) {
        Port p = ports[i++];
        if (pair_of.find(p) != pair_of.end()) continue;

        bool do_self = (double)(engine() % 1000000) / 1000000.0 < p_self;
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


namespace NLocalSearch {

    struct Delta {
        enum struct Type { LABEL, TWO_SWITCH };
        Type type;
        bool ok;
        int a, b;
    };

    struct State {
        
        std::vector<int> labels;
        std::vector<int> to;

        State() = default;
        State(const Labyrinth& L) : labels(L.labels), to(labels.size() * 6) {
            const int n = (int)labels.size();
            for (int r = 0; r < n; r++) {
                for (int d = 0; d < 6; d++) {
                    int p = r * 6 + d, q = L.to[r][d].to_int();
                    to[p] = q;
                }
            }
        }

        Delta modify_label(int idx, int to_label) {
            assert(1 <= idx && idx < (int)labels.size()); // start is fixed
            assert(0 <= to_label && to_label < 4);
            assert(labels[idx] != to_label);
            Delta d; d.type = Delta::Type::LABEL;
            int prev = labels[idx];
            labels[idx] = to_label;
            // return (idx, prev)
            d.ok = true;
            d.a = idx;
            d.b = prev;
            return d;
        }

        Delta modify_random_label(Xorshift& r) {
            const int n = (int)labels.size();
            int idx = r.next_u32(n - 1) + 1, to_label;
            do {
                to_label = r.next_u32(4);
            } while (to_label == labels[idx]);
            return modify_label(idx, to_label);
        }

        Delta modify_two_switch(int p, int q) {
            // p, q: compressed port
            Delta d; d.type = Delta::Type::TWO_SWITCH;
            assert(p < (int)to.size() && q < (int)to.size());
            assert(p != q); // 現行実装だと自己ループを含む操作の Undo で assertion error が起きる
            const int P = to[p], Q = to[q];
            //if (P == q || Q == p) {
            if (P == q || Q == p || p == P || q == Q || P == Q) {
                d.ok = false;
                return d;
            }
            to[p] = q; to[q] = p; to[P] = Q; to[Q] = P;
            // modify_two_switch(s, p, P) で元に戻る
            d.ok = true;
            d.a = p;
            d.b = P;
            return d;
        }

        Delta modify_random_two_switch(Xorshift& r) {
            const int n = (int)to.size(); // nrooms * 6
            int p = r.next_u32(n), q;
            do {
                q = r.next_u32(n);
            } while (p == q);
            return modify_two_switch(p, q);
        }

        Delta modify_random(Xorshift& r) {
            if (r.next_u32(2)) {
                return modify_random_label(r);
            }
            return modify_random_two_switch(r);
        }

        void undo(const Delta& d) {
            if (d.type == Delta::Type::LABEL) {
                modify_label(d.a, d.b);
            }
            else if (d.type == Delta::Type::TWO_SWITCH) {
                modify_two_switch(d.a, d.b);
            }
            else {
                assert(false);
            }
        }

        int calc_diff(const std::vector<int>& plan, const std::vector<int>& target_labels) const {
            assert(plan.size() + 1 == target_labels.size());
            int cur_room = 0; // start is fixed
            assert(target_labels.front() == labels[cur_room]);
            int diff = 0;
            for (int i = 1; i < (int)target_labels.size(); ++i) {
                cur_room = to[cur_room * 6 + plan[i - 1]] / 6;
                diff += labels[cur_room] != target_labels[i];
            }
            return diff;
        }

        bool operator==(const State& rhs) const {
            return labels == rhs.labels && to == rhs.to;
        }

    };

    Labyrinth to_labyrinth(const State& s) {
        Labyrinth L;
        const int n = L.labels.size();
        L.labels = s.labels;
        L.to.resize(L.labels.size());
        for (int r = 0; r < (int)L.labels.size(); r++) {
            for (int d = 0; d < 6; d++) {
                L.to[r][d] = Port::from_int(s.to[r * 6 + d]);
            }
        }
        // 全扉が定義済みか確認
        for (int r = 0; r < n; ++r)
            for (int d = 0; d < 6; ++d)
                if (L.to[r][d] == Port{ -1,-1 })
                    throw std::runtime_error("door undefined: room=" + std::to_string(r) + " door=" + std::to_string(d));
        for (int r = 0; r < n; ++r) {
            for (int d = 0; d < 6; ++d) {
                auto [r2, d2] = L.to[r][d];
                assert(r == L.to[r2][d2].room && d == L.to[r2][d2].door);
            }
        }
                
        return L;
    }

    GuessMap multistart_annealing(
        const int num_rooms,
        const std::vector<int>& plan,
        const std::vector<int>& target_labels
    ) {
        constexpr int multistart_iter = 100;
        constexpr int innerloop_iter = 1000000;
        constexpr double start_temp = 1.0;
        constexpr double end_temp = 0.01;

        Xorshift rnd;

        State global_best_state;
        int global_min_cost = INT_MAX;

        int trial;
        for (trial = 0; trial < multistart_iter; trial++) {
            if (!global_min_cost) break;

            ofs << format("--- trial %3d begin ---\n", trial);

            auto L = generate_random_labyrinth(num_rooms, rnd.next_u64());
            L.labels[0] = target_labels[0]; // 出発地点を 0 とし、そのラベルは固定

            State state(L); // Labyrinth -> State
            int cost = state.calc_diff(plan, target_labels);

            State best_state(state);
            int min_cost = cost;

            ofs << format("initial cost: %d\n", cost);

            int step;
            for (step = 0; step < innerloop_iter; step++) {
                auto delta = state.modify_random(rnd);
                if (!delta.ok) continue;
                int new_cost = state.calc_diff(plan, target_labels);
                int diff = new_cost - cost;
                double temp = get_temp(start_temp, end_temp, step, innerloop_iter);
                double prob = exp(-diff / temp);
                if (rnd.next_double() < prob) { // accepted
                    cost = new_cost;
                    if (chmin(min_cost, cost)) {
                        best_state = state;
                        ofs << format("step=%10d, min_cost=%4d, cost=%4d\n", step, min_cost, cost);
                        if (!min_cost) break;
                    }
                }
                else {
                    state.undo(delta);
                }
            }
            ofs << format("step=%10d, min_cost=%4d, cost=%4d\n", step, min_cost, cost);

            if (min_cost < global_min_cost) {
                ofs << format("global min cost updated: %4d -> %4d\n", global_min_cost, min_cost);
                global_min_cost = min_cost;
                global_best_state = best_state;
            }

            ofs << format("--- trial %3d end   ---\n", trial);
        }

        ofs << format("num trial: %3d, global min cost: %4d\n", trial, global_min_cost);

        GuessMap gm(to_labyrinth(global_best_state));
        return gm;
    }

}


int main() {

#ifdef _DEBUG
    WaitDebugger(INFINITE);
    ::DebugBreak();
#endif

    Timer timer;

    std::mt19937_64 engine(0);

    const int num_rooms = [&]() {
        std::string line;
        std::cin >> line;
        debug(line);
        assert(catalog.count(line));
        debug(catalog.at(line));
        return catalog.at(line);
    }();

    const int plan_length = num_rooms * 18;
    debug(plan_length);

    const auto plan_str = [&plan_length, &engine] () {
        std::string p;
        for (int i = 0; i < plan_length; i++) {
            p.push_back(char(i % 6) + '0');
        }
        std::shuffle(p.begin(), p.end(), engine);
        debug(p);
        return p;
    } ();
    
    // send query
    std::cout << 1 << std::endl; // single plan
    std::cout << plan_str << std::endl;

    // read labels
    const auto target_labels = []() {
        std::string labels_str;
        std::cin >> labels_str;
        std::vector<int> labels;
        for (char c : labels_str) {
            assert('0' <= c && c <= '3');
            labels.push_back(c - '0');
        }
        return labels;
        } ();
    debug(target_labels);

    std::vector<int> plan;
    for (char ch : plan_str) plan.push_back(ch - '0');
    debug(plan);

    auto gm = NLocalSearch::multistart_annealing(num_rooms, plan, target_labels);
    std::cout << gm.to_json() << std::endl;

    debug(timer.elapsed_ms());
	return 0;
}