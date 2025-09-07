# Summary
minotaurはジャッジプロトコルのプロキシとして振る舞うことで、任意の言語・OS・マシンで実行される解法プログラム（エージェントと呼びます）からの要求をスケジューリングし、ステートフルな本番ジャッジサーバとのやりとりを直列化します。
スケジューリングはデフォルトで一連のやりとり（/select から /guess 提出まで）単位での先着順です。/guess は正誤に関わらず挑戦を終了します（正解なら status=correct、不正解なら status=incorrect）。優先度を設定したり、常に優先する (Pin) や差し込む (Select as Next) こともできます。
また仲介したやりとりを保存しており、簡易的な解析を提供します。SQLiteファイルをダウンロードすることで独自の解析も可能です。

# Run
`uv run waitress-serve --listen=*:19384 run:app`

# Client-Side
local_judge_server.py を参照し、すべての POST リクエストで以下を設定します
- X-Agent-Name HTTP ヘッダを付加: 解法を区別できる名前を付けます（例: t-suzuki:bruteforce-1）
- X-Agent-ID HTTP ヘッダを付加: 解法が同一であっても実行インスタンスを区別できる名前を付けます（例: プロセス ID）
- HTTP リクエストのタイムアウトを無制限または長時間に設定します。Minotaur は POST /select をいつでも受け付けますが、自分の番が来るまでその応答は待たされます。

### Minotaur専用: /minotaur_giveup
本番サーバには存在しない、Minotaur専用の補助APIです。エージェントが「この挑戦を自発的にやめる」ことを明示できます。
利用は任意ですが、Minotaur側でタイムアウトするまでの間のアイドルを減らすため、積極的に実施することを推奨します。

- エンドポイント: `POST /minotaur_giveup`
- ボディ: 何でも可（空オブジェクト `{}` で十分）
- 応答例: `{ "ok": true, "cancelled": true }`（対象がない場合は `cancelled:false`）
- 作用: 実行中の同一 `X-Agent-Name` の挑戦を即座に終了（内部状態は `terminated_running`）。キュー中のジョブには影響しません。

利用例（Python）:
```python
import requests

base = "http://127.0.0.1:19384"
headers = {
    "Content-Type": "application/json",
    "X-Agent-Name": "yourname:strategy-1",
}
res = requests.post(f"{base}/minotaur_giveup", json={}, headers=headers, timeout=10)
print(res.status_code, res.json())
```

## サンプルコード（非実動）
```python
# Minimal client for Minotaur (ICFPC 2025)
# Extracted and arranged from scripts/local_judge_server.py client-test

import http.client
import json
import os
import time

HOST = "127.0.0.1"    # Minotaur host
PORT = 19384          # Minotaur default port (see minotaur/README.md)
AGENT_NAME = "yourname:strategy-1"  # e.g., "t-suzuki:bruteforce-1"     ###################### HERE
AGENT_ID = str(os.getpid())         # distinguish processes/instances   ###################### HERE
TIMEOUT_SEC = 6000                  # long timeout for /select blocking ###################### HERE

def post_json(path: str, obj: dict):
    conn = http.client.HTTPConnection(HOST, PORT, timeout=TIMEOUT_SEC)
    conn.request(
        "POST",
        path,
        body=json.dumps(obj),
        headers={
            "Content-Type": "application/json",
            "X-Agent-Name": AGENT_NAME,  #################### HERE
            "X-Agent-ID": AGENT_ID,      #################### HERE
        },
    )
    resp = conn.getresponse()
    raw = resp.read().decode("utf-8", errors="replace")
    try:
        data = json.loads(raw)
    except Exception:
        data = {"_raw": raw}
    conn.close()
    return resp.status, data

# Example usage:

# 1) Enqueue + wait for grant (Minotaur blocks until your turn)
#    Note: "id" is ignored by Minotaur; kept for compatibility with local judge.
status, sel = post_json("/select", {"problemName": "primus", "id": "ignored"})
print("select:", status, sel)

# 2) Explore with batched plans
status, ex = post_json("/explore", {"plans": ["0", "123", "505"]})
print("explore:", status, ex)

# 3) Submit a guess (example builds a trivial 1-room guess like the local judge demo)
#    Switch to the 1-room toy problem and auto-solve it:
#    Note: /guess を呼ぶと正誤に関わらず挑戦は終了扱いになります（正解時は correct、
#    不正解時は incorrect として記録）。次のエージェントにスロットが解放されます。
status, _ = post_json("/select", {"problemName": "superdumb", "id": "ignored"})
status, ex = post_json("/explore", {"plans": ["0"]})
label = ex.get("results", [[0]])[0][0]
guess = {
    "rooms": [label],
    "startingRoom": 0,
    "connections": [
        {"from": {"room": 0, "door": d}, "to": {"room": 0, "door": d}} for d in range(6)
    ],
}
status, g = post_json("/guess", {"map": guess})
print("guess:", status, g)

# 4) If your solver decides to give up mid-run without calling /guess,
#    you can notify Minotaur so the slot is freed immediately.
#    This endpoint does not exist on the official server.
try:
    _ = post_json("/minotaur_giveup", {})  ###################### HERE
except Exception:
    pass

```

# Control panel
minotaurの管理画面にアクセスします
- Schedulerでスケジューラの優先度などを設定します。
  - Pinはそのエージェントの要求がある限り最優先して扱い続けます。
  - Select as Nextは今の挑戦が終わったあとに選ぶべきエージェントを指定します。
  - Priority は優先度を1(最低)～100(最高)で指定します。
  - Delete は優先度設定を消去します。再度接続があれば再びエントリが作成されます。ジョブ自体を削除するものではありません。
- Running / Queued / Recent はそれぞれの状態の挑戦を表示します
- Analytics は完了した挑戦の結果(queryCountなど)の統計を、エージェント毎に集計します
- Admin は管理向けです
- Auto Updateをオフにすると自動更新を止めることが出来ます。オンにすることで再開します。

# Technology
- Flask + waitress + vanilla JS Server-Sent Events
