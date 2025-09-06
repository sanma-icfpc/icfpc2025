# Run
`uv run waitress-serve --listen=*:19384 run:app`

# Client-Side
local_judge_server.py を参照し、以下のようにします
- X-Agent-Name HTTPヘッダを付加: 解法を区別できる名前名付けます。例) t-suzuki:bruteforce-1
- X-Agent-ID HTTPヘッダを付加: 解法が同一であっても実行インスタンスを区別できる名前を付けます。例) プロセスID
- HTTPリクエストのタイムアウトを無制限または長時間に設定します。minotaurはPOST /selectをいつでも受付ますが、自分の番が来るまでその応答は待たされます。

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
