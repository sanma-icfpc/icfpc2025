# Minotaur — ICFPC submit proxy and scheduler

## Purpose
- Transparent proxy for official endpoints: `/select`, `/explore`, `/guess`.
- Serialize trials per team; add priority scheduling and lease/timeout.
- Log all HTTP requests/responses; provide a small UI for status and settings.

## Requirements
- Python: 3.11+
- Packages: `flask`, `requests`, `pyyaml` (optionally `waitress` or `gunicorn` for prod)

## Install
- uv: `uv venv && uv pip install flask requests pyyaml`
- pip: `python -m venv venv && source venv/bin/activate && pip install flask requests pyyaml`

## Run
- Preferred (CWD = `minotaur/`):
  - Dev server: `python -m minotaur.app` (requires running from repo root) or `uv run python -m minotaur.app` from repo root.
  - Production (waitress): `uv run waitress-serve --listen=*:19384 run:app` or `python -m waitress --listen=*:19384 run:app`
- From repo root:
  - Dev server: `python -m minotaur.app`
  - Production (waitress): `uv run waitress-serve --listen=*:19384 minotaur.app:app` or `python -m waitress --listen=*:19384 minotaur.app:app`
- UI: open `http://localhost:19384/` (Basic Auth required)
- Agents: call `/select`, `/explore`, `/guess` as in production against this server.

## Notes & Decisions
- Removed internal MOCK mode: always forward to `OFFICIAL_BASE`. For local testing, set it to your local judge URL.
- Settings persistence: YAML only (`SETTINGS_FILE`, default `minotaur/settings.yaml`). SQLite `settings` table removed.
- Startup reap: on boot, any lingering `running` or `queued` trials are marked as `error` and move to Recent.
- Preemption: if the same `X-Agent-ID` issues `/select`, the previous run for that agent is force-finished and the new ticket is granted immediately.
  Additionally, when someone is `running` and there is no other queued trial (besides the current request), the new `/select` preempts to keep flow snappy.
- Logging: directional stdout logs with timestamps
  - `[agent -> minotaur]`, `[agent <- minotaur]`, `[minotaur -> upstream]`, `[upstream -> minotaur]`, `[webui -> minotaur]`, `[webui <- minotaur]`.
  - JSONL per-day logs are stored under `LOG_DIR`.
- UI updates: switched Status auto-refresh to vanilla EventSource (SSE) + fetch
  - We avoid HTMX SSE extension because outerHTML swaps can re-trigger `load`/`revealed` and cause extra GETs.
  - A single SSE connection (`/minotaur/stream`) triggers `fetch('/minotaur/status')` on change events.
- Local judge client-test now sends `X-Agent-ID: local_judge_server:<pid>` for better attribution and preemption behavior.

## Paths & Working Directory
- All runtime files (logs, database, auth file) default to the `minotaur/` directory, regardless of the current working directory.
- Environment variables `LOG_DIR`, `AUTH_FILE` accept absolute paths; if relative, they are resolved under `minotaur/`.
- When running from `minotaur/` via `waitress-serve`, use `run:app` (provided by `minotaur/run.py`).

## Configuration (env vars)
- `ICFP_ID`: Team id (required, never commit)
- `OFFICIAL_BASE`: Upstream base URL（テスト時はローカルジャッジを指定）
- `PORT`: Listen port (default `19384`)
- `LOG_DIR`: JSONL logs directory (default `./logs`)
- `MINOTAUR_DB`: SQLite path (default `./coordinator.sqlite`)
- `SETTINGS_FILE`: YAML settings file (default `./minotaur/settings.yaml`)
- `TRIAL_TTL_SEC`: Lease seconds per trial (default `60`)
- Scheduling knobs: `BASE_PRIORITY_DEFAULT` (50), `AGEING_EVERY_MIN` (5), `AGEING_CAP` (100),
  `POST_SUCCESS_RETRY_CAP` (3), `POST_SUCCESS_PENALTY` (-10)
- `AUTH_FILE`: YAML users file (default `./minotaur/users.yaml`)

## UI Authentication
- File: `minotaur/users.yaml` (YAML list of users)

```yaml
- username: admin
  password: adminpass
```

- Only UI (`/`, `/minotaur/*`) is protected; agent endpoints remain open for transparency.

## Endpoints (agent-visible; transparent)
- `POST /select {id?:string, problemName:string}` → forwards upstream; replaces `id` with `ICFP_ID`.
- `POST /explore {id?:string, plans:[string]}` → forwards; extends lease and stores `queryCount`.
- `POST /guess {id?:string, map:{...}}` → forwards; on `correct:true` marks `success`.
- Behavior:
  - If another trial is running, `/select` blocks until granted (or may return `202` queued after a long wait).
  - 全ての呼び出しは上流（OFFICIAL_BASE）へ透過転送されます。

## Admin/Health (UI)
- `GET /` → Dashboard (requires Basic Auth)
- `GET /minotaur/status` → HTMX partial (running/queued/recent)
- `POST /minotaur/settings` → ランタイム設定を更新（サブセット）。YAML (`SETTINGS_FILE`) にのみ永続化。起動時にYAMLを読み込んで環境デフォルトを上書きします。
- `GET /minotaur/healthz` → Basic health check

## Scheduling & Lease
- At most one `running` trial via SQLite constraint and IMMEDIATE tx.
- Effective priority = base (default 50) + ageing (+1/5min, capped 100).
- Grant order: `effective_priority DESC, enqueued_at ASC`.
- Lease: `TRIAL_TTL_SEC` (default 60s); updated on `/explore` and `/guess`.
- Timeout: `lease_expire_at < now` → status `timeout`.

## Security
- UI Basic Auth only; choose strong plaintext passwords in `users.yaml`.
- Agent endpoints are intentionally unauthenticated to preserve transparency; rely on serialized execution and optional rate limiting.

## Decisions
- Document accepted `X-` headers here and keep them up to date.

## Headers
- Request (Agent → Minotaur):
  - `X-Agent-ID`: optional string; stored with trial metadata.
  - `X-Agent-Git`: optional string; Git SHA or identifier stored with trial metadata.
- Response (Minotaur → Agent):
  - none (no custom X-headers are emitted).

## Logging
- JSONL logs per day at `${LOG_DIR}/http-YYYYMMDD.jsonl`.
- Fields: `dir`, `path`, `session`, `status_code`, `req_body`, `res_body`, `ts`.

## Data Model (SQLite)
- `trials(id, ticket, problem_id, params_json, agent_name, git_sha,`
  `base_priority, effective_priority, status, enqueued_at, started_at, finished_at,`
  `lease_expire_at, session_id, upstream_query_count)`
  ※ 設定はSQLiteではなくYAML永続化（`SETTINGS_FILE`）。

## Sequence (blocking transparency)
- Agent A → `/select`
- Minotaur: grant if idle, then upstream `/select`; return 200 to A.
- Agent B → `/select` while A is running → queued; HTTP blocks until A finishes.
- A → `/explore`×m, `/guess`×n; if upstream `correct:true` then `success`.
- B gets granted and proceeds.

## Deployment
- Windows: `waitress-serve --port=%PORT% minotaur.app:app`
- Linux/macOS: `gunicorn -k gthread -w 1 -b 0.0.0.0:${PORT:-19384} minotaur.app:app`
- Back up `coordinator.sqlite` and `logs/` daily.

## TODO
- NEXT
- NOT NOW
  - Extend settings UI to all scheduling knobs and rate limit.
  - Enhance `scripts/local_judge_server.py --client-test` to send `X-Agent-ID` for integration testing.
  - QPS limiter (global or per problem).
  - Dashboard metrics (success rate, avg time, counts).
