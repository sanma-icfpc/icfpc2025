from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

from ..context import AppCtx


def fmt_ts(ts: str) -> str:
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        disp = dt.strftime("%Y/%m/%d %H:%M:%S")
        delta = datetime.now(timezone.utc) - dt.astimezone(timezone.utc)
        sec = int(delta.total_seconds())
        if sec < 60:
            rel = f"{sec}秒前"
        elif sec < 3600:
            rel = f"{sec//60}分前"
        else:
            rel = f"{sec//3600}時間前"
        return f"{disp} ({rel})"
    except Exception:
        return ts


def flows(ctx: AppCtx, ch_id: int):
    rows = ctx.conn.execute(
        "SELECT api, req_key, phase, status_code, ts, req_body, res_body FROM challenge_requests WHERE challenge_id=? ORDER BY id ASC",
        (ch_id,),
    ).fetchall()
    d: Dict[str, Dict[str, Any]] = {}
    import json as _json
    for r in rows:
        key = r["req_key"] or f"{r['api']}-{r['ts']}"
        item = d.get(key)
        if not item:
            item = {
                "api": r["api"],
                "phases": set(),
                "ts": r["ts"],
                "codes": {},
                "req_key": key,
                "req_pretty": None,
                "res_pretty": None,
                "summary": "",
                "internal": False,
            }
            d[key] = item
        ph = r["phase"]
        item["phases"].add(ph)
        if r["api"] == "event":
            item["internal"] = True
        try:
            sc = int(r["status_code"]) if r["status_code"] is not None else 0
        except Exception:
            sc = 0
        item["codes"][ph] = sc
        if ph in ("agent_in", "to_upstream") and item["req_pretty"] is None:
            try:
                obj = _json.loads(r["req_body"]) if r["req_body"] else None
                item["req_pretty"] = _json.dumps(obj, ensure_ascii=False, indent=2) if obj is not None else (r["req_body"] or "")
            except Exception:
                item["req_pretty"] = r["req_body"] or ""
        if ph in ("from_upstream", "agent_out"):
            try:
                obj = _json.loads(r["res_body"]) if r["res_body"] else None
                item["res_pretty"] = _json.dumps(obj, ensure_ascii=False, indent=2) if obj is not None else (r["res_body"] or "")
            except Exception:
                item["res_pretty"] = r["res_body"] or ""
        try:
            if r["api"] == "select" and ph == "agent_in" and not item["summary"]:
                obj = _json.loads(r["req_body"]) if r["req_body"] else {}
                pn = obj.get("problemName")
                item["summary"] = f"problem={pn}" if pn else ""
            elif r["api"] == "explore" and ph == "from_upstream":
                robj = _json.loads(r["res_body"]) if r["res_body"] else {}
                results = robj.get("results") or []
                q = robj.get("queryCount")
                lens = [len(x) if isinstance(x, list) else 0 for x in results]
                item["summary"] = f"plans={len(lens)} results={lens} qc={q}"
            elif r["api"] == "guess" and ph == "from_upstream":
                robj = _json.loads(r["res_body"]) if r["res_body"] else {}
                ok = robj.get("correct")
                req = _json.loads(r["req_body"]) if r["req_body"] else {}
                m = req.get("map") or {}
                rooms = m.get("rooms") or []
                conns = m.get("connections") or []
                item["summary"] = f"rooms={len(rooms)} conns={len(conns)} correct={ok}"
            elif r["api"] == "event":
                try:
                    evs = sorted(list(item["phases"]))
                    if evs:
                        item["summary"] = "state change: " + ", ".join(evs)
                    else:
                        item["summary"] = "state change"
                except Exception:
                    item["summary"] = "state change"
        except Exception:
            pass
    out = list(d.values())
    out.sort(key=lambda x: x["ts"])  # oldest first
    return out

