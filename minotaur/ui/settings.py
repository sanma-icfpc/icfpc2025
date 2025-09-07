from __future__ import annotations

from flask import Blueprint, Response, render_template, request
from ..context import AppCtx
from ..config import resolve_under_base, save_persisted_settings


def create_settings_bp(ctx: AppCtx) -> Blueprint:
    bp = Blueprint("ui_settings", __name__, url_prefix="/minotaur")
    guard = ctx.ui_guard

    @bp.route("/settings", methods=["GET", "POST"])
    @guard.require()
    def settings():
        if request.method == "POST":
            try:
                ob = request.form.get("OFFICIAL_BASE")
                if ob is not None:
                    ctx.s.official_base = ob or None
                ttl = request.form.get("TRIAL_TTL_SEC")
                if ttl is not None:
                    ctx.s.trial_ttl_sec = int(ttl)
                lg = request.form.get("LOG_DIR")
                if lg is not None:
                    ctx.s.log_dir = resolve_under_base(lg)
                rr = request.form.get("RSS_REBOOT_MB")
                if rr is not None:
                    try:
                        ctx.s.rss_reboot_mb = int(rr)
                    except Exception:
                        pass
                save_persisted_settings(
                    ctx.s.settings_file,
                    {
                        "OFFICIAL_BASE": ctx.s.official_base,
                        "TRIAL_TTL_SEC": ctx.s.trial_ttl_sec,
                        "LOG_DIR": ctx.s.log_dir,
                        "RSS_REBOOT_MB": ctx.s.rss_reboot_mb,
                    },
                )
            except Exception:
                pass
            if ctx.coord and ctx.coord.on_change:
                try:
                    ctx.coord.on_change("settings")
                except Exception:
                    pass
            return Response(status=204)
        return render_template(
            "settings_form.html",
            official_base=ctx.s.official_base,
            trial_ttl_sec=ctx.s.trial_ttl_sec,
            log_dir=ctx.s.log_dir,
            rss_reboot_mb=ctx.s.rss_reboot_mb,
        )

    return bp
