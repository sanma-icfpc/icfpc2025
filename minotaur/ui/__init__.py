from __future__ import annotations

from flask import Flask
from ..context import AppCtx


def register_ui_blueprints(app: Flask, ctx: AppCtx) -> None:
    """Register modular UI blueprints. Call from register_ui_routes."""
    from .admin_db import create_admin_db_bp
    from .admin_sys import create_admin_sys_bp
    from .admin_threads import create_admin_threads_bp

    app.register_blueprint(create_admin_db_bp(ctx))
    app.register_blueprint(create_admin_sys_bp(ctx))
    app.register_blueprint(create_admin_threads_bp(ctx))
