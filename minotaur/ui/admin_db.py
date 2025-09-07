from __future__ import annotations

from flask import Blueprint, jsonify, send_file
from .. import db as dbm
from ..context import AppCtx
import os


def create_admin_db_bp(ctx: AppCtx) -> Blueprint:
    bp = Blueprint("ui_admin_db", __name__, url_prefix="/minotaur")
    guard = ctx.ui_guard

    @bp.route("/download_db")
    @guard.require()
    def download_db():
        import sqlite3, tempfile, io
        try:
            src_path = ctx.s.db_path
            use_uri = src_path.startswith("file:") or ("mode=memory" in src_path) or src_path.startswith("sqlite:")
            src = sqlite3.connect(src_path, uri=use_uri, check_same_thread=False)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".sqlite") as tf:
                tmp_path = tf.name
            dst = sqlite3.connect(tmp_path)
            try:
                src.backup(dst)
            finally:
                try: dst.close()
                except Exception: pass
                try: src.close()
                except Exception: pass
            with open(tmp_path, "rb") as f:
                data = f.read()
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            bio = io.BytesIO(data)
            bio.seek(0)
            return send_file(bio, as_attachment=True, download_name="coordinator.sqlite", mimetype="application/x-sqlite3")
        except Exception as e:
            return jsonify({"error": "download_failed", "detail": str(e)}), 500

    @bp.route("/db_info")
    @guard.require()
    def db_info():
        try:
            dbp = ctx.s.db_path
            if dbp.startswith("file:") and ("mode=memory" in dbp):
                return jsonify({"in_memory": True}), 200
            fs_path = dbp
            if dbp.startswith("file:"):
                fs_path = dbp[5:]
                q = fs_path.find('?')
                if q != -1:
                    fs_path = fs_path[:q]
                if not os.path.isabs(fs_path):
                    from ..config import BASE_DIR
                    fs_path = os.path.join(BASE_DIR, fs_path)
            if not os.path.isfile(fs_path):
                return jsonify({"error": "not_found"}), 404
            size_bytes = 0
            try:
                size_bytes += os.path.getsize(fs_path)
            except Exception:
                pass
            wal = fs_path + "-wal"
            shm = fs_path + "-shm"
            try:
                if os.path.isfile(wal):
                    size_bytes += os.path.getsize(wal)
            except Exception:
                pass
            try:
                if os.path.isfile(shm):
                    size_bytes += os.path.getsize(shm)
            except Exception:
                pass
            size_mb = round(size_bytes / (1024 * 1024), 1)
            try:
                mtime = os.path.getmtime(fs_path)
            except Exception:
                mtime = None
            return jsonify({
                "path": fs_path,
                "sizeBytes": int(size_bytes),
                "sizeMB": float(size_mb),
                "mtime": mtime,
            })
        except Exception:
            return jsonify({"error": "stat_failed"}), 500

    @bp.route("/db_close_conns", methods=["POST"])
    @guard.require()
    def db_close_conns():
        try:
            stats = dbm.close_all_connections()
            try:
                if ctx.coord and ctx.coord.on_change:
                    ctx.coord.on_change("db_close_conns")
            except Exception:
                pass
            return jsonify({"ok": True, **stats})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    @bp.route("/db_conn_stats")
    @guard.require()
    def db_conn_stats():
        try:
            st = dbm.conn_stats()
            return jsonify({"ok": True, **st})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    return bp

