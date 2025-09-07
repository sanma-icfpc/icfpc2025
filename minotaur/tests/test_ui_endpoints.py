import os
import base64
import tempfile


def _auth_header(user: str = "test", pwd: str = "test") -> dict:
    tok = base64.b64encode(f"{user}:{pwd}".encode("utf-8")).decode("ascii")
    return {"Authorization": f"Basic {tok}"}


def _with_test_app():
    # Configure in-memory DB and a temporary auth file before importing app
    os.environ.setdefault("MINOTAUR_DB_IN_MEMORY", "1")
    fd, path = tempfile.mkstemp(prefix="users_", suffix=".yaml")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        f.write("- {username: test, password: test}\n")
    os.environ["AUTH_FILE"] = path
    # Import after env is set so app picks up our config
    from minotaur.app import app as flask_app  # type: ignore
    return flask_app, path


def test_panes_and_admin_endpoints_smoke():
    app, auth_path = _with_test_app()
    try:
        c = app.test_client()
        h = _auth_header()

        # Panes
        assert c.get("/minotaur/pane_scheduler", headers=h).status_code == 200
        assert c.get("/minotaur/pane_running", headers=h).status_code == 200
        assert c.get("/minotaur/pane_queued", headers=h).status_code == 200
        assert c.get("/minotaur/pane_recent?recent_page=1", headers=h).status_code == 200
        assert c.get("/minotaur/pane_analytics", headers=h).status_code == 200

        # Admin diagnostics
        assert c.get("/minotaur/fd_info", headers=h).status_code == 200
        assert c.get("/minotaur/mem_info", headers=h).status_code == 200
        assert c.get("/minotaur/threads_os", headers=h).status_code == 200

        # Events / counters
        assert c.get("/minotaur/agent_count", headers=h).status_code == 200

        # Combined status
        assert c.get("/minotaur/status?recent_page=1", headers=h).status_code == 200
    finally:
        try:
            os.remove(auth_path)
        except Exception:
            pass

