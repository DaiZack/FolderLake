import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from folderlake.appserver import App
from folderlake.catalog import scan
from folderlake.registry import Registry


# -- registry ---------------------------------------------------------

def test_registry_catalogs_and_settings(tmp_path):
    reg = Registry(tmp_path / "reg.sqlite3")
    folder = tmp_path / "docs"
    folder.mkdir()
    rec = reg.upsert_catalog(folder, 12, 3456,
                             {"use_llm": True, "llm_provider": "ollama",
                              "llm_model": "llama3.1"})
    assert rec["file_count"] == 12
    assert rec["settings"]["llm_model"] == "llama3.1"
    assert reg.list_catalogs()[0]["id"] == rec["id"]
    # rescan updates in place, no duplicate
    reg.upsert_catalog(folder, 13, 4000, {"use_llm": False})
    cats = reg.list_catalogs()
    assert len(cats) == 1 and cats[0]["file_count"] == 13
    assert cats[0]["settings"]["use_llm"] is False

    s = reg.save_settings({"llm_provider": "openai", "llm_api_key": "sk-x",
                           "use_llm": True})
    assert s["llm_provider"] == "openai" and s["use_llm"] is True
    assert s["llm_api_key_set"] is True
    # a blank key field keeps the saved key
    s2 = reg.save_settings({"llm_api_key": ""})
    assert s2["llm_api_key_set"] is True
    assert reg.get_api_key() == "sk-x"

    gone = reg.delete_catalog(rec["id"])
    assert gone is not None and reg.list_catalogs() == []


def test_scan_progress_callback(tmp_path):
    (tmp_path / "a.txt").write_text("alpha beta gamma", encoding="utf-8")
    (tmp_path / "b.txt").write_text("delta epsilon zeta", encoding="utf-8")
    calls = []
    scan(tmp_path, tmp_path / "c.duckdb",
         progress=lambda d, t, n: calls.append((d, t, n)))
    assert calls, "progress was never called"
    assert calls[-1][0] == calls[-1][1] == 2  # finishes at done == total


# -- app server ---------------------------------------------------------

class _Client:
    def __init__(self, app: App):
        self.base = app.home_url.rstrip("/")

    def call(self, method, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            self.base + path, data=data, method=method,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    def get_text(self, url):
        with urllib.request.urlopen(url, timeout=30) as r:
            return r.read().decode()


def _wait_job(client: _Client, job_id: str, timeout=60) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        _, job = client.call("GET", f"/api/jobs/{job_id}")
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.15)
    raise AssertionError("job did not finish")


def test_app_server_flow(tmp_path):
    folder = tmp_path / "proj"
    folder.mkdir()
    (folder / "notes.txt").write_text(
        "Riverside workshop notes. Alex Chen runs the workshop.",
        encoding="utf-8")
    (folder / "data.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    reg = Registry(tmp_path / "reg.sqlite3")
    app = App(port=0, registry=reg)
    thread = threading.Thread(target=app.server.serve_forever, daemon=True)
    thread.start()
    client = _Client(app)
    try:
        home = client.get_text(app.home_url)
        assert "Your catalogs" in home and "Settings" in home

        status, _ = client.call("POST", "/api/scan",
                                {"path": str(tmp_path / "missing")})
        assert status == 400

        status, started = client.call(
            "POST", "/api/scan", {"path": str(folder), "use_llm": False})
        assert status == 202
        job = _wait_job(client, started["id"])
        assert job["status"] == "done", job

        _, cats = client.call("GET", "/api/catalogs")
        assert len(cats) == 1
        cat = cats[0]
        assert cat["file_count"] == 2
        assert cat["settings"]["use_llm"] is False  # snapshot recorded
        assert (folder / "folderlake.duckdb").exists()

        status, opened = client.call(
            "POST", f"/api/catalogs/{cat['id']}/open")
        assert status == 200
        # Same port, path-based: no second server is involved, so the
        # app works through single-port forwarding (SSH tunnels etc.).
        assert opened["url"] == f"/c/{cat['id']}/"
        viewer = client.get_text(client.base + opened["url"])
        assert "FolderLake Home" in viewer  # back-to-home link injected
        assert f"/c/{cat['id']}/api/" in viewer  # API paths prefixed
        _, files = client.call("GET", opened["url"] + "api/files")
        assert len(files) == 2
        cypher = client.get_text(client.base + opened["url"]
                                 + "api/export.cypher")
        assert cypher.startswith("// FolderLake")

        # refresh picks up a new file
        (folder / "more.txt").write_text("workshop schedule update",
                                         encoding="utf-8")
        status, started = client.call(
            "POST", f"/api/catalogs/{cat['id']}/refresh")
        assert status == 202
        job = _wait_job(client, started["id"])
        assert job["status"] == "done", job
        _, cats = client.call("GET", "/api/catalogs")
        assert cats[0]["file_count"] == 3

        # settings roundtrip through the API
        status, s = client.call("POST", "/api/settings",
                                {"llm_provider": "anthropic",
                                 "use_llm": True})
        assert status == 200 and s["llm_provider"] == "anthropic"
        _, s = client.call("GET", "/api/settings")
        assert s["use_llm"] is True

        # delete removes the record and the catalog file
        status, _ = client.call("DELETE", f"/api/catalogs/{cat['id']}")
        assert status == 200
        _, cats = client.call("GET", "/api/catalogs")
        assert cats == []
        assert not (folder / "folderlake.duckdb").exists()
    finally:
        app.server.shutdown()


def test_browse_folder_returns_string():
    # No desktop in the test environment: must degrade to "" quietly.
    assert isinstance(App.browse_folder(), str)
