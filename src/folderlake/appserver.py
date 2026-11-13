"""FolderLake home server — the web-app front door.

``folderlake app`` starts this server and opens it immediately: no
folder picker, no scan before the page appears. The home page lists
every catalog built so far (from the SQLite registry), starts new
scans with live progress, refreshes or deletes old ones, and holds
the LLM settings. Opening a catalog spins up the ordinary
single-catalog viewer server (in a thread, on its own port) with a
"← FolderLake Home" link injected into its header.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .catalog import CATALOG_NAME, scan
from .hierarchy import build_hierarchy
from .registry import Registry, catalog_id_for
from .relations import build_relations
from .server import create_server
from .catalog import try_build_fts

HOME_PATH = Path(__file__).with_name("home.html")


class App:
    def __init__(self, host: str = "127.0.0.1", port: int = 8770,
                 registry: Registry | None = None):
        self.host = host
        self.registry = registry or Registry()
        self.jobs: dict[str, dict] = {}
        self._job_lock = threading.Lock()
        self.viewers: dict[str, tuple[ThreadingHTTPServer, int]] = {}
        handler = type("AppHandler", (_Handler,), {"app": self})
        try:
            self.server = ThreadingHTTPServer((host, port), handler)
        except OSError:
            self.server = ThreadingHTTPServer((host, 0), handler)
        self.port = self.server.server_address[1]
        self.home_url = f"http://{host}:{self.port}/"

    # -- jobs ---------------------------------------------------------
    def running_job(self) -> dict | None:
        with self._job_lock:
            for j in self.jobs.values():
                if j["status"] == "running":
                    return j
        return None

    def start_scan(self, folder: Path, use_llm: bool) -> dict:
        if self.running_job() is not None:
            raise RuntimeError("a scan is already running")
        job = {
            "id": uuid.uuid4().hex[:12], "status": "running",
            "phase": "starting", "done": 0, "total": 0, "current": "",
            "message": "", "catalog_id": None, "path": str(folder),
        }
        with self._job_lock:
            self.jobs[job["id"]] = job
        threading.Thread(target=self._run_scan, args=(job, folder, use_llm),
                         daemon=True).start()
        return job

    def _apply_llm_env(self, use_llm: bool) -> dict:
        """Snapshot the settings a scan runs with; when the LLM is on,
        also publish them to the env the LLM provider layer reads."""
        s = self.registry.get_settings()
        snapshot = {
            "use_llm": bool(use_llm),
            "llm_provider": s["llm_provider"],
            "llm_model": s["llm_model"],
        }
        if use_llm:
            os.environ["FOLDERLAKE_LLM_PROVIDER"] = s["llm_provider"]
            if s["llm_model"]:
                os.environ["FOLDERLAKE_LLM_MODEL"] = s["llm_model"]
            if s["llm_base_url"]:
                os.environ["FOLDERLAKE_LLM_BASE_URL"] = s["llm_base_url"]
            key = self.registry.get_api_key()
            if key:
                os.environ["FOLDERLAKE_LLM_API_KEY"] = key
        return snapshot

    def _run_scan(self, job: dict, folder: Path, use_llm: bool) -> None:
        cid = catalog_id_for(folder)
        was_open = cid in self.viewers
        try:
            if was_open:
                self._stop_viewer(cid)  # DuckDB allows one writer
            snapshot = self._apply_llm_env(use_llm)
            db_path = folder / CATALOG_NAME

            def progress(done, total, name):
                job.update(phase="scanning", done=done, total=total,
                           current=name)

            stats = scan(folder, db_path, use_llm=use_llm,
                         progress=progress)
            job.update(phase="linking files", current="")
            build_relations(db_path)
            build_hierarchy(db_path)
            try_build_fts(db_path)
            rec = self.registry.upsert_catalog(
                folder, stats["seen"], db_path.stat().st_size, snapshot)
            job.update(status="done", phase="done",
                       catalog_id=rec["id"])
        except Exception as exc:  # surfaced on the home page, not a crash
            job.update(status="error", message=str(exc))
        finally:
            if was_open and job["status"] == "done":
                self.open_catalog(cid)

    # -- catalog viewers ------------------------------------------------
    def open_catalog(self, catalog_id: str) -> str:
        rec = self.registry.get_catalog(catalog_id)
        if rec is None:
            raise KeyError("no such catalog")
        if catalog_id in self.viewers:
            _, port = self.viewers[catalog_id]
            self.registry.touch_opened(catalog_id)
            return f"http://{self.host}:{port}/"
        folder = Path(rec["path"])
        db_path = folder / CATALOG_NAME
        if not db_path.exists():
            raise KeyError("catalog file is missing — refresh to rebuild it")
        server = create_server(db_path, folder, self.host, 0,
                               home_url=self.home_url)
        port = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.viewers[catalog_id] = (server, port)
        self.registry.touch_opened(catalog_id)
        return f"http://{self.host}:{port}/"

    def _stop_viewer(self, catalog_id: str) -> None:
        entry = self.viewers.pop(catalog_id, None)
        if entry is not None:
            server, _ = entry
            server.shutdown()
            server.server_close()

    def delete_catalog(self, catalog_id: str) -> dict:
        self._stop_viewer(catalog_id)
        rec = self.registry.delete_catalog(catalog_id)
        if rec is None:
            raise KeyError("no such catalog")
        db_path = Path(rec["path"]) / CATALOG_NAME
        for p in (db_path, Path(str(db_path) + ".wal")):
            try:
                p.unlink()
            except FileNotFoundError:
                pass
        return rec

    # -- folder dialog (subprocess: tkinter needs its own main thread) --
    @staticmethod
    def browse_folder() -> str:
        code = (
            "import tkinter\n"
            "from tkinter import filedialog\n"
            "root = tkinter.Tk(); root.withdraw()\n"
            "print(filedialog.askdirectory("
            "title='Choose a folder for FolderLake'))\n"
            "root.destroy()\n"
        )
        try:
            out = subprocess.run(
                [sys.executable, "-c", code], capture_output=True,
                text=True, timeout=600)
            return out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""
        except Exception:
            return ""


class _Handler(BaseHTTPRequestHandler):
    app: App

    def log_message(self, *args):
        pass

    def _send_json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, text: str) -> None:
        body = text.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return {}

    def do_GET(self):  # noqa: N802 - stdlib name
        parsed = urllib.parse.urlparse(self.path)
        app = self.app
        if parsed.path == "/":
            self._send_html(HOME_PATH.read_text(encoding="utf-8"))
            return
        if parsed.path == "/api/catalogs":
            self._send_json(app.registry.list_catalogs())
            return
        if parsed.path == "/api/settings":
            self._send_json(app.registry.get_settings())
            return
        if parsed.path.startswith("/api/jobs/"):
            job = app.jobs.get(parsed.path[len("/api/jobs/"):])
            if job is None:
                self._send_json({"error": "no such job"}, 404)
            else:
                self._send_json(job)
            return
        self._send_json({"error": "not found"}, 404)

    def do_POST(self):  # noqa: N802 - stdlib name
        parsed = urllib.parse.urlparse(self.path)
        app = self.app
        body = self._body()
        try:
            if parsed.path == "/api/scan":
                folder = Path(str(body.get("path", ""))).expanduser()
                if not folder.is_dir():
                    self._send_json(
                        {"error": f"not a folder: {folder}"}, 400)
                    return
                use_llm = bool(body.get("use_llm"))
                job = app.start_scan(folder.resolve(), use_llm)
                self._send_json({"id": job["id"]}, 202)
                return
            if parsed.path == "/api/browse":
                self._send_json({"path": app.browse_folder()})
                return
            if parsed.path == "/api/settings":
                self._send_json(app.registry.save_settings(body))
                return
            if parsed.path.startswith("/api/catalogs/") and \
                    parsed.path.endswith("/open"):
                cid = parsed.path.split("/")[3]
                self._send_json({"url": app.open_catalog(cid)})
                return
            if parsed.path.startswith("/api/catalogs/") and \
                    parsed.path.endswith("/refresh"):
                cid = parsed.path.split("/")[3]
                rec = app.registry.get_catalog(cid)
                if rec is None:
                    self._send_json({"error": "no such catalog"}, 404)
                    return
                snap = rec["settings"] or {}
                job = app.start_scan(
                    Path(rec["path"]), bool(snap.get("use_llm")))
                self._send_json({"id": job["id"]}, 202)
                return
        except KeyError as exc:
            self._send_json({"error": str(exc)}, 404)
            return
        except RuntimeError as exc:
            self._send_json({"error": str(exc)}, 409)
            return
        self._send_json({"error": "not found"}, 404)

    def do_DELETE(self):  # noqa: N802 - stdlib name
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path.startswith("/api/catalogs/"):
            cid = parsed.path.split("/")[3]
            try:
                self.app.delete_catalog(cid)
                self._send_json({"deleted": cid})
                return
            except KeyError:
                self._send_json({"error": "no such catalog"}, 404)
                return
        self._send_json({"error": "not found"}, 404)


def run_app(host: str = "127.0.0.1", port: int = 8770,
            open_browser: bool = True) -> int:
    app = App(host, port)
    print(f"FolderLake home: {app.home_url}", flush=True)
    print("Press Ctrl-C to stop.", flush=True)
    if open_browser:
        import webbrowser

        webbrowser.open(app.home_url)
    try:
        app.server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0
