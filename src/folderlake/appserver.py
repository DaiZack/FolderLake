"""FolderLake home server — the web-app front door.

``folderlake app`` starts this server and opens it immediately: no
folder picker, no scan before the page appears. The home page lists
every catalog built so far (from the SQLite registry), starts new
scans with live progress, refreshes or deletes old ones, and holds
the LLM settings.

Everything is served from this one port: each catalog's viewer lives
under ``/c/<catalog-id>/`` with its API at ``/c/<catalog-id>/api/…``
(the viewer HTML is rewritten to the prefix when served). A single
port keeps the app usable through SSH tunnels, Docker port mappings
and other one-port forwarding — an earlier design ran each catalog
viewer on its own random port, which broke exactly there.
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

from .catalog import CATALOG_NAME, scan, try_build_fts
from .cypher import build_cypher
from .hierarchy import build_hierarchy
from .registry import Registry
from .relations import build_relations
from .server import (ECHARTS_PATH, VIEWER_PATH, facets, file_detail,
                     graph, list_files, topics_tree)

HOME_PATH = Path(__file__).with_name("home.html")


class App:
    def __init__(self, host: str = "127.0.0.1", port: int = 8770,
                 registry: Registry | None = None):
        self.host = host
        self.registry = registry or Registry()
        self.jobs: dict[str, dict] = {}
        self._job_lock = threading.Lock()
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
        try:
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

    # -- catalogs under /c/<id>/ ----------------------------------------
    def catalog_rec(self, catalog_id: str) -> dict:
        rec = self.registry.get_catalog(catalog_id)
        if rec is None:
            raise KeyError("no such catalog")
        return rec

    def open_catalog(self, catalog_id: str) -> str:
        rec = self.catalog_rec(catalog_id)
        if not (Path(rec["path"]) / CATALOG_NAME).exists():
            raise KeyError("catalog file is missing — refresh to rebuild it")
        self.registry.touch_opened(catalog_id)
        return f"/c/{catalog_id}/"

    def catalog_viewer_html(self, catalog_id: str) -> str:
        """The viewer page with its API/asset paths rewritten to this
        catalog's prefix, plus the back-to-home link."""
        rec = self.catalog_rec(catalog_id)
        if not (Path(rec["path"]) / CATALOG_NAME).exists():
            raise KeyError("catalog file is missing — refresh to rebuild it")
        prefix = f"/c/{catalog_id}"
        page = VIEWER_PATH.read_text(encoding="utf-8")
        page = page.replace("/api/", prefix + "/api/")
        page = page.replace("/assets/", prefix + "/assets/")
        link = (
            '<a href="/" style="margin-left:auto;margin-right:12px;'
            'color:#4a80d0;text-decoration:none;font-size:13px;'
            'white-space:nowrap">← FolderLake Home</a>'
        )
        return page.replace("</header>", link + "</header>", 1)

    def delete_catalog(self, catalog_id: str) -> dict:
        running = self.running_job()
        rec = self.catalog_rec(catalog_id)
        if running is not None and running["path"] == rec["path"]:
            raise RuntimeError("a scan is running for this catalog")
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

    def _send_bytes(self, body: bytes, content_type: str,
                    disposition: str | None = None) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if disposition:
            self.send_header("Content-Disposition", disposition)
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

    def _serve_catalog(self, parsed) -> None:
        """GET /c/<id>/… — the catalog viewer and its API, same port."""
        app = self.app
        parts = parsed.path.split("/")  # ['', 'c', '<id>', <rest...>]
        cid = parts[2] if len(parts) > 2 else ""
        rest = "/".join(parts[3:])
        try:
            rec = app.catalog_rec(cid)
        except KeyError:
            self._send_json({"error": "no such catalog"}, 404)
            return
        db_path = Path(rec["path"]) / CATALOG_NAME
        query = urllib.parse.parse_qs(parsed.query)
        try:
            if rest in ("", "/"):
                self._send_html(app.catalog_viewer_html(cid))
                return
            if rest == "assets/echarts.min.js":
                if ECHARTS_PATH.exists():
                    self._send_bytes(
                        ECHARTS_PATH.read_bytes(),
                        "application/javascript; charset=utf-8")
                else:
                    self._send_json({"error": "echarts asset missing"}, 404)
                return
            if rest == "api/export.cypher":
                self._send_bytes(build_cypher(db_path).encode(),
                                 "text/plain; charset=utf-8",
                                 'attachment; filename="folderlake.cypher"')
                return
            if rest == "api/meta":
                self._send_json({
                    "folder": rec["path"],
                    "catalog": str(db_path),
                    "catalog_bytes": (db_path.stat().st_size
                                      if db_path.exists() else 0),
                })
                return
            if rest == "api/files":
                kinds = query.get("kind", [None])[0]
                self._send_json(list_files(
                    db_path,
                    kinds=kinds.split(",") if kinds else None,
                    topic=query.get("topic", [None])[0],
                    query=query.get("q", [None])[0],
                    node=query.get("node", [None])[0]))
                return
            if rest == "api/facets":
                self._send_json(facets(db_path))
                return
            if rest == "api/topics":
                self._send_json(topics_tree(db_path))
                return
            if rest == "api/graph":
                self._send_json(graph(db_path))
                return
            if rest.startswith("api/file/"):
                detail = file_detail(
                    db_path, urllib.parse.unquote(rest[len("api/file/"):]))
                if detail is None:
                    self._send_json({"error": "file not found"}, 404)
                else:
                    self._send_json(detail)
                return
        except KeyError as exc:
            self._send_json({"error": str(exc)}, 404)
            return
        self._send_json({"error": "not found"}, 404)

    def do_GET(self):  # noqa: N802 - stdlib name
        parsed = urllib.parse.urlparse(self.path)
        app = self.app
        if parsed.path == "/":
            self._send_html(HOME_PATH.read_text(encoding="utf-8"))
            return
        if parsed.path.startswith("/c/"):
            self._serve_catalog(parsed)
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
            except RuntimeError as exc:
                self._send_json({"error": str(exc)}, 409)
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
