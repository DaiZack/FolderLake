"""Local viewer server — Phase 2 (three-pane UI) and Phase 3 (graph + search).

Standard library only, no dependencies: a ThreadingHTTPServer serving one
single-file viewer page plus a small read-only JSON API over the catalog
built in Phase 1. Files are never modified by the server; every request
opens its own read-only DuckDB connection.

Search is a LIKE fallback over extracted text, name and summary. The
catalog's FTS extension (``try_build_fts``) cannot be installed inside the
Muse sandbox (egress proxy blocks the extension download), so search does
not depend on it — on a normal machine the scan still tries to build the
FTS index, but the API works either way.
"""

from __future__ import annotations

import html
import json
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import duckdb

from .cypher import build_cypher
from .preview import build_preview

VIEWER_PATH = Path(__file__).with_name("viewer.html")
ECHARTS_PATH = Path(__file__).parent / "assets" / "echarts.min.js"


def _connect(db_path: Path) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(db_path), read_only=True)


def _like_pattern(raw: str) -> str:
    """Escape a user string for use inside a LIKE pattern."""
    return "%" + raw.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


def list_files(db_path: Path, kinds: list[str] | None = None,
               topic: str | None = None, query: str | None = None,
               node: str | None = None) -> list[dict]:
    """Files matching the filters, each with its tags. Ordered by name."""
    con = _connect(db_path)
    try:
        sql = """
            SELECT f.id, f.name, f.ext, f.kind, f.size_bytes, f.mtime,
                   f.title, coalesce(string_agg(t.tag, ', '), '') AS tags
            FROM files f
            LEFT JOIN file_topics t ON t.file_id = f.id
        """
        where: list[str] = []
        params: list = []
        if kinds:
            where.append("f.kind IN (" + ",".join("?" * len(kinds)) + ")")
            params.extend(kinds)
        if topic:
            where.append(
                "EXISTS (SELECT 1 FROM file_topics t2 "
                "WHERE t2.file_id = f.id AND t2.tag = ?)"
            )
            params.append(topic)
        if node:
            where.append(
                "EXISTS (SELECT 1 FROM topic_node_files nf "
                "WHERE nf.file_id = f.id AND nf.node_id = ?)"
            )
            params.append(node)
        if query:
            where.append(
                "(f.name ILIKE ? ESCAPE '\\' "
                "OR EXISTS (SELECT 1 FROM file_text x WHERE x.file_id = f.id "
                "AND (x.summary ILIKE ? ESCAPE '\\' OR x.text ILIKE ? ESCAPE '\\')))"
            )
            pat = _like_pattern(query)
            params.extend([pat, pat, pat])
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " GROUP BY f.id, f.name, f.ext, f.kind, f.size_bytes, f.mtime, f.title ORDER BY f.name"
        rows = con.execute(sql, params).fetchall()
        return [
            {
                "id": r[0], "name": r[1], "ext": r[2], "kind": r[3],
                "size_bytes": r[4], "mtime": r[5], "title": r[6],
                "tags": [t for t in r[7].split(", ") if t] if r[7] else [],
            }
            for r in rows
        ]
    finally:
        con.close()


def facets(db_path: Path) -> dict:
    con = _connect(db_path)
    try:
        kinds = [
            {"kind": k, "count": c}
            for k, c in con.execute(
                "SELECT kind, count(*) FROM files GROUP BY kind ORDER BY count(*) DESC"
            ).fetchall()
        ]
        topics = [
            {"tag": t, "count": c}
            for t, c in con.execute(
                "SELECT tag, count(*) FROM file_topics GROUP BY tag "
                "ORDER BY count(*) DESC, tag"
            ).fetchall()
        ]
        total = con.execute("SELECT count(*) FROM files").fetchone()[0]
        return {"kinds": kinds, "topics": topics, "total_files": total}
    finally:
        con.close()


def topics_tree(db_path: Path) -> dict:
    """The hierarchical topic tree: nested nodes per kind root.

    Empty roots when no hierarchy has been built (the viewer then
    falls back to the flat facet topics).
    """
    con = _connect(db_path)
    try:
        try:
            rows = con.execute(
                "SELECT id, parent_id, kind, label, depth, file_count, "
                "cohesion FROM topic_nodes ORDER BY depth, label"
            ).fetchall()
        except Exception:
            return {"roots": []}
        nodes: dict[str, dict] = {}
        roots: list[dict] = []
        for nid, parent, kind, label, depth, count, cohesion in rows:
            nodes[nid] = {
                "id": nid, "kind": kind, "label": label, "depth": depth,
                "count": count, "cohesion": cohesion, "children": [],
            }
        for nid, parent, kind, label, depth, count, cohesion in rows:
            if parent is not None and parent in nodes:
                nodes[parent]["children"].append(nodes[nid])
            elif parent is None:
                roots.append(nodes[nid])
        return {"roots": roots}
    finally:
        con.close()


def graph(db_path: Path) -> dict:
    con = _connect(db_path)
    try:
        nodes = [
            {
                "id": r[0], "name": r[1], "kind": r[2],
                "tags": [t for t in r[3].split(", ") if t] if r[3] else [],
            }
            for r in con.execute(
                """
                SELECT f.id, f.name, f.kind,
                       coalesce(string_agg(t.tag, ', '), '')
                FROM files f LEFT JOIN file_topics t ON t.file_id = f.id
                GROUP BY f.id, f.name, f.kind ORDER BY f.name
                """
            ).fetchall()
        ]
        edges = [
            {"a": r[0], "b": r[1], "kind": r[2], "detail": r[3]}
            for r in con.execute(
                "SELECT file_a, file_b, kind, detail FROM relations"
            ).fetchall()
        ]
        return {"nodes": nodes, "edges": edges}
    finally:
        con.close()


def file_detail(db_path: Path, file_id: str) -> dict | None:
    preview = build_preview(db_path, file_id)
    if preview is None:
        return None
    con = _connect(db_path)
    try:
        preview["entities"] = [
            r[0]
            for r in con.execute(
                "SELECT entity FROM file_entities WHERE file_id = ?", [file_id]
            ).fetchall()
        ]
        preview["relations"] = [
            {"other_id": r[0], "kind": r[1], "detail": r[2]}
            for r in con.execute(
                """
                SELECT CASE WHEN file_a = ? THEN file_b ELSE file_a END,
                       kind, detail FROM relations
                WHERE file_a = ? OR file_b = ?
                """,
                [file_id, file_id, file_id],
            ).fetchall()
        ]
        preview["mtime"] = con.execute(
            "SELECT mtime FROM files WHERE id = ?", [file_id]
        ).fetchone()[0]
        return preview
    finally:
        con.close()


class _Handler(BaseHTTPRequestHandler):
    db_path: Path = Path("folderlake.duckdb")
    folder: Path = Path(".")

    def log_message(self, *args):  # keep the console quiet
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

    def _send_bytes(self, body: bytes, content_type: str,
                    disposition: str | None = None) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if disposition:
            self.send_header("Content-Disposition", disposition)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - stdlib name
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)
        db_path = self.db_path

        if parsed.path == "/":
            self._send_html(VIEWER_PATH.read_text(encoding="utf-8"))
            return
        if parsed.path == "/assets/echarts.min.js":
            if ECHARTS_PATH.exists():
                self._send_bytes(ECHARTS_PATH.read_bytes(),
                                 "application/javascript; charset=utf-8")
            else:
                self._send_json({"error": "echarts asset missing"}, 404)
            return
        if parsed.path == "/api/export.cypher":
            script = build_cypher(db_path)
            self._send_bytes(script.encode(),
                             "text/plain; charset=utf-8",
                             'attachment; filename="folderlake.cypher"')
            return
        if parsed.path == "/api/meta":
            self._send_json({
                "folder": str(self.folder),
                "catalog": str(db_path),
                "catalog_bytes": db_path.stat().st_size if db_path.exists() else 0,
            })
            return
        if parsed.path == "/api/files":
            kinds = query.get("kind", [None])[0]
            self._send_json(list_files(
                db_path,
                kinds=kinds.split(",") if kinds else None,
                topic=query.get("topic", [None])[0],
                query=query.get("q", [None])[0],
                node=query.get("node", [None])[0],
            ))
            return
        if parsed.path == "/api/facets":
            self._send_json(facets(db_path))
            return
        if parsed.path == "/api/topics":
            self._send_json(topics_tree(db_path))
            return
        if parsed.path == "/api/graph":
            self._send_json(graph(db_path))
            return
        if parsed.path.startswith("/api/file/"):
            file_id = urllib.parse.unquote(parsed.path[len("/api/file/"):])
            detail = file_detail(db_path, file_id)
            if detail is None:
                self._send_json({"error": f"not found: {html.escape(file_id)}"}, 404)
            else:
                self._send_json(detail)
            return
        self._send_json({"error": "not found"}, 404)


def create_server(db_path: Path, folder: Path, host: str = "127.0.0.1",
                  port: int = 0) -> ThreadingHTTPServer:
    """Build (but do not start) the viewer server. Port 0 picks a free port."""
    handler = type("FolderLakeHandler", (_Handler,), {
        "db_path": db_path, "folder": folder,
    })
    return ThreadingHTTPServer((host, port), handler)


def serve(db_path: Path, folder: Path, host: str = "127.0.0.1",
          port: int = 0) -> str:
    """Start the viewer server in the foreground; returns the URL."""
    server = create_server(db_path, folder, host, port)
    url = f"http://{host}:{server.server_address[1]}/"
    print(f"FolderLake viewer: {url}  (folder: {folder}, catalog: {db_path})")
    print("Press Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    return url
