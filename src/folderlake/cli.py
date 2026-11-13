"""FolderLake command line.

    folderlake scan <folder> [--db PATH] [--llm]
    folderlake list [--db PATH]
    folderlake preview <file-id-or-name> [--db PATH]
    folderlake serve <folder> [--db PATH] [--port N] [--open]

``serve`` is the one-command workflow from the README: it scans the
folder (incremental — unchanged files are skipped), builds relations,
and opens the three-pane viewer (filters | list/graph | preview).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import webbrowser
from pathlib import Path

from .catalog import CATALOG_NAME, connect, scan, try_build_fts
from .cypher import build_cypher
from .hierarchy import build_hierarchy
from .preview import build_preview
from .relations import build_relations
from .server import create_server


def _default_db(folder: Path) -> Path:
    return folder / CATALOG_NAME


def _run_server(folder: Path, db_path: Path, use_llm: bool, host: str,
                port: int, open_browser: bool) -> int:
    stats = scan(folder, db_path, use_llm=use_llm)
    build_relations(db_path)
    build_hierarchy(db_path)
    try_build_fts(db_path)
    server = create_server(db_path, folder.resolve(), host, port)
    url = f"http://{host}:{server.server_address[1]}/"
    print(f"scanned: {stats}", flush=True)
    print(f"FolderLake viewer: {url}  (folder: {folder}, catalog: {db_path})", flush=True)
    print("Press Ctrl-C to stop.", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


def _pick_folder() -> Path | None:
    """Native folder picker (stdlib tkinter); None if unavailable/cancelled."""
    try:
        import tkinter
        from tkinter import filedialog
    except Exception:
        return None
    try:
        root = tkinter.Tk()
        root.withdraw()
        chosen = filedialog.askdirectory(title="Choose a folder for FolderLake")
        root.destroy()
    except Exception:
        return None
    return Path(chosen) if chosen else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="folderlake")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_scan = sub.add_parser("scan", help="scan a folder into a DuckDB catalog")
    p_scan.add_argument("folder")
    p_scan.add_argument("--db", default=None)
    p_scan.add_argument("--llm", action="store_true",
                        help="use a local LLM (Ollama) for topics instead of rules")

    p_list = sub.add_parser("list", help="list catalogued files with topics")
    p_list.add_argument("--db", required=True)

    p_prev = sub.add_parser("preview", help="preview one file by id or name")
    p_prev.add_argument("target")
    p_prev.add_argument("--db", required=True)

    p_serve = sub.add_parser(
        "serve", help="scan a folder and open the local viewer")
    p_serve.add_argument("folder")
    p_serve.add_argument("--db", default=None)
    p_serve.add_argument("--llm", action="store_true",
                         help="use a local LLM (Ollama) for topics instead of rules")
    p_serve.add_argument("--port", type=int, default=0,
                         help="port to listen on (0 = pick a free port)")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--open", action="store_true",
                         help="open the viewer in the default browser")

    p_cypher = sub.add_parser(
        "cypher", help="export the catalog as a Neo4j Cypher script")
    p_cypher.add_argument("folder")
    p_cypher.add_argument("--db", default=None)
    p_cypher.add_argument("-o", "--output", default=None,
                          help="output file (default: <folder>/folderlake.cypher)")

    p_app = sub.add_parser(
        "app", help="no-shell mode: pick a folder in a dialog, open the viewer")
    p_app.add_argument("--llm", action="store_true",
                       help="use the configured LLM provider for topics")
    p_app.add_argument("--port", type=int, default=0,
                       help="port to listen on (0 = pick a free port)")
    p_app.add_argument("--host", default="127.0.0.1")

    p_mcp = sub.add_parser(
        "mcp", help="serve the catalog to other agents over MCP (stdio)")
    p_mcp.add_argument("folder", nargs="?", default=None,
                       help="folder whose catalog to serve")
    p_mcp.add_argument("--db", default=None,
                       help="catalog path (overrides the folder default)")

    args = parser.parse_args(argv)

    if args.cmd == "scan":
        folder = Path(args.folder)
        if not folder.is_dir():
            print(f"not a folder: {folder}", file=sys.stderr)
            return 2
        db_path = Path(args.db) if args.db else _default_db(folder)
        started = time.perf_counter()
        stats = scan(folder, db_path, use_llm=args.llm)
        elapsed = time.perf_counter() - started
        rel = build_relations(db_path)
        hier = build_hierarchy(db_path)
        fts = try_build_fts(db_path)
        print(json.dumps({
            **stats, "relations": rel, "hierarchy": hier, "fts_index": fts,
            "seconds": round(elapsed, 3),
            "catalog": str(db_path),
            "catalog_bytes": db_path.stat().st_size,
        }, indent=2))
        return 0

    if args.cmd == "list":
        con = connect(Path(args.db))
        try:
            rows = con.execute(
                """
                SELECT f.name, f.kind,
                       coalesce(string_agg(t.tag, ', '), '') AS tags
                FROM files f LEFT JOIN file_topics t ON t.file_id = f.id
                GROUP BY f.name, f.kind ORDER BY f.name
                """
            ).fetchall()
            for name, kind, tags in rows:
                print(f"{name}\t{kind}\t{tags}")
            print(f"-- {len(rows)} files")
        finally:
            con.close()
        return 0

    if args.cmd == "preview":
        con = connect(Path(args.db))
        try:
            row = con.execute(
                "SELECT id FROM files WHERE id = ? OR name = ? LIMIT 1",
                [args.target, args.target],
            ).fetchone()
        finally:
            con.close()
        if row is None:
            print(f"not found: {args.target}", file=sys.stderr)
            return 2
        print(json.dumps(build_preview(Path(args.db), row[0]), indent=2, ensure_ascii=False))
        return 0

    if args.cmd == "cypher":
        folder = Path(args.folder)
        db_path = Path(args.db) if args.db else _default_db(folder)
        if not db_path.exists():
            print(f"no catalog found: {db_path} (run scan first)", file=sys.stderr)
            return 2
        script = build_cypher(db_path)
        out = Path(args.output) if args.output else folder / "folderlake.cypher"
        out.write_text(script, encoding="utf-8")
        print(json.dumps({
            "cypher": str(out),
            "bytes": out.stat().st_size,
            "statements": script.count(";\n") + (1 if script.rstrip().endswith(";") else 0),
        }, indent=2))
        return 0

    if args.cmd == "serve":
        folder = Path(args.folder)
        if not folder.is_dir():
            print(f"not a folder: {folder}", file=sys.stderr)
            return 2
        db_path = Path(args.db) if args.db else _default_db(folder)
        return _run_server(folder, db_path, args.llm, args.host, args.port,
                           args.open)

    if args.cmd == "mcp":
        from .mcp_server import main as mcp_main

        if args.db:
            db_path = Path(args.db)
        elif args.folder:
            db_path = _default_db(Path(args.folder))
        else:
            print("folderlake mcp: give a folder or --db", file=sys.stderr)
            return 2
        if not db_path.exists():
            print(f"catalog not found: {db_path} (scan the folder first)",
                  file=sys.stderr)
            return 2
        return mcp_main(["--db", str(db_path)])

    if args.cmd == "app":
        folder = _pick_folder()
        if folder is None:
            print(
                "No folder chosen (or no desktop available for the picker). "
                "Fallback: folderlake serve <folder>",
                file=sys.stderr,
            )
            return 2
        return _run_server(folder, _default_db(folder), args.llm, args.host,
                           args.port, True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
