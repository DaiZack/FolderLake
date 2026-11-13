"""MCP server — let other AI agents query a FolderLake catalog.

The tool handlers are plain Python functions, independent of any SDK,
registered in the ``TOOLS`` dict below (name -> (description, handler)).
``serve_stdio`` wires them up through the official ``mcp`` Python SDK,
imported lazily inside the function: the SDK is an optional dependency
(``pip install folderlake[mcp]``), never a hard one — the same pattern
the LLM layer uses with LiteLLM.

Everything is read-only: handlers reuse the viewer's read APIs
(``server.list_files`` / ``facets`` / ``file_detail``) or open the
catalog read-only themselves. Results are capped and long text fields
truncated — agent context is precious.

Run it with:

    python -m folderlake.mcp_server --db /path/to/folderlake.duckdb

or set ``FOLDERLAKE_DB``. A typical MCP client config:

    {"mcpServers": {"folderlake": {
        "command": "python", "args": ["-m", "folderlake.mcp_server"],
        "env": {"FOLDERLAKE_DB": "/path/to/folderlake.duckdb"}}}}
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable

import duckdb

from .server import facets, file_detail, list_files

MAX_TEXT = 500          # summaries / text previews are cut to this
MAX_DETAIL = 200        # relation details are short labels; cap anyway
DEFAULT_LIMIT = 50
MAX_LIMIT = 200
MAX_RELATED = 100
MAX_COLUMNS = 50


def _truncate(text, limit: int = MAX_TEXT) -> str:
    if text is None:
        return ""
    text = str(text)
    return text if len(text) <= limit else text[:limit] + "…"


def _file_summary(row: dict) -> dict:
    """Slim a list_files row down to what an agent needs to triage."""
    return {
        "id": row["id"],
        "name": row["name"],
        "kind": row["kind"],
        "ext": row["ext"],
        "size_bytes": row["size_bytes"],
        "title": row.get("title"),
        "tags": row.get("tags", []),
    }


def _paged(rows: list[dict], limit: int) -> dict:
    return {
        "count": len(rows),
        "returned": min(len(rows), limit),
        "truncated": len(rows) > limit,
        "files": [_file_summary(r) for r in rows[:limit]],
    }


# ----------------------------------------------------------------------
# Tool handlers — plain functions: (db_path, **args) -> JSON-able result
# ----------------------------------------------------------------------
def _tool_stats(db_path: Path) -> dict:
    f = facets(db_path)
    return {
        "total_files": f["total_files"],
        "kinds": f["kinds"],
        "topic_count": len(f["topics"]),
    }


def _tool_search(db_path: Path, query: str, kind: str | None = None) -> dict:
    rows = list_files(db_path, kinds=[kind] if kind else None, query=query)
    result = _paged(rows, MAX_LIMIT)
    result["query"] = query
    return result


def _tool_list_files(db_path: Path, kind: str | None = None,
                     topic: str | None = None,
                     limit: int = DEFAULT_LIMIT) -> dict:
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = DEFAULT_LIMIT
    limit = max(1, min(limit, MAX_LIMIT))
    rows = list_files(db_path, kinds=[kind] if kind else None, topic=topic)
    return _paged(rows, limit)


def _tool_file_detail(db_path: Path, file_id: str) -> dict:
    detail = file_detail(db_path, file_id)
    if detail is None:
        return {"error": f"not found: {file_id}"}
    out: dict = {
        "id": detail["id"],
        "name": detail["name"],
        "path": detail["path"],
        "kind": detail["kind"],
        "size_bytes": detail["size_bytes"],
        "title": detail.get("title"),
        "author": detail.get("author"),
        "summary": _truncate(detail.get("summary")),
        "tags": detail.get("tags", []),
        "entities": detail.get("entities", []),
    }
    if "text_preview" in detail:
        out["text_preview"] = _truncate(detail["text_preview"])
    if "data_review" in detail:
        review = detail["data_review"]
        out["data_review"] = {
            "columns": review.get("columns", [])[:MAX_COLUMNS],
            "row_count": review.get("row_count"),
            "nulls": dict(list((review.get("nulls") or {}).items())[:MAX_COLUMNS]),
            "summarize": review.get("summarize", [])[:MAX_COLUMNS],
            "sample_query": review.get("sample_query"),
        }
    return out


def _tool_related(db_path: Path, file_id: str) -> dict:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = con.execute(
            """
            SELECT other_id, kind, detail, name, other_kind FROM (
                SELECT CASE WHEN r.file_a = ? THEN r.file_b ELSE r.file_a END
                           AS other_id,
                       r.kind AS kind, r.detail AS detail,
                       f.name AS name, f.kind AS other_kind
                FROM relations r
                JOIN files f
                  ON f.id = CASE WHEN r.file_a = ? THEN r.file_b ELSE r.file_a END
                WHERE r.file_a = ? OR r.file_b = ?
            ) ORDER BY kind, name LIMIT ?
            """,
            [file_id, file_id, file_id, file_id, MAX_RELATED],
        ).fetchall()
    finally:
        con.close()
    return {
        "file_id": file_id,
        "count": len(rows),
        "related": [
            {
                "file_id": r[0],
                "name": r[3],
                "kind": r[4],
                "relation_kind": r[1],
                "detail": _truncate(r[2], MAX_DETAIL),
            }
            for r in rows
        ],
    }


def _topic_tree(db_path: Path) -> dict | None:
    """Nested topic hierarchy from topic_nodes, or None if unavailable.

    The hierarchy tables are built by a separate step and may not exist
    in a catalog; any problem at all falls back to the flat topics.
    """
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        exists = con.execute(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_name = 'topic_nodes'"
        ).fetchone()[0]
        if not exists:
            return None
        rows = con.execute(
            "SELECT id, parent_id, kind, label, file_count FROM topic_nodes"
        ).fetchall()
        if not rows:
            return None  # table exists but no hierarchy was built
    except Exception:
        return None
    finally:
        con.close()
    nodes: dict[str, dict] = {
        r[0]: {
            "id": r[0], "label": r[3], "kind": r[2],
            "file_count": r[4], "children": [],
        }
        for r in rows
    }
    roots: list[dict] = []
    for r in rows:
        node = nodes[r[0]]
        parent = nodes.get(r[1]) if r[1] else None
        if parent is None:
            roots.append(node)
        else:
            parent["children"].append(node)
    return {"mode": "hierarchy", "roots": roots}


def _tool_topics(db_path: Path) -> dict:
    tree = _topic_tree(db_path)
    if tree is not None:
        return tree
    return {"mode": "flat", "topics": facets(db_path)["topics"]}


TOOLS: dict[str, tuple[str, Callable[..., dict]]] = {
    "folderlake_stats": (
        "Catalog overview: total file count, counts per kind "
        "(document/data/text), and the number of distinct topics.",
        _tool_stats,
    ),
    "folderlake_search": (
        "Full-text search over file names, summaries and extracted text. "
        "Args: query (required), kind (optional: document, data or text).",
        _tool_search,
    ),
    "folderlake_list_files": (
        "List catalogued files, optionally filtered by kind and/or a "
        "topic tag. Args: kind, topic, limit (default 50, max 200).",
        _tool_list_files,
    ),
    "folderlake_file_detail": (
        "Detail for one file by id: tags, entities, summary, a text "
        "preview for documents, and a data-review block (schema, row "
        "count, sample query) for data files.",
        _tool_file_detail,
    ),
    "folderlake_related": (
        "Files related to the given file id, from the catalog's "
        "relations (shared topic/entity, explicit reference, same "
        "folder), each with the relation kind and detail.",
        _tool_related,
    ),
    "folderlake_topics": (
        "The topic organisation of the catalog: the hierarchical topic "
        "tree when one has been built, otherwise the flat topic list "
        "with per-topic file counts.",
        _tool_topics,
    ),
}


# ----------------------------------------------------------------------
# SDK wiring (optional dependency, imported lazily)
# ----------------------------------------------------------------------
def serve_stdio(db_path: Path) -> None:
    """Serve the TOOLS handlers over MCP stdio via the official SDK."""
    server_cls = None
    try:  # mcp 1.x
        from mcp.server.fastmcp import FastMCP

        server_cls = FastMCP
    except ImportError:
        try:  # mcp >= 2: FastMCP was renamed MCPServer (same shape)
            from mcp.server.mcpserver import MCPServer

            server_cls = MCPServer
        except ImportError:
            pass
    if server_cls is None:
        raise RuntimeError(
            "The MCP SDK is not installed. Install it with: "
            "pip install folderlake[mcp]"
        )

    server = server_cls("folderlake")

    @server.tool(description=TOOLS["folderlake_stats"][0])
    def folderlake_stats() -> dict:
        return _tool_stats(db_path)

    @server.tool(description=TOOLS["folderlake_search"][0])
    def folderlake_search(query: str, kind: str | None = None) -> dict:
        return _tool_search(db_path, query, kind)

    @server.tool(description=TOOLS["folderlake_list_files"][0])
    def folderlake_list_files(kind: str | None = None,
                              topic: str | None = None,
                              limit: int = DEFAULT_LIMIT) -> dict:
        return _tool_list_files(db_path, kind, topic, limit)

    @server.tool(description=TOOLS["folderlake_file_detail"][0])
    def folderlake_file_detail(file_id: str) -> dict:
        return _tool_file_detail(db_path, file_id)

    @server.tool(description=TOOLS["folderlake_related"][0])
    def folderlake_related(file_id: str) -> dict:
        return _tool_related(db_path, file_id)

    @server.tool(description=TOOLS["folderlake_topics"][0])
    def folderlake_topics() -> dict:
        return _tool_topics(db_path)

    server.run()  # stdio transport by default


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="folderlake-mcp",
        description="Serve a FolderLake catalog to other agents over MCP (stdio).",
    )
    parser.add_argument(
        "--db", default=None,
        help="path to the catalog (default: $FOLDERLAKE_DB or ./folderlake.duckdb)",
    )
    args = parser.parse_args(argv)
    raw = args.db or os.environ.get("FOLDERLAKE_DB") or "folderlake.duckdb"
    db_path = Path(raw)
    if not db_path.exists():
        print(f"catalog not found: {db_path}", file=sys.stderr)
        return 2
    serve_stdio(db_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
