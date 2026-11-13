"""Preview payload for one file.

Documents/text: summary + extracted text.
Data files get a small side feature: schema, row count, per-column nulls,
and a ready-to-run DuckDB query — a review, not an analytics product.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb


def build_preview(db_path: Path, file_id: str) -> dict | None:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        row = con.execute(
            "SELECT path, name, kind, size_bytes, title, author, extra_json "
            "FROM files WHERE id = ?",
            [file_id],
        ).fetchone()
        if row is None:
            return None
        path, name, kind, size, title, author, extra_json = row
        extra = json.loads(extra_json or "{}")
        text_row = con.execute(
            "SELECT summary, text FROM file_text WHERE file_id = ?", [file_id]
        ).fetchone()
        summary, text = text_row if text_row else ("", "")
        tags = [
            r[0]
            for r in con.execute(
                "SELECT tag FROM file_topics WHERE file_id = ? ORDER BY weight DESC",
                [file_id],
            ).fetchall()
        ]
        preview: dict = {
            "id": file_id, "name": name, "path": path, "kind": kind,
            "size_bytes": size, "title": title, "author": author,
            "summary": summary, "tags": tags,
        }
        if kind == "data":
            escaped = path.replace("'", "''")
            preview["data_review"] = {
                "columns": extra.get("columns", []),
                "row_count": extra.get("row_count"),
                "nulls": extra.get("nulls", {}),
                "summarize": extra.get("summarize", []),
                "sample_query": f"SELECT * FROM '{escaped}' LIMIT 100;",
            }
        else:
            preview["text_preview"] = (text or "")[:4000]
        return preview
    finally:
        con.close()
