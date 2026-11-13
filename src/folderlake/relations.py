"""Four simple, labelled relation kinds between files. Nothing fancier.

shared_topic / shared_entity / reference / same_folder — every edge
stores its reason in `detail`, so the graph never shows a mystery link.
Implemented as plain SQL over the catalog tables.
"""

from __future__ import annotations

from pathlib import Path

import duckdb


def build_relations(db_path: Path) -> int:
    con = duckdb.connect(str(db_path))
    try:
        con.execute("DELETE FROM relations")
        con.execute(
            """
            INSERT INTO relations
            SELECT a.file_id, b.file_id, 'shared_topic', a.tag
            FROM file_topics a JOIN file_topics b
              ON a.tag = b.tag AND a.file_id < b.file_id
            """
        )
        con.execute(
            """
            INSERT INTO relations
            SELECT a.file_id, b.file_id, 'shared_entity', a.entity
            FROM file_entities a JOIN file_entities b
              ON a.entity = b.entity AND a.file_id < b.file_id
            """
        )
        # explicit reference: file A's text mentions file B's name
        rows = con.execute(
            """
            SELECT a.file_id, b.id, b.name
            FROM file_text a JOIN files b ON a.file_id <> b.id
            WHERE a.text ILIKE '%' || b.name || '%'
            """
        ).fetchall()
        for fa, fb, name in rows:
            a, b = (fa, fb) if fa < fb else (fb, fa)
            con.execute(
                "INSERT INTO relations VALUES (?,?, 'reference', ?)", [a, b, name]
            )
        con.execute(
            """
            INSERT INTO relations
            SELECT a.id, b.id, 'same_folder',
                   regexp_replace(a.path, '/[^/]*$', '')
            FROM files a JOIN files b ON a.id < b.id
            WHERE regexp_replace(a.path, '/[^/]*$', '')
                = regexp_replace(b.path, '/[^/]*$', '')
            """
        )
        return con.execute("SELECT count(*) FROM relations").fetchone()[0]
    finally:
        con.close()
