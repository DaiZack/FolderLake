"""Four simple, labelled relation kinds between files. Nothing fancier.

shared_topic / shared_entity / reference / same_folder — every edge
stores its reason in `detail`, so the graph never shows a mystery link.
Implemented as plain SQL over the catalog tables.
"""

from __future__ import annotations

import re
from pathlib import Path

import duckdb

# Dataset names that are also ordinary English words: a capitalised
# mention of these is too weak to count as a subject reference.
_COMMON_STEMS = {"tips", "flights", "exercise", "attention", "data",
                 "file", "files", "table"}


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
        ref_pairs: set[tuple[str, str]] = set()
        for fa, fb, name in rows:
            a, b = (fa, fb) if fa < fb else (fb, fa)
            ref_pairs.add((a, b))
            con.execute(
                "INSERT INTO relations VALUES (?,?, 'reference', ?)", [a, b, name]
            )
        # subject reference: a text/document that mentions a data
        # file's subject by its distinctive name links to that file —
        # a Wikipedia article about the Titanic points at titanic.csv.
        # The mention must be capitalised in the text ("Titanic",
        # "Penguins", "MPG"), which keeps ordinary lowercase uses of
        # common words from creating junk links.
        data_files = con.execute(
            "SELECT id, name FROM files WHERE kind = 'data'"
        ).fetchall()
        texts = con.execute(
            """
            SELECT t.file_id, t.text FROM file_text t
            JOIN files f ON f.id = t.file_id
            WHERE f.kind IN ('text', 'document')
            """
        ).fetchall()
        for data_id, data_name in data_files:
            stem = Path(data_name).stem.lower()
            if len(stem) < 3 or stem in _COMMON_STEMS:
                continue
            forms = {stem.capitalize(), stem.upper(), stem.title()}
            pat = re.compile(
                r"\b(?:" + "|".join(re.escape(f) for f in forms) + r")\b")
            for file_id, text in texts:
                if file_id == data_id or not pat.search(text or ""):
                    continue
                a, b = ((file_id, data_id) if file_id < data_id
                        else (data_id, file_id))
                if (a, b) in ref_pairs:
                    continue
                ref_pairs.add((a, b))
                con.execute(
                    "INSERT INTO relations VALUES (?,?, 'reference', ?)",
                    [a, b, f"mentions {stem}"],
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
