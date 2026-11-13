"""The DuckDB catalog: five tables, incremental scanning.

Incremental rule: a file whose (size, mtime) matches the stored row is
skipped on stat alone — no hashing, no extraction. Only new or changed
files are hashed and processed. That is what makes re-opening a folder
near-instant.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import duckdb

from .extractors import extract
from .topics import compute_topics, term_counts_for

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    id VARCHAR PRIMARY KEY,
    path VARCHAR, name VARCHAR, ext VARCHAR, kind VARCHAR,
    size_bytes BIGINT, mtime DOUBLE, hash VARCHAR,
    title VARCHAR, author VARCHAR, extra_json VARCHAR
);
CREATE TABLE IF NOT EXISTS file_topics (
    file_id VARCHAR, tag VARCHAR, weight DOUBLE, source VARCHAR
);
CREATE TABLE IF NOT EXISTS file_entities (
    file_id VARCHAR, entity VARCHAR
);
CREATE TABLE IF NOT EXISTS relations (
    file_a VARCHAR, file_b VARCHAR, kind VARCHAR, detail VARCHAR
);
CREATE TABLE IF NOT EXISTS file_text (
    file_id VARCHAR, summary VARCHAR, text VARCHAR
);
CREATE TABLE IF NOT EXISTS file_terms (
    file_id VARCHAR, term VARCHAR, cnt BIGINT
);
CREATE TABLE IF NOT EXISTS file_topic_meta (
    file_id VARCHAR PRIMARY KEY, hash VARCHAR, method VARCHAR
);
CREATE TABLE IF NOT EXISTS topic_nodes (
    id VARCHAR PRIMARY KEY, parent_id VARCHAR, kind VARCHAR,
    label VARCHAR, depth INTEGER, file_count INTEGER, cohesion DOUBLE
);
CREATE TABLE IF NOT EXISTS topic_node_files (
    node_id VARCHAR, file_id VARCHAR
);
"""

CATALOG_NAME = "folderlake.duckdb"


def file_id_for(path: Path) -> str:
    return hashlib.md5(str(path).encode()).hexdigest()[:16]


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def connect(db_path: Path) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(db_path))
    con.execute(SCHEMA)
    try:  # catalogs built before the source column existed
        con.execute("ALTER TABLE file_topics ADD COLUMN IF NOT EXISTS source VARCHAR")
    except Exception:
        pass
    return con


def _iter_files(root: Path, catalog_path: Path):
    # The catalog itself and the generated Neo4j export are not content.
    catalog_names = {catalog_path.name, catalog_path.name + ".wal", "folderlake.cypher"}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.name in catalog_names:
            continue
        if any(part.startswith(".") for part in path.relative_to(root).parts):
            continue
        yield path


def scan(root: Path, db_path: Path, use_llm: bool = False) -> dict:
    root = root.resolve()
    con = connect(db_path)
    stats = {"seen": 0, "added": 0, "updated": 0, "skipped": 0, "removed": 0}
    existing = {
        row[0]: (row[1], row[2])
        for row in con.execute("SELECT path, size_bytes, mtime FROM files").fetchall()
    }
    seen_paths: set[str] = set()
    changed_ids: set[str] = set()
    try:
        # One transaction for the whole scan: committing per file costs a
        # disk sync per statement and made a 1,000-file scan take ~194s;
        # batched, the same scan is extraction-bound instead.
        con.execute("BEGIN TRANSACTION")
        for path in _iter_files(root, db_path):
            stats["seen"] += 1
            key = str(path)
            seen_paths.add(key)
            st = path.stat()
            prev = existing.get(key)
            if prev is not None and prev[0] == st.st_size and abs(prev[1] - st.st_mtime) < 1e-6:
                stats["skipped"] += 1
                continue
            fid = file_id_for(path)
            result = extract(path)
            con.execute("DELETE FROM files WHERE id = ?", [fid])
            con.execute("DELETE FROM file_topics WHERE file_id = ?", [fid])
            con.execute("DELETE FROM file_entities WHERE file_id = ?", [fid])
            con.execute("DELETE FROM file_text WHERE file_id = ?", [fid])
            con.execute("DELETE FROM file_terms WHERE file_id = ?", [fid])
            con.execute("DELETE FROM file_topic_meta WHERE file_id = ?", [fid])
            con.execute(
                "INSERT INTO files VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                [
                    fid, key, path.name, path.suffix.lower(), result["kind"],
                    st.st_size, st.st_mtime, _hash_file(path),
                    result["title"], result["author"],
                    json.dumps(result["extra"], ensure_ascii=False),
                ],
            )
            # Term counts are the incremental base for the post-scan
            # TF-IDF step: unchanged files keep theirs, so adding a
            # file never re-reads the rest of the folder.
            for term, cnt in term_counts_for(result, path.name).items():
                con.execute("INSERT INTO file_terms VALUES (?,?,?)", [fid, term, cnt])
            con.execute(
                "INSERT INTO file_text VALUES (?,?,?)",
                [fid, "", result["text"]],
            )
            changed_ids.add(fid)
            stats["updated" if prev is not None else "added"] += 1
        for old_path, _ in [(p, None) for p in existing if p not in seen_paths]:
            fid = file_id_for(Path(old_path))
            for tbl, col in (("files", "id"), ("file_topics", "file_id"),
                             ("file_entities", "file_id"), ("file_text", "file_id"),
                             ("file_terms", "file_id"), ("file_topic_meta", "file_id")):
                con.execute(f"DELETE FROM {tbl} WHERE {col} = ?", [fid])
            stats["removed"] += 1
        con.execute("COMMIT")
        # Post-scan topic pass on the same connection: TF-IDF over
        # stored term counts (no file I/O for unchanged files); the
        # LLM, when enabled, is called only for files added or updated
        # in this scan.
        stats["topics"] = compute_topics(
            use_llm=use_llm, llm_file_ids=changed_ids, con=con
        )
    finally:
        con.close()
    return stats


def try_build_fts(db_path: Path) -> bool:
    """Best-effort full-text index over extracted text (used by search later)."""
    con = duckdb.connect(str(db_path))
    try:
        con.execute("INSTALL fts; LOAD fts;")
        con.execute("PRAGMA create_fts_index('file_text', 'file_id', 'text', overwrite=1)")
        return True
    except Exception:
        return False
    finally:
        con.close()
