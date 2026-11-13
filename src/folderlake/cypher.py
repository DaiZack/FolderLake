"""Export the catalog as a Neo4j Cypher script.

Graph model:
  (:File {id, name, path, kind, ext, size_bytes, title})
  (:Tag {name})            (:Entity {name})
  (File)-[:TAGGED {weight, source}]->(Tag)
  (File)-[:MENTIONS]->(Entity)
  (File)-[:RELATED {kind, detail}]->(File)

The script is idempotent (MERGE everywhere) and self-contained: run it with
cypher-shell or paste it into Neo4j Browser. Constraints use Neo4j 5 syntax.
"""

from __future__ import annotations

from pathlib import Path

import duckdb


def _q(value) -> str:
    """Quote a value as a Cypher literal."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    text = str(value).replace("\\", "\\\\").replace("'", "\\'")
    return f"'{text}'"


def build_cypher(db_path: Path) -> str:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        files = con.execute(
            "SELECT id, name, path, kind, ext, size_bytes, title FROM files ORDER BY name"
        ).fetchall()
        tags = con.execute(
            "SELECT file_id, tag, weight, source FROM file_topics ORDER BY file_id, weight DESC"
        ).fetchall()
        entities = con.execute(
            "SELECT file_id, entity FROM file_entities ORDER BY file_id, entity"
        ).fetchall()
        relations = con.execute(
            "SELECT file_a, file_b, kind, detail FROM relations ORDER BY file_a, file_b"
        ).fetchall()
    finally:
        con.close()

    lines = [
        "// FolderLake catalog export for Neo4j",
        f"// {len(files)} files, {len(tags)} tag links, "
        f"{len(entities)} entity links, {len(relations)} file relations",
        "// Run with cypher-shell or paste into Neo4j Browser. Safe to re-run (MERGE).",
        "",
        "CREATE CONSTRAINT file_id_unique IF NOT EXISTS FOR (f:File) REQUIRE f.id IS UNIQUE;",
        "CREATE CONSTRAINT tag_name_unique IF NOT EXISTS FOR (t:Tag) REQUIRE t.name IS UNIQUE;",
        "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE;",
        "",
        "// ---- Files ----",
    ]
    for fid, name, path, kind, ext, size, title in files:
        props = (
            f"id: {_q(fid)}, name: {_q(name)}, path: {_q(path)}, kind: {_q(kind)}, "
            f"ext: {_q(ext)}, size_bytes: {_q(size)}, title: {_q(title)}"
        )
        lines.append(f"MERGE (f:File {{id: {_q(fid)}}}) SET f += {{{props}}};")

    lines.append("")
    lines.append("// ---- Tags ----")
    tag_names = sorted({t[1] for t in tags})
    for name in tag_names:
        lines.append(f"MERGE (:Tag {{name: {_q(name)}}});")
    for fid, tag, weight, source in tags:
        lines.append(
            f"MATCH (f:File {{id: {_q(fid)}}}), (t:Tag {{name: {_q(tag)}}}) "
            f"MERGE (f)-[:TAGGED {{weight: {_q(round(weight, 4))}, source: {_q(source)}}}]->(t);"
        )

    lines.append("")
    lines.append("// ---- Entities ----")
    entity_names = sorted({e[1] for e in entities})
    for name in entity_names:
        lines.append(f"MERGE (:Entity {{name: {_q(name)}}});")
    for fid, entity in entities:
        lines.append(
            f"MATCH (f:File {{id: {_q(fid)}}}), (e:Entity {{name: {_q(entity)}}}) "
            "MERGE (f)-[:MENTIONS]->(e);"
        )

    lines.append("")
    lines.append("// ---- File relations ----")
    for a, b, kind, detail in relations:
        lines.append(
            f"MATCH (a:File {{id: {_q(a)}}}), (b:File {{id: {_q(b)}}}) "
            f"MERGE (a)-[:RELATED {{kind: {_q(kind)}, detail: {_q(detail)}}}]->(b);"
        )
    lines.append("")
    return "\n".join(lines)
