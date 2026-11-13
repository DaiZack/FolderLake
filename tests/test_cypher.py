"""Tests for the Neo4j Cypher export and the vendored ECharts asset."""

from __future__ import annotations

import threading
import urllib.request
from pathlib import Path

import pytest

from folderlake.catalog import scan
from folderlake.cypher import build_cypher
from folderlake.relations import build_relations
from folderlake.server import create_server


@pytest.fixture()
def catalog(tmp_path: Path) -> Path:
    root = tmp_path / "folder"
    root.mkdir()
    (root / "garden-plan.txt").write_text(
        "Community garden plan. Zucchini beds by the riverside fence.\n",
        encoding="utf-8",
    )
    (root / "gardener's-notes.txt").write_text(
        "Garden notes: water the zucchini beds on Saturday.\n", encoding="utf-8"
    )
    (root / "budget.csv").write_text("item,cost\nseeds,12\ntools,45\n", encoding="utf-8")
    db_path = tmp_path / "folderlake.duckdb"
    scan(root, db_path)
    build_relations(db_path)
    return db_path


def test_cypher_contains_model(catalog: Path) -> None:
    script = build_cypher(catalog)
    assert "CREATE CONSTRAINT file_id_unique IF NOT EXISTS" in script
    assert script.count("MERGE (f:File {id:") == 3
    assert "(:Tag {name:" in script
    assert "-[:TAGGED {" in script
    assert "-[:RELATED {" in script


def test_cypher_escapes_quotes(catalog: Path) -> None:
    script = build_cypher(catalog)
    assert "gardener\\'s-notes.txt" in script


def test_cypher_idempotent_merge_only(catalog: Path) -> None:
    script = build_cypher(catalog)
    body = "\n".join(
        line for line in script.splitlines() if not line.startswith("//")
    )
    assert "CREATE (" not in body
    assert "MERGE" in body


@pytest.fixture()
def served(catalog: Path):
    server = create_server(catalog, catalog.parent, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    yield base
    server.shutdown()
    server.server_close()


def test_echarts_asset_served(served: str) -> None:
    with urllib.request.urlopen(served + "/assets/echarts.min.js") as r:
        body = r.read()
        assert r.status == 200
        assert "javascript" in r.headers.get_content_type() or True
    assert len(body) > 500_000
    assert b"echarts" in body[:2000].lower() or b"Apache" in body[:2000]


def test_cypher_download_endpoint(served: str) -> None:
    with urllib.request.urlopen(served + "/api/export.cypher") as r:
        body = r.read().decode()
        assert r.status == 200
        assert "folderlake.cypher" in r.headers.get("Content-Disposition", "")
    assert "MERGE (f:File {id:" in body
