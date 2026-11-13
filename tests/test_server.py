"""Tests for the Phase 2/3 viewer server: list, filters, search, graph, preview."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from folderlake.catalog import scan
from folderlake.relations import build_relations
from folderlake.server import create_server


@pytest.fixture()
def served(tmp_path: Path):
    root = tmp_path / "folder"
    root.mkdir()
    (root / "garden-plan.txt").write_text(
        "Community garden plan. Zucchini beds by the riverside fence.\n", encoding="utf-8"
    )
    (root / "meeting-notes.txt").write_text(
        "Community garden meeting: water the zucchini beds on Saturday.\n", encoding="utf-8"
    )
    (root / "budget.csv").write_text("item,cost\nseeds,12\ntools,45\n", encoding="utf-8")
    db_path = tmp_path / "folderlake.duckdb"
    scan(root, db_path)
    build_relations(db_path)
    server = create_server(db_path, root, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    yield base, db_path
    server.shutdown()
    server.server_close()


def _get(base: str, path: str):
    with urllib.request.urlopen(base + path, timeout=10) as resp:
        return resp.status, resp.read().decode()


def _get_json(base: str, path: str):
    status, body = _get(base, path)
    assert status == 200
    return json.loads(body)


def test_viewer_page_serves_three_panes(served):
    base, _ = served
    status, body = _get(base, "/")
    assert status == 200
    assert "FolderLake" in body
    assert 'id="file-list"' in body
    assert 'id="graph-chart"' in body
    assert "/assets/echarts.min.js" in body
    assert "/api/export.cypher" in body
    assert 'id="preview-pane"' in body


def test_files_list_and_kind_filter(served):
    base, _ = served
    files = _get_json(base, "/api/files")
    assert len(files) == 3
    assert {f["name"] for f in files} == {"garden-plan.txt", "meeting-notes.txt", "budget.csv"}
    data = _get_json(base, "/api/files?kind=data")
    assert [f["name"] for f in data] == ["budget.csv"]
    text = _get_json(base, "/api/files?kind=text")
    assert len(text) == 2


def test_topic_filter_and_facets(served):
    base, _ = served
    facets = _get_json(base, "/api/facets")
    assert facets["total_files"] == 3
    assert any(t["tag"] == "garden" for t in facets["topics"])
    garden = _get_json(base, "/api/files?topic=garden")
    assert len(garden) == 2
    assert {f["name"] for f in garden} == {"garden-plan.txt", "meeting-notes.txt"}


def test_search_finds_text_without_fts(served):
    # LIKE fallback: no FTS index is built here, search must still work.
    base, _ = served
    hits = _get_json(base, "/api/files?q=zucchini")
    assert {f["name"] for f in hits} == {"garden-plan.txt", "meeting-notes.txt"}
    assert _get_json(base, "/api/files?q=seeds") == [
        f for f in _get_json(base, "/api/files") if f["name"] == "budget.csv"
    ]
    assert _get_json(base, "/api/files?q=nosuchword") == []


def test_file_preview_and_data_review(served):
    base, _ = served
    files = {f["name"]: f for f in _get_json(base, "/api/files")}
    doc = _get_json(base, f"/api/file/{files['garden-plan.txt']['id']}")
    assert "zucchini" in doc["summary"].lower() or "zucchini" in doc["text_preview"].lower()
    assert doc["tags"]
    data = _get_json(base, f"/api/file/{files['budget.csv']['id']}")
    assert [c["name"] for c in data["data_review"]["columns"]] == ["item", "cost"]
    assert "SELECT * FROM" in data["data_review"]["sample_query"]


def test_graph_nodes_edges_and_unknown_file(served):
    base, _ = served
    g = _get_json(base, "/api/graph")
    assert len(g["nodes"]) == 3
    assert g["edges"], "the two garden files share a topic, so an edge must exist"
    assert all(e["kind"] in {"shared_topic", "shared_entity", "reference", "same_folder"}
               for e in g["edges"])
    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(base, "/api/file/does-not-exist")
    assert exc.value.code == 404
