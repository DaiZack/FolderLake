"""Tests for the MCP tool handlers and registry (SDK not required)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from folderlake.catalog import scan
from folderlake.mcp_server import MAX_TEXT, TOOLS, serve_stdio
from folderlake.relations import build_relations

BIKES = (
    "Riverside Bike Workshop notes. The workshop repairs bicycle "
    "brakes, bicycle chains and bicycle gears. Every bicycle tune-up "
    "at the workshop includes a safety check. "
)
REPAIR = (
    "Bicycle repair log for the Riverside workshop. This bicycle "
    "needed a new chain and brake pads. Bicycle repair took one hour. "
)
GARDEN = (
    "Garden plan for spring. Plant tomatoes and cucumbers in the "
    "garden beds. Water the garden every morning and add compost. "
)


def _catalog(tmp_path: Path) -> Path:
    """Scan a tiny folder (+ build relations, as the CLI does)."""
    (tmp_path / "bikes.txt").write_text(BIKES * 12, encoding="utf-8")
    (tmp_path / "repair.txt").write_text(REPAIR * 6, encoding="utf-8")
    (tmp_path / "garden.md").write_text(GARDEN * 6, encoding="utf-8")
    (tmp_path / "sales.csv").write_text(
        "total_bill,tip_amount\n10,2\n20,4\n30,6\n", encoding="utf-8")
    db = tmp_path / "catalog.duckdb"
    scan(tmp_path, db)
    build_relations(db)
    return db


def _handler(name: str):
    return TOOLS[name][1]


def _file_id(db: Path, name: str) -> str:
    files = _handler("folderlake_list_files")(db)["files"]
    return next(f["id"] for f in files if f["name"] == name)


def test_tools_registry_complete():
    assert set(TOOLS) == {
        "folderlake_stats", "folderlake_search", "folderlake_list_files",
        "folderlake_file_detail", "folderlake_related", "folderlake_topics",
    }
    for description, handler in TOOLS.values():
        assert isinstance(description, str) and description
        assert callable(handler)


def test_stats_handler(tmp_path):
    db = _catalog(tmp_path)
    stats = _handler("folderlake_stats")(db)
    json.dumps(stats)
    assert stats["total_files"] == 4
    kinds = {k["kind"]: k["count"] for k in stats["kinds"]}
    assert kinds == {"text": 3, "data": 1}
    assert stats["topic_count"] >= 1


def test_search_handler(tmp_path):
    db = _catalog(tmp_path)
    hits = _handler("folderlake_search")(db, "bicycle")
    json.dumps(hits)
    names = {f["name"] for f in hits["files"]}
    assert "bikes.txt" in names and "repair.txt" in names
    assert hits["count"] == len(hits["files"])
    by_kind = _handler("folderlake_search")(db, "sales", kind="data")
    assert [f["name"] for f in by_kind["files"]] == ["sales.csv"]
    assert _handler("folderlake_search")(db, "bicycle", kind="data")["count"] == 0


def test_list_files_handler(tmp_path):
    db = _catalog(tmp_path)
    all_files = _handler("folderlake_list_files")(db)
    assert all_files["count"] == 4 and all_files["returned"] == 4
    assert all_files["truncated"] is False
    paged = _handler("folderlake_list_files")(db, limit=2)
    assert paged["returned"] == 2 and paged["truncated"] is True
    data_only = _handler("folderlake_list_files")(db, kind="data")
    assert [f["name"] for f in data_only["files"]] == ["sales.csv"]
    bikes = next(f for f in all_files["files"] if f["name"] == "bikes.txt")
    assert bikes["tags"]
    same_topic = _handler("folderlake_list_files")(db, topic=bikes["tags"][0])
    assert "bikes.txt" in {f["name"] for f in same_topic["files"]}


def test_file_detail_handler_truncates(tmp_path):
    db = _catalog(tmp_path)
    detail = _handler("folderlake_file_detail")(db, _file_id(db, "bikes.txt"))
    json.dumps(detail)
    assert detail["name"] == "bikes.txt"
    assert detail["tags"]
    assert isinstance(detail["entities"], list)
    assert detail["summary"]
    # Source text is ~2,800 chars: the preview must be capped.
    assert len(detail["text_preview"]) <= MAX_TEXT + 1
    assert detail["text_preview"].endswith("…")


def test_file_detail_data_review(tmp_path):
    db = _catalog(tmp_path)
    detail = _handler("folderlake_file_detail")(db, _file_id(db, "sales.csv"))
    json.dumps(detail)
    review = detail["data_review"]
    col_names = [
        c["name"] if isinstance(c, dict) else c for c in review["columns"]
    ]
    assert "total_bill" in col_names
    assert review["row_count"] == 3
    assert "SELECT" in review["sample_query"]


def test_file_detail_not_found(tmp_path):
    db = _catalog(tmp_path)
    assert "error" in _handler("folderlake_file_detail")(db, "no-such-id")


def test_related_handler(tmp_path):
    db = _catalog(tmp_path)
    related = _handler("folderlake_related")(db, _file_id(db, "bikes.txt"))
    json.dumps(related)
    assert related["count"] >= 1
    names = {r["name"] for r in related["related"]}
    assert "repair.txt" in names
    for r in related["related"]:
        assert {"file_id", "name", "kind", "relation_kind", "detail"} <= set(r)
    kinds = {r["relation_kind"] for r in related["related"]}
    assert kinds & {"shared_topic", "shared_entity", "reference", "same_folder"}


def test_topics_falls_back_to_flat(tmp_path):
    db = _catalog(tmp_path)
    topics = _handler("folderlake_topics")(db)
    json.dumps(topics)
    # A freshly scanned catalog has no hierarchy tables (build_hierarchy
    # was not run), so the handler must serve the flat facet topics.
    assert topics["mode"] == "flat"
    assert topics["topics"]
    assert {"tag", "count"} <= set(topics["topics"][0])


def test_serve_stdio_requires_sdk(tmp_path):
    if importlib.util.find_spec("mcp") is not None:
        pytest.skip("mcp SDK is installed in this environment")
    with pytest.raises(RuntimeError, match=r"folderlake\[mcp\]"):
        serve_stdio(tmp_path / "catalog.duckdb")
