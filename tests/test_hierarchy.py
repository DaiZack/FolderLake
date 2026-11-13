"""Tests for the hierarchical topic tree (hierarchy.py + /api/topics)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import duckdb

from folderlake.catalog import scan
from folderlake.hierarchy import build_hierarchy
from folderlake.server import list_files, topics_tree

BIKE = ("bicycle repair maintenance chain brake wheel pedal gear "
        "cycling ride route trail mountain road bike shop tune service ")
GARDEN = ("garden planting tomato cucumber watering soil compost "
          "spring seeds growing vegetables flowers beds harvest ")
FINANCE = ("budget savings finance investment portfolio stocks bonds "
           "retirement income expenses planning money market returns ")


def _write_docs(folder: Path) -> None:
    """16 documents: bike-repair x4, bike-route x4, garden x4, finance x4."""
    for i in range(4):
        (folder / f"bike-repair-{i}.txt").write_text(
            (BIKE + "brake pads chain lubricant adjustment fix ") * 6,
            encoding="utf-8")
        (folder / f"bike-route-{i}.txt").write_text(
            (BIKE + "trail map distance elevation scenic ride ") * 6,
            encoding="utf-8")
        (folder / f"garden-{i}.txt").write_text(GARDEN * 6, encoding="utf-8")
        (folder / f"finance-{i}.txt").write_text(FINANCE * 6, encoding="utf-8")


def _scan(folder: Path) -> Path:
    db = folder / "catalog.duckdb"
    scan(folder, db)
    build_hierarchy(db)
    return db


def _nodes(db: Path):
    con = duckdb.connect(str(db), read_only=True)
    rows = con.execute(
        "SELECT id, parent_id, kind, label, depth, file_count "
        "FROM topic_nodes").fetchall()
    con.close()
    return rows


def test_tree_groups_documents_by_topic(tmp_path):
    _write_docs(tmp_path)
    db = _scan(tmp_path)
    tree = topics_tree(db)
    assert len(tree["roots"]) == 1
    root = tree["roots"][0]
    assert root["kind"] == "text" and root["count"] == 16
    groups = root["children"]
    assert len(groups) >= 3  # bike / garden / finance at least
    # Each top-level group is majority one family (>= 2/3 pure).
    name_of = {}
    con = duckdb.connect(str(db), read_only=True)
    for fid, name in con.execute("SELECT id, name FROM files").fetchall():
        name_of[fid] = name
    members = {}
    for nid, fid in con.execute(
            "SELECT node_id, file_id FROM topic_node_files").fetchall():
        members.setdefault(nid, []).append(fid)
    con.close()
    for g in groups:
        fams = [name_of[f].split("-")[0] for f in members[g["id"]]]
        top = max(set(fams), key=fams.count)
        assert fams.count(top) >= len(fams) * 2 / 3, (g["label"], fams)


def test_tree_has_second_level_for_bike(tmp_path):
    _write_docs(tmp_path)
    db = _scan(tmp_path)
    rows = _nodes(db)
    assert any(r[4] >= 2 for r in rows), rows  # some depth-2 node exists


def test_kinds_never_mix(tmp_path):
    _write_docs(tmp_path)
    for i in range(5):
        (tmp_path / f"sales-{i}.csv").write_text(
            "order_id,amount,customer\n1,10,x\n2,20,y\n", encoding="utf-8")
    db = _scan(tmp_path)
    tree = topics_tree(db)
    kinds = {r["kind"] for r in tree["roots"]}
    assert kinds == {"text", "data"}
    rows = _nodes(db)
    by_id = {r[0]: r for r in rows}
    for r in rows:
        if r[1] is not None:
            assert by_id[r[1]][2] == r[2]  # parent kind == child kind


def test_node_filter_returns_subtree_only(tmp_path):
    _write_docs(tmp_path)
    db = _scan(tmp_path)
    tree = topics_tree(db)
    root = tree["roots"][0]
    assert len(list_files(db, node=root["id"])) == 16
    group = root["children"][0]
    got = list_files(db, node=group["id"])
    assert len(got) == group["count"] and 0 < len(got) < 16


def test_tiny_folder_gets_root_only(tmp_path):
    (tmp_path / "a.txt").write_text("hello world notes " * 5, encoding="utf-8")
    (tmp_path / "b.txt").write_text("another small note " * 5, encoding="utf-8")
    db = _scan(tmp_path)
    tree = topics_tree(db)
    assert len(tree["roots"]) == 1
    assert tree["roots"][0]["children"] == []
    assert tree["roots"][0]["count"] == 2
