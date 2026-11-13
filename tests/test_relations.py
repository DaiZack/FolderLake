import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import duckdb

from folderlake.catalog import scan
from folderlake.relations import build_relations


def test_relations_kinds(tmp_path):
    (tmp_path / "one.txt").write_text(
        "Alex Chen leads the Riverside workshop. Alex Chen runs the "
        "workshop schedule every week. See also two.txt for details.",
        encoding="utf-8")
    (tmp_path / "two.txt").write_text(
        "The Riverside workshop schedule by Alex Chen. Alex Chen posted "
        "the workshop hours for members.",
        encoding="utf-8")
    db = tmp_path / "catalog.duckdb"
    scan(tmp_path, db)
    n = build_relations(db)
    assert n > 0
    con = duckdb.connect(str(db), read_only=True)
    kinds = {r[0] for r in con.execute("SELECT DISTINCT kind FROM relations").fetchall()}
    con.close()
    assert "same_folder" in kinds
    assert "shared_entity" in kinds or "shared_topic" in kinds
    assert "reference" in kinds
