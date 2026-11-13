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


def test_subject_reference_text_to_data(tmp_path):
    (tmp_path / "titanic.csv").write_text(
        "name,survived\nJack Dawson,0\nRose DeWitt,1\n", encoding="utf-8")
    (tmp_path / "tips.csv").write_text(
        "total_bill,tip\n20.0,3.0\n", encoding="utf-8")
    (tmp_path / "note.txt").write_text(
        "The Titanic sank in 1912 on its maiden voyage across the "
        "Atlantic. These are my Tips for maintaining a bicycle chain.",
        encoding="utf-8")
    (tmp_path / "lower.txt").write_text(
        "the titanic effort of moving house took all weekend",
        encoding="utf-8")
    db = tmp_path / "catalog.duckdb"
    scan(tmp_path, db)
    build_relations(db)
    con = duckdb.connect(str(db), read_only=True)
    refs = con.execute(
        "SELECT detail FROM relations WHERE kind='reference'").fetchall()
    con.close()
    details = {r[0] for r in refs}
    # capitalised subject mention links the note to the dataset
    assert "mentions titanic" in details
    # common-word dataset names never fire, even capitalised
    assert "mentions tips" not in details
    # lowercase-only mentions do not count (one edge, not two)
    assert len([d for d in refs if d[0] == "mentions titanic"]) == 1
