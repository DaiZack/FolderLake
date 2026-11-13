import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from folderlake.catalog import scan


def _mk(folder: Path, name: str, text: str) -> Path:
    p = folder / name
    p.write_text(text, encoding="utf-8")
    return p


def test_scan_and_incremental_skip(tmp_path):
    _mk(tmp_path, "a.txt", "Bicycles and tune-up notes for the workshop.")
    _mk(tmp_path, "b.txt", "Garden beds, tomatoes and watering schedule.")
    _mk(tmp_path, "c.md", "# Garden\nMore garden planting notes and herbs.")
    db = tmp_path / "catalog.duckdb"

    s1 = scan(tmp_path, db)
    assert s1["added"] == 3 and s1["skipped"] == 0

    s2 = scan(tmp_path, db)
    assert s2["skipped"] == 3 and s2["added"] == 0 and s2["updated"] == 0


def test_scan_update_and_remove(tmp_path):
    a = _mk(tmp_path, "a.txt", "Bicycles and tune-up notes.")
    b = _mk(tmp_path, "b.txt", "Garden beds and tomatoes.")
    db = tmp_path / "catalog.duckdb"
    scan(tmp_path, db)

    time.sleep(0.02)
    a.write_text("Bicycles and tune-up notes, revised with much more detail added.",
                 encoding="utf-8")
    b.unlink()

    s = scan(tmp_path, db)
    assert s["updated"] == 1
    assert s["removed"] == 1
