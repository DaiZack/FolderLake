import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import duckdb

from folderlake.catalog import scan
from folderlake.preview import build_preview


def _tags(db, name):
    con = duckdb.connect(str(db), read_only=True)
    rows = con.execute(
        """
        SELECT t.tag, t.weight FROM file_topics t
        JOIN files f ON f.id = t.file_id WHERE f.name = ?
        ORDER BY t.weight DESC
        """,
        [name],
    ).fetchall()
    con.close()
    return rows


def test_tfidf_distinctive_term_beats_folder_common_term(tmp_path):
    body = "folderlake notes about shared paperwork and filing. "
    (tmp_path / "a.txt").write_text(body * 5, encoding="utf-8")
    (tmp_path / "b.txt").write_text(body * 5, encoding="utf-8")
    (tmp_path / "c.txt").write_text(
        body * 2 + "telescope telescope telescope astronomy telescope lens. " * 4,
        encoding="utf-8")
    db = tmp_path / "catalog.duckdb"
    scan(tmp_path, db)
    tags = _tags(db, "c.txt")
    names = [t for t, _ in tags]
    assert "telescope" in names
    if "folderlake" in names:
        weight = dict(tags)
        assert weight["telescope"] > weight["folderlake"]


def test_chinese_text_gets_tags(tmp_path):
    (tmp_path / "a.txt").write_text(
        "自行车维修保养记录：这次维修更换了刹车片，保养了链条。"
        "维修师傅说自行车保养很重要，定期维修可以延长寿命。", encoding="utf-8")
    (tmp_path / "b.txt").write_text(
        "花园种植计划：春天播种番茄和黄瓜，需要定期浇水施肥，"
        "花园里的蔬菜长势不错，种植让人心情愉快。", encoding="utf-8")
    db = tmp_path / "catalog.duckdb"
    scan(tmp_path, db)
    tags = [t for t, _ in _tags(db, "a.txt")]
    assert any(any("\u3400" <= ch <= "\u9fff" for ch in t) for t in tags)


def test_data_tags_come_from_schema_not_sample_values(tmp_path):
    (tmp_path / "sales.csv").write_text(
        "sku_code,price_amount,customer_city\n"
        "banana,10,banana\nbanana,20,banana\nbanana,30,banana\n",
        encoding="utf-8")
    db = tmp_path / "catalog.duckdb"
    scan(tmp_path, db)
    tags = [t for t, _ in _tags(db, "sales.csv")]
    assert "banana" not in tags
    # v3: schema concepts surface as column-name phrases, not single tokens
    assert {"sku code", "price amount", "customer city"} & set(tags)


def test_data_summarize_reaches_preview(tmp_path):
    (tmp_path / "t.csv").write_text(
        "id,score\n1,10\n2,20\n3,30\n", encoding="utf-8")
    db = tmp_path / "catalog.duckdb"
    scan(tmp_path, db)
    con = duckdb.connect(str(db), read_only=True)
    fid = con.execute("SELECT id FROM files WHERE name = 't.csv'").fetchone()[0]
    con.close()
    prev = build_preview(db, fid)
    summ = {s["name"]: s for s in prev["data_review"]["summarize"]}
    assert summ["score"]["approx_unique"] == 3
    assert summ["score"]["min"] == "10"
    assert summ["score"]["max"] == "30"


def test_incremental_add_file_reuses_existing_topics(tmp_path):
    (tmp_path / "a.txt").write_text(
        "Bicycle workshop repair notes and brake inspection guide.", encoding="utf-8")
    (tmp_path / "b.txt").write_text(
        "Garden planting watering tomatoes herbs soil sunlight.", encoding="utf-8")
    db = tmp_path / "catalog.duckdb"
    s1 = scan(tmp_path, db)
    assert s1["added"] == 2
    tags_before = _tags(db, "a.txt")
    assert tags_before

    (tmp_path / "c.txt").write_text(
        "Cooking recipe oven seasoning dinner kitchen ingredients.", encoding="utf-8")
    s2 = scan(tmp_path, db)
    assert s2["added"] == 1 and s2["skipped"] == 2 and s2["updated"] == 0
    # old files keep topics; the new file gets its own
    assert _tags(db, "a.txt")
    assert _tags(db, "c.txt")
    con = duckdb.connect(str(db), read_only=True)
    n_terms = con.execute("SELECT count(*) FROM file_terms").fetchone()[0]
    con.close()
    assert n_terms > 0
