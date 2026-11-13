import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from folderlake.extractors import extract


def test_txt(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("Hello FolderLake\nSome body text about bicycles.", encoding="utf-8")
    r = extract(p)
    assert r["kind"] == "text"
    assert "bicycles" in r["text"]


def test_html(tmp_path):
    p = tmp_path / "a.html"
    p.write_text("<html><head><title>My Page</title></head>"
                 "<body><p>Visible text here</p><script>var x=1;</script></body></html>",
                 encoding="utf-8")
    r = extract(p)
    assert r["kind"] == "document"
    assert r["title"] == "My Page"
    assert "Visible text" in r["text"]
    assert "var x" not in r["text"]


def test_docx(tmp_path):
    docx = __import__("docx")
    p = tmp_path / "a.docx"
    d = docx.Document()
    d.add_paragraph("Handbook body about workshop safety.")
    d.core_properties.author = "Alex Chen"
    d.save(str(p))
    r = extract(p)
    assert r["kind"] == "document"
    assert r["author"] == "Alex Chen"
    assert "workshop safety" in r["text"]


def test_pdf(tmp_path):
    reportlab = __import__("pytest").importorskip("reportlab")
    from reportlab.pdfgen import canvas

    p = tmp_path / "a.pdf"
    c = canvas.Canvas(str(p))
    c.drawString(72, 720, "Garden guide tomatoes watering")
    c.save()
    r = extract(p)
    assert r["kind"] == "document"
    assert r["extra"]["pages"] == 1
    assert "tomatoes" in r["text"]


def test_csv_profile(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("id,name,score\n1,Ann,10\n2,Bob,\n3,Cid,30\n", encoding="utf-8")
    r = extract(p)
    assert r["kind"] == "data"
    assert r["extra"]["row_count"] == 3
    assert [c["name"] for c in r["extra"]["columns"]] == ["id", "name", "score"]
    assert r["extra"]["nulls"]["score"] == 1


def test_json_data(tmp_path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps([{"sku": "A1", "price": 10}, {"sku": "B2", "price": 20}]),
                 encoding="utf-8")
    r = extract(p)
    assert r["kind"] == "data"
    assert r["extra"]["row_count"] == 2


def test_xlsx(tmp_path):
    openpyxl = __import__("openpyxl")
    p = tmp_path / "a.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["CustomerID", "total"])
    ws.append([1, 99])
    wb.save(str(p))
    r = extract(p)
    assert r["kind"] == "data"
    assert r["extra"]["row_count"] == 1
    assert "CustomerID" in r["text"]


def test_sqlite(tmp_path):
    p = tmp_path / "a.sqlite"
    db = sqlite3.connect(str(p))
    db.execute("CREATE TABLE t (sku TEXT, qty INTEGER)")
    db.execute("INSERT INTO t VALUES ('A1', 5)")
    db.commit()
    db.close()
    r = extract(p)
    assert r["kind"] == "data"
    assert r["extra"]["row_count"] == 1


def test_unknown_format_does_not_crash(tmp_path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"\x00\x01\x02")
    r = extract(p)
    assert r["kind"] == "other"
