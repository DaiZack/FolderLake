"""Generate the fictional demo folder: a bike shop + a community garden.

Mixed formats, two topic clusters, one shared person (Alex Chen) and one
explicit cross-reference, so topics/relations have something to find.
Run with the project venv python.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "demo_folder"


def main() -> None:
    if ROOT.exists():
        shutil.rmtree(ROOT)
    (ROOT / "notes").mkdir(parents=True)
    (ROOT / "web").mkdir()
    (ROOT / "docs").mkdir()
    (ROOT / "data").mkdir()

    (ROOT / "notes" / "meeting-notes.txt").write_text(
        "Spring tune-up campaign planning\n\n"
        "Alex Chen will lead the spring tune-up campaign at Riverside Bike Shop. "
        "The campaign offers brake checks, gear tuning and tire replacement "
        "packages through April. Workshop capacity is twelve bikes per day. "
        "Every tune-up package includes a safety inspection and chain cleaning.\n",
        encoding="utf-8",
    )
    (ROOT / "notes" / "garden-plan.md").write_text(
        "# Riverside Community Garden Plan\n\n"
        "The community garden opens in May with forty raised beds. "
        "Alex Chen coordinates the volunteer schedule for the garden. "
        "Planting starts with tomatoes, beans and herbs. The garden "
        "committee meets every Saturday morning near the Riverside beds.\n",
        encoding="utf-8",
    )
    (ROOT / "notes" / "references.txt").write_text(
        "Reference list for the spring review\n\n"
        "Sales figures are in orders.xlsx and the workshop rules are in "
        "coop-handbook.docx. The garden budget draft is garden-plan.md.\n",
        encoding="utf-8",
    )
    (ROOT / "web" / "spring-sale.html").write_text(
        "<html><head><title>Spring Tune-Up Sale — Riverside Bike Shop</title></head>"
        "<body><h1>Spring Tune-Up Sale</h1>"
        "<p>Save on brake service, gear tuning and tire packages this spring. "
        "Book a tune-up at Riverside Bike Shop before April ends.</p></body></html>\n",
        encoding="utf-8",
    )

    import docx

    doc = docx.Document()
    doc.add_heading("Riverside Bike Co-op Handbook", level=1)
    doc.add_paragraph(
        "Workshop rules for co-op members. Every repair stand must be cleaned "
        "after use. Tune-up work follows the safety inspection checklist. "
        "Members book repair stands through the front desk."
    )
    doc.core_properties.author = "Alex Chen"
    doc.core_properties.title = "Riverside Bike Co-op Handbook"
    doc.save(str(ROOT / "docs" / "coop-handbook.docx"))

    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    pdf_path = ROOT / "docs" / "garden-guide.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    c.setTitle("Community Garden Growing Guide")
    c.setAuthor("Garden Committee")
    c.drawString(72, 720, "Community Garden Growing Guide")
    c.drawString(72, 700, "Tomatoes need full sun and weekly watering in the garden beds.")
    c.drawString(72, 684, "Beans climb the trellis along the north fence of the garden.")
    c.drawString(72, 668, "Herbs grow best in the raised beds near the garden gate.")
    c.save()

    (ROOT / "data" / "customers.csv").write_text(
        "cust_id,full_name,city,signup_date\n"
        "1,Maya Patel,Toronto,2026-01-05\n"
        "2,Liam Ross,Hamilton,2026-01-11\n"
        "3,Sofia Nguyen,St. Catharines,2026-02-02\n"
        "4,Noah Brooks,Niagara Falls,2026-02-14\n",
        encoding="utf-8",
    )
    (ROOT / "data" / "products.json").write_text(
        json.dumps([
            {"sku": "BK-100", "product": "City Bike", "price": 549},
            {"sku": "AC-210", "product": "Helmet", "price": 79},
            {"sku": "AC-330", "product": "Bike Lock", "price": 45},
        ]),
        encoding="utf-8",
    )

    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Orders"
    ws.append(["CustomerID", "OrderDate", "total"])
    ws.append([1, "2026-03-01", 549])
    ws.append([2, "2026-03-03", 79])
    ws.append([1, "2026-03-05", 45])
    wb.save(str(ROOT / "data" / "orders.xlsx"))

    import duckdb

    con = duckdb.connect(":memory:")
    con.execute(
        "CREATE TABLE payments AS SELECT * FROM (VALUES "
        "(1, 'ORD-1', 549.0), (2, 'ORD-2', 79.0), (3, 'ORD-3', 45.0)"
        ") AS t(cust_no, order_ref, amount)"
    )
    con.execute(f"COPY payments TO '{ROOT / 'data' / 'payments.parquet'}' (FORMAT PARQUET)")
    con.close()

    db = sqlite3.connect(str(ROOT / "data" / "inventory.sqlite"))
    db.execute("CREATE TABLE inventory (sku TEXT, qty INTEGER)")
    db.executemany("INSERT INTO inventory VALUES (?,?)",
                   [("BK-100", 7), ("AC-210", 25), ("AC-330", 40)])
    db.commit()
    db.close()

    print(f"demo folder written: {ROOT}")
    for p in sorted(ROOT.rglob("*")):
        if p.is_file():
            print(" ", p.relative_to(ROOT))


if __name__ == "__main__":
    sys.exit(main())
