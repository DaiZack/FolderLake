"""One small extractor per file format.

Every extractor returns the same shape:
    {"kind": str, "title": str|None, "author": str|None,
     "extra": dict, "text": str}

kind is one of: document, text, data, other.
Adding a format means adding one function here and one registry entry —
nothing else in the system changes.
"""

from __future__ import annotations

import json
import sqlite3
from html.parser import HTMLParser
from pathlib import Path

MAX_TEXT_CHARS = 200_000


def _clean(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)[:MAX_TEXT_CHARS]


# ---------------------------------------------------------------- text files
def extract_plain(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8", errors="replace")
    title = None
    for ln in raw.splitlines():
        if ln.strip():
            title = ln.strip().lstrip("# ").strip()[:200]
            break
    return {"kind": "text", "title": title, "author": None, "extra": {}, "text": _clean(raw)}


# ---------------------------------------------------------------- html
class _HTMLText(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title: str | None = None
        self._in_title = False
        self._skip = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self._in_title = True
        if tag in ("script", "style"):
            self._skip += 1

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if tag in ("script", "style") and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if self._skip:
            return
        if self._in_title:
            self.title = (self.title or "") + data.strip()
        text = data.strip()
        if text:
            self.parts.append(text)


def extract_html(path: Path) -> dict:
    parser = _HTMLText()
    parser.feed(path.read_text(encoding="utf-8", errors="replace"))
    return {
        "kind": "document",
        "title": parser.title,
        "author": None,
        "extra": {},
        "text": _clean("\n".join(parser.parts)),
    }


# ---------------------------------------------------------------- docx
def extract_docx(path: Path) -> dict:
    import docx  # python-docx

    doc = docx.Document(str(path))
    text = "\n".join(p.text for p in doc.paragraphs)
    props = doc.core_properties
    return {
        "kind": "document",
        "title": props.title or None,
        "author": props.author or None,
        "extra": {"paragraphs": len(doc.paragraphs)},
        "text": _clean(text),
    }


# ---------------------------------------------------------------- pdf
def extract_pdf(path: Path) -> dict:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    text = "\n".join((page.extract_text() or "") for page in reader.pages)
    meta = reader.metadata or {}
    return {
        "kind": "document",
        "title": (getattr(meta, "title", None) or None),
        "author": (getattr(meta, "author", None) or None),
        "extra": {"pages": len(reader.pages)},
        "text": _clean(text),
    }


# ---------------------------------------------------------------- data files
_SHARED_CON = None


def _shared_con():
    global _SHARED_CON
    if _SHARED_CON is None:
        import duckdb

        _SHARED_CON = duckdb.connect(":memory:")
    return _SHARED_CON


def _duckdb_profile(path: Path, reader_sql: str) -> dict:
    """Schema + row count + per-column nulls, computed by DuckDB itself.

    Uses one shared in-memory connection across files — opening a fresh
    DuckDB connection per file costs ~200ms and would dominate scan time.
    """
    con = _shared_con()
    try:
        cols = con.execute(
            f"DESCRIBE SELECT * FROM {reader_sql}"
        ).fetchall()  # (name, type, ...)
        columns = [{"name": c[0], "type": c[1]} for c in cols]
        row_count = con.execute(
            f"SELECT count(*) FROM {reader_sql}"
        ).fetchone()[0]
        nulls = {}
        if columns and row_count:
            exprs = ", ".join(
                f'count(*) - count("{c["name"]}")' for c in columns
            )
            row = con.execute(
                f"SELECT {exprs} FROM {reader_sql}"
            ).fetchone()
            nulls = {c["name"]: row[i] for i, c in enumerate(columns)}
        sample_rows = con.execute(
            f"SELECT * FROM {reader_sql} LIMIT 5"
        ).fetchall()
    except Exception:
        raise
    # SUMMARIZE: per-column statistics (min/max, approx_unique, avg,
    # null %) — the profile data-file topics are built from, so sample
    # values never have to leak into topic computation.
    summarize = []
    try:
        for r in con.execute(f"SUMMARIZE SELECT * FROM {reader_sql}").fetchall():
            summarize.append({
                "name": r[0], "type": r[1],
                "min": None if r[2] is None else str(r[2]),
                "max": None if r[3] is None else str(r[3]),
                "approx_unique": r[4],
                "avg": None if r[5] is None else str(r[5]),
                "null_percentage": None if r[11] is None else str(r[11]),
            })
    except Exception:
        summarize = []
    text_parts = [
        f"columns: {', '.join(c['name'] for c in columns)}",
        f"rows: {row_count}",
    ]
    for r in sample_rows:
        text_parts.append(" | ".join("" if v is None else str(v) for v in r))
    return {
        "kind": "data",
        "title": None,
        "author": None,
        "extra": {
            "columns": columns,
            "row_count": row_count,
            "nulls": nulls,
            "summarize": summarize,
        },
        "text": _clean("\n".join(text_parts)),
    }


def _quoted(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def extract_csv(path: Path) -> dict:
    delim = "\t" if path.suffix.lower() == ".tsv" else ","
    return _duckdb_profile(path, f"read_csv({_quoted(path)}, header=true, delim='{delim}')")


def extract_json_data(path: Path) -> dict:
    try:
        return _duckdb_profile(path, f"read_json_auto({_quoted(path)})")
    except Exception:
        # Not tabular JSON — treat as text.
        result = extract_plain(path)
        result["kind"] = "text"
        return result


def extract_parquet(path: Path) -> dict:
    return _duckdb_profile(path, f"read_parquet({_quoted(path)})")


def extract_xlsx(path: Path) -> dict:
    import openpyxl

    wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    sheets = {}
    text_parts = []
    total_rows = 0
    summarize = []
    for ws in wb.worksheets:
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            sheets[ws.title] = {"rows": 0, "columns": []}
            continue
        header = ["" if h is None else str(h) for h in rows[0]]
        sheets[ws.title] = {"rows": len(rows) - 1, "columns": header}
        total_rows += len(rows) - 1
        text_parts.append(f"sheet: {ws.title}; columns: {', '.join(header)}")
        for r in rows[1:6]:
            text_parts.append(" | ".join("" if v is None else str(v) for v in r))
        if not summarize and rows[1:]:
            # SUMMARIZE-equivalent for the first sheet, computed from the
            # rows already in memory: distinct count (capped set), min/max
            # over comparable values, null percentage.
            body = rows[1:]
            for ci, cname in enumerate(header):
                vals = [r[ci] for r in body if ci < len(r) and r[ci] is not None]
                distinct = len({str(v) for v in vals[:5000]})
                nums = [v for v in vals if isinstance(v, (int, float))]
                pool = nums if nums else [str(v) for v in vals]
                summarize.append({
                    "name": cname, "type": "unknown",
                    "min": None if not pool else str(min(pool)),
                    "max": None if not pool else str(max(pool)),
                    "approx_unique": distinct,
                    "avg": (str(round(sum(nums) / len(nums), 4)) if nums else None),
                    "null_percentage": str(round(
                        100.0 * (len(body) - len(vals)) / len(body), 2)),
                })
    wb.close()
    first = next(iter(sheets.values()), {"columns": []})
    return {
        "kind": "data",
        "title": None,
        "author": None,
        "extra": {
            "sheets": sheets,
            "row_count": total_rows,
            "columns": [{"name": c, "type": "unknown"} for c in first["columns"]],
            "nulls": {},
            "summarize": summarize,
        },
        "text": _clean("\n".join(text_parts)),
    }


def extract_sqlite(path: Path) -> dict:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        tables = [
            r[0]
            for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        ]
        text_parts = []
        columns_all = []
        total = 0
        summarize = []
        for t in tables:
            cols = [r[1] for r in con.execute(f'PRAGMA table_info("{t}")')]
            n = con.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0]
            total += n
            columns_all += cols
            text_parts.append(f"table: {t}; columns: {', '.join(cols)}; rows: {n}")
            for c in cols:
                try:
                    cnt, uniq, mn, mx, avg = con.execute(
                        f'SELECT count("{c}"), count(DISTINCT "{c}"), min("{c}"), '
                        f'max("{c}"), avg("{c}") FROM "{t}"'
                    ).fetchone()
                except Exception:
                    cnt, uniq, mn, mx, avg = con.execute(
                        f'SELECT count("{c}"), count(DISTINCT "{c}"), min("{c}"), '
                        f'max("{c}") , NULL FROM "{t}"'
                    ).fetchone()
                summarize.append({
                    "name": c, "type": "unknown", "table": t,
                    "min": None if mn is None else str(mn),
                    "max": None if mx is None else str(mx),
                    "approx_unique": uniq,
                    "avg": None if avg is None else str(avg),
                    "null_percentage": (
                        str(round(100.0 * (n - cnt) / n, 2)) if n else "0"
                    ),
                })
    finally:
        con.close()
    return {
        "kind": "data",
        "title": None,
        "author": None,
        "extra": {
            "tables": tables,
            "row_count": total,
            "columns": [{"name": c, "type": "unknown"} for c in columns_all],
            "nulls": {},
            "summarize": summarize,
        },
        "text": _clean("\n".join(text_parts)),
    }


def extract_other(path: Path) -> dict:
    return {"kind": "other", "title": None, "author": None, "extra": {}, "text": ""}


EXTRACTORS = {
    ".txt": extract_plain,
    ".md": extract_plain,
    ".html": extract_html,
    ".htm": extract_html,
    ".docx": extract_docx,
    ".pdf": extract_pdf,
    ".csv": extract_csv,
    ".tsv": extract_csv,
    ".json": extract_json_data,
    ".parquet": extract_parquet,
    ".xlsx": extract_xlsx,
    ".sqlite": extract_sqlite,
    ".db": extract_sqlite,
}

SUPPORTED_EXTS = set(EXTRACTORS)


def extract(path: Path) -> dict:
    fn = EXTRACTORS.get(path.suffix.lower(), extract_other)
    try:
        return fn(path)
    except Exception as exc:  # a broken file must never kill a scan
        return {
            "kind": "other",
            "title": None,
            "author": None,
            "extra": {"extract_error": str(exc)[:300]},
            "text": "",
        }
