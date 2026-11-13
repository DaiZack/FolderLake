"""Topic extraction v2: corpus TF-IDF + hotspot segments + optional LLM.

Design (agreed 2026-11-13):
- Topics are a *post-scan* step, not per-file during extraction, because
  TF-IDF needs corpus statistics (document frequency across the folder).
- Per-file term counts are stored in `file_terms` at scan time, so adding
  a file never re-reads or re-extracts unchanged files: DF is recomputed
  from the stored counts (a DuckDB GROUP BY), rule tags are recomputed
  from those counts in memory, and the LLM is called only for
  new/changed files (tracked in `file_topic_meta` by content hash).
- Incremental cost rule: an unchanged file only gets its tag weights
  re-scored from stored counts — its summary and entities (pure content
  derivations) are kept, and its text is never re-tokenized. Only
  new/changed files pay the full segment/entity/LLM treatment.
- Documents: tags come from TF-IDF (distinctive terms) with title and
  filename terms boosted (folder-wide theme words would otherwise be
  suppressed by IDF alone). The LLM package, when --llm is on, is the
  filename/title + top TF terms + top TF-IDF terms + hotspot segments
  (best-scoring paragraph from each third of the document), capped at
  ~10k chars instead of a blind head slice.
- Data files take a separate path: their terms come only from schema
  identifiers (file/table/sheet/column names, split on snake_case and
  camelCase) and their summary/LLM input is the DESCRIBE + SUMMARIZE
  profile card. Sample-row values never enter topic computation.
- Tokenizer: English words plus CJK character bigrams (zero-dependency
  segmentation that works for Chinese text, which the v1 English-only
  regex could not tag at all).
- v3 phrases: tags should read as concepts, not single words, so the
  tokenizer also emits English word bigrams ("personal finance") and
  CJK trigrams ("自行车"), and data files emit joined column-name phrases
  ("total bill"). At selection time phrases get a length bonus and
  suppress their own components (no "finance" next to "personal
  finance", no "自行" next to "自行车"). Catalogs built before v3 are
  migrated in compute_topics by recomputing term counts from the stored
  text — no file re-extraction (tracked by a terms_version marker).
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

from . import llm

STOPWORDS = set(
    """the and for are but not you all any can had her was one our out has have been
    this that with they from were will would there their what when make like just over
    such take year them well back than been good much some time very know see your come
    could now than first into its who after how said each about many then she may these
    file files data table columns rows column row name value values using use used
    more quot amp http https www com org net html htm
    xlsx parquet sqlite json csv tsv sheet sheets format formats""".split()
)

EN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]*")
CJK_RE = re.compile(r"[\u3400-\u9fff]+")
CJK_STOP_CHARS = set("的了是在和与及对为以之也都而被把让从向于中上下不无有无其此该等")
ENTITY_RE = re.compile(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})\b")
TOPIC_PACKAGE_CHARS = 10_000


# --------------------------------------------------------------------------
# Tokenizing
# --------------------------------------------------------------------------
def _split_identifier(token: str) -> list[str]:
    parts: list[str] = []
    for chunk in re.split(r"[_\-]+", token):
        parts.extend(
            p.lower()
            for p in re.findall(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+", chunk)
        )
    return parts


PHRASE_STOP = set(
    """the and for are with from this that was were have has been will would
    their there its it in on of to a an is as at by or not but you all any
    can had her one our out more http https www quot amp""".split()
)


def tokenize(text: str) -> list[str]:
    """English words + word bigrams, CJK bigrams + trigrams, filtered."""
    terms: list[str] = []
    seq: list[str] = []  # phrase-eligible word sequence (keeps short words)
    for raw in EN_RE.findall(text or ""):
        for part in _split_identifier(raw):
            if part.isdigit():
                continue
            if len(part) >= 3 and part not in STOPWORDS:
                terms.append(part)
            if len(part) >= 2 and part not in PHRASE_STOP:
                seq.append(part)
    for a, b in zip(seq, seq[1:]):
        if a != b:
            terms.append(f"{a} {b}")
    for run in CJK_RE.findall(text or ""):
        if len(run) == 1:
            continue
        for i in range(len(run) - 1):
            bg = run[i : i + 2]
            if bg[0] in CJK_STOP_CHARS and bg[1] in CJK_STOP_CHARS:
                continue
            terms.append(bg)
        for i in range(len(run) - 2):
            tg = run[i : i + 3]
            if tg[0] in CJK_STOP_CHARS and tg[2] in CJK_STOP_CHARS:
                continue
            terms.append(tg)
    return terms


def identifier_tokens(name: str) -> list[str]:
    return [
        p
        for p in _split_identifier(name or "")
        if len(p) >= 3 and p not in STOPWORDS and not p.isdigit()
    ]


def term_counts_for(result: dict, filename: str) -> Counter:
    """Term counts stored per file at scan time (the incremental DF base).

    Data files count schema identifiers only — never sample-row values.
    Everything else counts tokens over the extracted text.
    """
    if result.get("kind") == "data":
        extra = result.get("extra") or {}
        counts: Counter = Counter()
        for tok in identifier_tokens(Path(filename).stem):
            counts[tok] += 2
        names: list[str] = []
        for col in extra.get("columns", []):
            names.append(col.get("name", "") if isinstance(col, dict) else str(col))
        names.extend(extra.get("tables", []))
        names.extend((extra.get("sheets") or {}).keys())
        for n in names:
            parts = identifier_tokens(n)
            for tok in parts:
                counts[tok] += 1
            if len(parts) >= 2:
                # A column name is a deliberate label, not prose: its
                # phrase form gets the same ×2 weight as a filename.
                counts[" ".join(parts)] += 2
        stem_parts = identifier_tokens(Path(filename).stem)
        if len(stem_parts) >= 2:
            counts[" ".join(stem_parts)] += 2
        return counts
    return Counter(tokenize(result.get("text") or ""))


def extract_entities(text: str) -> list[str]:
    freq = Counter(m.group(1) for m in ENTITY_RE.finditer(text or ""))
    out = []
    for ent, cnt in freq.most_common(8):
        if cnt >= 2 and ent.lower() not in STOPWORDS:
            out.append(ent)
    return out[:5]


# --------------------------------------------------------------------------
# Hotspot segments
# --------------------------------------------------------------------------
def _paragraphs(text: str) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]
    if len(paras) >= 3:
        return paras
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    chunks, buf = [], ""
    for ln in lines:
        buf = (buf + "\n" + ln) if buf else ln
        if len(buf) >= 600:
            chunks.append(buf)
            buf = ""
    if buf:
        chunks.append(buf)
    return chunks


def hotspot_segments(text: str, scores: dict[str, float], per_segment: int = 2200) -> list[str]:
    """Best TF-IDF paragraph from each third of the document, in order."""
    paras = _paragraphs(text)
    if not paras:
        return []

    def score(p: str) -> float:
        toks = tokenize(p)
        if not toks:
            return 0.0
        return sum(scores.get(t, 0.0) for t in toks) / math.sqrt(len(toks))

    picked: set[int] = {0}
    n = len(paras)
    for third in range(3):
        lo, hi = third * n // 3, max((third + 1) * n // 3, third * n // 3 + 1)
        best = max(range(lo, min(hi, n)), key=lambda i: score(paras[i]))
        picked.add(best)
    picked.add(n - 1)
    return [paras[i][:per_segment] for i in sorted(picked)]


# --------------------------------------------------------------------------
# LLM tier
# --------------------------------------------------------------------------
def _llm_call(prompt: str) -> dict | None:
    """One LLM round-trip via the provider layer (see llm.py)."""
    parsed = llm.chat_json(prompt)
    if parsed is None:
        return None
    return {
        "tags": [{"tag": str(t), "weight": 1.0} for t in parsed.get("tags", [])[:5]],
        "summary": str(parsed.get("summary", ""))[:200],
        "entities": [str(e) for e in parsed.get("entities", [])[:5]],
    }


def _llm_package_doc(name, title, tf_top, tfidf_top, segments) -> str:
    parts = [
        f"Filename: {name}",
        f"Title: {title or ''}",
        "Frequent terms (theme): " + ", ".join(tf_top),
        "Distinctive TF-IDF terms: " + ", ".join(tfidf_top),
        "Key segments:",
        "\n---\n".join(segments),
    ]
    return "\n".join(parts)[:TOPIC_PACKAGE_CHARS]


def _llm_package_data(name, profile_text) -> str:
    return f"Filename: {name}\nTable profile (DESCRIBE + SUMMARIZE):\n{profile_text}"[
        :TOPIC_PACKAGE_CHARS
    ]


def profile_card_text(extra: dict, row_count) -> str:
    """Human/LLM-readable DESCRIBE + SUMMARIZE card for a data file."""
    lines = [f"rows: {row_count}"]
    summ = {s.get("name"): s for s in (extra.get("summarize") or [])}
    for col in extra.get("columns", []):
        cname = col.get("name") if isinstance(col, dict) else str(col)
        ctype = col.get("type", "") if isinstance(col, dict) else ""
        s = summ.get(cname) or {}
        bits = [f"column {cname} ({ctype})"]
        if s:
            bits.append(f"unique≈{s.get('approx_unique')}")
            bits.append(f"min={s.get('min')} max={s.get('max')}")
            bits.append(f"null%={s.get('null_percentage')}")
            if s.get("avg") not in (None, "None"):
                bits.append(f"avg={s.get('avg')}")
        lines.append("; ".join(bits))
    if extra.get("tables"):
        lines.append("tables: " + ", ".join(extra["tables"]))
    if extra.get("sheets"):
        lines.append("sheets: " + ", ".join(extra["sheets"].keys()))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Post-scan computation (batched: bulk reads, in-memory scoring, bulk writes)
# --------------------------------------------------------------------------
TERMS_VERSION = 4


def _term_bonus(term: str) -> float:
    """Phrases outrank their own components at equal evidence."""
    if " " in term:
        return 1.5
    if len(term) == 3 and CJK_RE.fullmatch(term):
        return 1.3
    return 1.0


def _is_phrase(term: str) -> bool:
    return " " in term or (len(term) == 3 and CJK_RE.fullmatch(term) is not None)


def _overlaps(a: str, b: str) -> bool:
    """True when one term is a component/substring of the other."""
    if " " in a or " " in b:
        pa, pb = set(a.split(" ")), set(b.split(" "))
        return (len(pa) == 1 and pa <= pb) or (len(pb) == 1 and pb <= pa)
    if CJK_RE.fullmatch(a) and CJK_RE.fullmatch(b):
        return a != b and (a in b or b in a)
    return False


def _select_tags(ranked, counts, limit: int = 5):
    """Pick top tags: phrases first-class, components suppressed.

    A phrase needs at least 2 occurrences in the file to be trusted;
    if that starves the list, a second pass relaxes the count rule.
    """
    picked: list = []
    for term, score in ranked:
        if len(picked) >= limit:
            break
        if _is_phrase(term) and counts.get(term, 0) < 2:
            continue
        if any(_overlaps(term, p) for p, _ in picked):
            continue
        picked.append((term, score))
    if len(picked) < limit:
        for term, score in ranked:
            if len(picked) >= limit:
                break
            if any(term == p or _overlaps(term, p) for p, _ in picked):
                continue
            picked.append((term, score))
    return picked


def _chunks(seq, size=500):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _delete_for(con, table: str, fids: list[str]) -> None:
    for part in _chunks(fids):
        marks = ",".join("?" * len(part))
        con.execute(f"DELETE FROM {table} WHERE file_id IN ({marks})", part)


def compute_topics(db_path: Path | None = None, use_llm: bool = False,
                   llm_file_ids: set[str] | None = None, con=None) -> dict:
    """Recompute topics from stored term counts; no file I/O.

    Unchanged files (content hash matches file_topic_meta) keep their
    summary/entities and only get tag weights re-scored in memory.
    New/changed files get the full treatment (hotspot segments,
    entities, and — when enabled — an LLM call). LLM results are cached
    by hash and never re-requested for unchanged files.

    Pass an open `con` to reuse the caller's connection (scan does —
    reopening a DuckDB file just to rescore tags costs a checkpoint
    on close); otherwise a connection to db_path is opened and closed.
    """
    import duckdb

    own_con = con is None
    if own_con:
        con = duckdb.connect(str(db_path))
    try:
        con.execute(
            "CREATE TABLE IF NOT EXISTS file_terms "
            "(file_id VARCHAR, term VARCHAR, cnt BIGINT)"
        )
        con.execute(
            "CREATE TABLE IF NOT EXISTS file_topic_meta "
            "(file_id VARCHAR PRIMARY KEY, hash VARCHAR, method VARCHAR)"
        )
        try:
            con.execute("ALTER TABLE file_topics ADD COLUMN IF NOT EXISTS source VARCHAR")
        except Exception:
            pass

        files = con.execute(
            "SELECT id, name, kind, title, hash, extra_json FROM files"
        ).fetchall()
        if not files:
            return {"files": 0, "llm": 0, "rule": 0}
        file_ids = [f[0] for f in files]

        terms_by_file: dict[str, Counter] = {fid: Counter() for fid in file_ids}
        for fid, term, cnt in con.execute("SELECT file_id, term, cnt FROM file_terms").fetchall():
            if fid in terms_by_file:
                terms_by_file[fid][term] = cnt

        # Terms-format migration: catalogs built before v3 have no
        # phrase terms. Recompute counts from the stored text/extras —
        # no file re-extraction — and rewrite file_terms wholesale.
        con.execute(
            "CREATE TABLE IF NOT EXISTS catalog_meta "
            "(key VARCHAR PRIMARY KEY, value VARCHAR)"
        )
        vrow = con.execute(
            "SELECT value FROM catalog_meta WHERE key = 'terms_version'"
        ).fetchone()
        if (int(vrow[0]) if vrow else 0) < TERMS_VERSION:
            texts_all = {
                r[0]: r[1]
                for r in con.execute("SELECT file_id, text FROM file_text").fetchall()
            }
            all_rows = []
            for fid, name, kind, _t, _h, extra_json in files:
                result = {
                    "kind": kind,
                    "extra": json.loads(extra_json or "{}"),
                    "text": texts_all.get(fid, ""),
                }
                terms_by_file[fid] = term_counts_for(result, name)
                all_rows.extend(
                    (fid, term, cnt) for term, cnt in terms_by_file[fid].items()
                )
            # Bulk-load via a temp CSV inside one transaction:
            # executemany in autocommit mode costs a disk sync per row
            # and turns this migration into minutes on large folders.
            con.execute("BEGIN TRANSACTION")
            con.execute("DELETE FROM file_terms")
            if all_rows:
                import csv
                import tempfile

                with tempfile.NamedTemporaryFile(
                    "w", suffix=".csv", newline="", delete=False
                ) as fh:
                    csv.writer(fh).writerows(all_rows)
                    tmp_csv = fh.name
                con.execute(
                    "INSERT INTO file_terms SELECT * FROM read_csv(?, header=false, "
                    "columns={'file_id': 'VARCHAR', 'term': 'VARCHAR', 'cnt': 'BIGINT'})",
                    [tmp_csv],
                )
                Path(tmp_csv).unlink(missing_ok=True)
            con.execute(
                "INSERT OR REPLACE INTO catalog_meta VALUES ('terms_version', ?)",
                [str(TERMS_VERSION)],
            )
            con.execute("COMMIT")

        meta = {
            r[0]: (r[1], r[2])
            for r in con.execute("SELECT file_id, hash, method FROM file_topic_meta").fetchall()
        }
        existing_topics: dict[str, list[str]] = {fid: [] for fid in file_ids}
        for fid, tag in con.execute(
            "SELECT file_id, tag FROM file_topics ORDER BY file_id, weight DESC"
        ).fetchall():
            if fid in existing_topics:
                existing_topics[fid].append(tag)

        # Backfill term counts for catalogs built before file_terms
        # existed (one bulk text read, only for files that lack terms).
        missing = [f for f in files if not terms_by_file[f[0]]]
        if missing:
            texts = {
                r[0]: r[1]
                for r in con.execute("SELECT file_id, text FROM file_text").fetchall()
            }
            backfill_rows = []
            for fid, name, kind, _t, _h, extra_json in missing:
                result = {
                    "kind": kind,
                    "extra": json.loads(extra_json or "{}"),
                    "text": texts.get(fid, ""),
                }
                terms_by_file[fid] = term_counts_for(result, name)
                backfill_rows.extend(
                    (fid, term, cnt) for term, cnt in terms_by_file[fid].items()
                )
            if backfill_rows:
                con.executemany("INSERT INTO file_terms VALUES (?,?,?)", backfill_rows)

        df: Counter = Counter()
        for counts in terms_by_file.values():
            for term in counts:
                df[term] += 1
        n_docs = len(files)
        idf = {t: math.log((n_docs + 1) / (d + 1)) + 1.0 for t, d in df.items()}

        # Classify: llm-cached (skip), fresh (full treatment), or
        # unchanged (tag re-score only — summary/entities kept).
        fresh_ids = []
        for fid, _n, _k, _t, fhash, _e in files:
            prev = meta.get(fid)
            if prev and prev[0] == fhash:
                continue
            fresh_ids.append(fid)
        fresh_set = set(fresh_ids)
        fresh_texts: dict[str, str] = {}
        if fresh_ids:
            wanted = set(fresh_ids)
            fresh_texts = {
                r[0]: r[1]
                for r in con.execute("SELECT file_id, text FROM file_text").fetchall()
                if r[0] in wanted
            }

        topic_rows, entity_rows, summary_rows, meta_rows = [], [], [], []
        stats = {"files": 0, "llm": 0, "rule": 0}
        recomputed_ids: list[str] = []

        for fid, name, kind, title, fhash, extra_json in files:
            prev = meta.get(fid)
            if prev and prev[1] == "llm" and prev[0] == fhash:
                stats["files"] += 1
                stats["llm"] += 1
                continue
            counts = terms_by_file[fid]
            total = sum(counts.values()) or 1
            scores = {t: (c / total) * idf.get(t, 1.0) for t, c in counts.items()}
            if kind != "data":
                for t in set(tokenize(f"{title or ''} {Path(name).stem}")):
                    if t in scores:
                        scores[t] *= 3.0
            ranked = sorted(
                ((t, s * _term_bonus(t)) for t, s in scores.items()),
                key=lambda kv: -kv[1],
            )
            top = _select_tags(ranked, counts)
            max_score = top[0][1] if top else 1.0
            tags = [{"tag": t, "weight": round(s / max_score, 3)} for t, s in top]
            method = "rule"
            stats["files"] += 1

            if fid not in fresh_set:
                # Unchanged: re-scored tags only; summary/entities stay.
                # If the tag sequence is identical to what's stored
                # (the usual case — IDF drift rarely reorders the top
                # 5), skip the write entirely.
                stats["rule"] += 1
                if [t["tag"] for t in tags] == existing_topics.get(fid, []):
                    continue
                recomputed_ids.append(fid)
                topic_rows.extend(
                    (fid, t["tag"], t["weight"], method) for t in tags
                )
                meta_rows.append((fid, fhash, prev[1] if prev else method))
                continue
            recomputed_ids.append(fid)

            extra = json.loads(extra_json or "{}")
            text = fresh_texts.get(fid, "")
            if kind == "data":
                card = profile_card_text(extra, extra.get("row_count"))
                summary = card.splitlines()[0] + "; " + ", ".join(
                    (c.get("name") if isinstance(c, dict) else str(c))
                    for c in extra.get("columns", [])[:6]
                )
                entities = list(extra.get("tables") or []) + list(
                    (extra.get("sheets") or {}).keys()
                )
                package = _llm_package_data(name, card)
            else:
                segments = hotspot_segments(text, scores)
                if segments:
                    first = segments[0].splitlines()[0].strip()
                    summary = (
                        first if len(first) >= 20 else segments[0].replace("\n", " ")
                    )[:200]
                else:
                    summary = ""
                entities = extract_entities(text)
                package = _llm_package_doc(
                    name, title,
                    [t for t, _ in counts.most_common(10)],
                    [t for t, _ in ranked[:15]],
                    segments,
                )

            if use_llm and (llm_file_ids is None or fid in llm_file_ids):
                prompt = (
                    "Read this file profile and reply with JSON only: "
                    '{"tags": [up to 5 short topic tags], "summary": "one sentence", '
                    '"entities": [up to 5 key people/places/projects/organizations]}.\n'
                    + package
                )
                result = _llm_call(prompt)
                if result is not None:
                    tags = result["tags"] or tags
                    summary = result["summary"] or summary
                    entities = result["entities"] or entities
                    method = "llm"
            stats[method] += 1
            topic_rows.extend((fid, t["tag"], t["weight"], method) for t in tags)
            entity_rows.extend((fid, e) for e in entities[:5])
            summary_rows.append((summary, fid))
            meta_rows.append((fid, fhash, method))

        con.execute("BEGIN TRANSACTION")
        _delete_for(con, "file_topics", recomputed_ids)
        if topic_rows:
            con.executemany("INSERT INTO file_topics VALUES (?,?,?,?)", topic_rows)
        _delete_for(con, "file_entities", fresh_ids)
        if entity_rows:
            con.executemany("INSERT INTO file_entities VALUES (?,?)", entity_rows)
        if summary_rows:
            con.executemany(
                "UPDATE file_text SET summary = ? WHERE file_id = ?", summary_rows
            )
        if meta_rows:
            con.executemany(
                "INSERT OR REPLACE INTO file_topic_meta VALUES (?,?,?)", meta_rows
            )
        con.execute("COMMIT")
        return stats
    finally:
        if own_con:
            con.close()
