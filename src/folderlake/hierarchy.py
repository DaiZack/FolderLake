"""Hierarchical topic tree — agglomerative clustering per file kind.

Flat per-file tags stay as metadata; the *organisation* of topics is
this tree. Files of each kind (document, data, text) are clustered
separately — the kinds are never mixed, per the product decision that
their browsing scenarios differ.

Method: corpus TF-IDF vectors from the stored per-file term counts
(log TF x IDF, L2-normalised, light DF hygiene), then agglomerative
clustering with centroid linkage. The merge tree becomes a display
tree of at most MAX_DEPTH levels: a sub-cluster earns its own node
when it holds at least MIN_GROUP files (smaller branches simply stay
covered by their parent's member list), and a single child that holds
essentially all of its parent's files replaces the parent instead of
adding an empty level. Node labels are the cluster's top phrase terms
(the same bonus/suppression rules as per-file tags, in topics.py).

topic_node_files stores each node's FULL subtree membership, so
filtering by any node is a single lookup.

Scale honesty: centroid agglomeration is O(N^2) pairwise work, so a
kind with more than MAX_FILES_PER_KIND files gets no tree (the viewer
falls back to the flat topic list) rather than a slow scan.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import duckdb

from .topics import _select_tags, _term_bonus

MIN_GROUP = 3
MAX_DEPTH = 4
# A cluster bigger than this whose halves both qualify as groups is
# split at the SAME display level instead of becoming one wide node —
# otherwise a binary merge tree gives every level at most 2 children
# and the tree stays sparse.
WIDE_SPLIT = 12
MAX_FILES_PER_KIND = 400

KIND_LABELS = {"document": "Documents", "data": "Data files", "text": "Texts"}


# ----------------------------------------------------------------------
# Vectors
# ----------------------------------------------------------------------
def _vectors(terms_by_file: dict[str, Counter]) -> dict[str, dict[str, float]]:
    n = len(terms_by_file)
    df: Counter = Counter()
    for counts in terms_by_file.values():
        df.update(counts.keys())
    vectors: dict[str, dict[str, float]] = {}
    for fid, counts in terms_by_file.items():
        vec: dict[str, float] = {}
        for term, cnt in counts.items():
            d = df[term]
            if n >= 10 and (d < 2 or d > 0.7 * n):
                continue  # DF hygiene: hapax noise + corpus wallpaper
            idf = math.log((n + 1) / (d + 1)) + 1.0
            vec[term] = math.log1p(cnt) * idf
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        vectors[fid] = {t: v / norm for t, v in vec.items()}
    return vectors


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(t, 0.0) for t, v in a.items())


# ----------------------------------------------------------------------
# Agglomerative clustering (centroid linkage)
# ----------------------------------------------------------------------
class _Cluster:
    __slots__ = ("members", "total", "left", "right", "sim")

    def __init__(self, members, total, left=None, right=None, sim=0.0):
        self.members = members          # file ids in this subtree
        self.total = total              # summed member vectors (dict)
        self.left, self.right, self.sim = left, right, sim

    def centroid(self) -> dict[str, float]:
        norm = math.sqrt(sum(v * v for v in self.total.values())) or 1.0
        inv = 1.0 / norm
        return {t: v * inv for t, v in self.total.items()}


def _agglomerate(vectors: dict[str, dict[str, float]]) -> _Cluster:
    active: dict[int, _Cluster] = {
        i: _Cluster([fid], dict(vec))
        for i, (fid, vec) in enumerate(vectors.items())
    }
    sims: dict[tuple[int, int], float] = {}

    def sim(i: int, j: int) -> float:
        key = (i, j) if i < j else (j, i)
        if key not in sims:
            sims[key] = _cosine(active[i].centroid(), active[j].centroid())
        return sims[key]

    next_id = len(active)
    while len(active) > 1:
        ids = list(active)
        best, best_pair = -2.0, None
        for x in range(len(ids)):
            for y in range(x + 1, len(ids)):
                s = sim(ids[x], ids[y])
                if s > best:
                    best, best_pair = s, (ids[x], ids[y])
        i, j = best_pair
        a, b = active.pop(i), active.pop(j)
        total = dict(a.total)
        for t, v in b.total.items():
            total[t] = total.get(t, 0.0) + v
        active[next_id] = _Cluster(a.members + b.members, total,
                                   left=a, right=b, sim=max(best, 0.0))
        sims = {k: v for k, v in sims.items() if i not in k and j not in k}
        next_id += 1
    return next(iter(active.values()))


# ----------------------------------------------------------------------
# Display tree
# ----------------------------------------------------------------------
def _label(cluster: _Cluster, corpus_total: dict[str, float],
           corpus_size: float) -> str:
    """Label by lift: terms over-represented in this cluster versus the
    whole kind corpus. Raw frequency crowns corpus wallpaper ('more',
    header boilerplate) that appears everywhere at the same rate; lift
    sends those to zero and surfaces what the group is actually about.
    """
    cluster_size = sum(cluster.total.values()) or 1.0
    ranked = []
    for term, value in cluster.total.items():
        share_in = value / cluster_size
        share_all = corpus_total.get(term, 0.0) / (corpus_size or 1.0)
        lift = math.log((share_in + 1e-9) / (share_all + 1e-9))
        if lift > 0:
            ranked.append((term, value * lift * _term_bonus(term)))
    ranked.sort(key=lambda kv: -kv[1])
    top = _select_tags(ranked, Counter(cluster.total), limit=3)
    return " · ".join(t for t, _ in top) or "misc"


def _should_flatten(cluster: _Cluster) -> bool:
    """A forced marriage: two internally-cohesive halves that barely
    resemble each other. Such a node gets no display level — its
    children are promoted to its parent instead. (A binary dendrogram
    otherwise invents meaningless top groups, e.g. finance+garden
    labelled 'finance'.)"""
    if cluster.left is None or cluster.right is None:
        return False
    internal = min(cluster.left.sim, cluster.right.sim)
    return internal > 0 and cluster.sim < 0.4 * internal


def _child_clusters(cluster: _Cluster, depth: int) -> list[_Cluster]:
    """The frontier of sub-clusters that become nodes at depth + 1.

    Walk down the merge tree from this cluster's halves: a branch
    becomes a node when it is a qualifying group; we keep descending
    through branches that are forced marriages (see _should_flatten)
    or simply too wide to be one topic (WIDE_SPLIT).
    """
    if cluster.left is None or depth >= MAX_DEPTH:
        return []
    out: list[_Cluster] = []
    stack = [cluster.left, cluster.right]
    while stack:
        c = stack.pop()
        if len(c.members) < MIN_GROUP:
            continue
        if c.left is not None and (
                _should_flatten(c) or len(c.members) > WIDE_SPLIT):
            stack.extend([c.left, c.right])
        else:
            out.append(c)
    return out


def build_tree_for_kind(root_cluster: _Cluster, kind: str,
                        corpus_total: dict[str, float] | None = None
                        ) -> list[dict]:
    """Flatten a merge tree into display nodes.

    Each node dict: prefixed id, parent id (or None), kind, label,
    depth, members (FULL subtree file ids), cohesion.
    """
    if corpus_total is None:
        corpus_total = dict(root_cluster.total)
    corpus_size = sum(corpus_total.values()) or 1.0
    nodes: list[dict] = []

    def emit(cluster: _Cluster, parent_id: str | None, depth: int,
             label: str, cohesion: float) -> None:
        kids = _child_clusters(cluster, depth)
        # Chain collapse: one child holding (nearly) every file means
        # this level adds nothing — emit the child in our place. The
        # kind root is exempt: it carries the kind label.
        if parent_id is not None and len(kids) == 1 and \
                len(kids[0].members) >= len(cluster.members) - (MIN_GROUP - 1):
            emit(kids[0], parent_id, depth,
                 _label(kids[0], corpus_total, corpus_size), kids[0].sim)
            return
        node_id = f"{kind}-{len(nodes) + 1}"
        nodes.append({
            "id": node_id, "parent": parent_id, "kind": kind,
            "label": label, "depth": depth,
            "members": list(cluster.members),
            "cohesion": round(cohesion, 3),
        })
        for c in kids:
            emit(c, node_id, depth + 1,
                 _label(c, corpus_total, corpus_size), c.sim)

    emit(root_cluster, None, 0, KIND_LABELS.get(kind, kind), 1.0)
    return nodes


# ----------------------------------------------------------------------
# Build + store
# ----------------------------------------------------------------------
def build_hierarchy(db_path: Path) -> dict:
    con = duckdb.connect(str(db_path))
    try:
        con.execute(
            "CREATE TABLE IF NOT EXISTS topic_nodes ("
            "id VARCHAR PRIMARY KEY, parent_id VARCHAR, kind VARCHAR, "
            "label VARCHAR, depth INTEGER, file_count INTEGER, cohesion DOUBLE)"
        )
        con.execute(
            "CREATE TABLE IF NOT EXISTS topic_node_files ("
            "node_id VARCHAR, file_id VARCHAR)"
        )
        terms: dict[str, dict[str, Counter]] = {}
        for fid, kind, term, cnt in con.execute(
            "SELECT f.id, f.kind, t.term, t.cnt FROM files f "
            "JOIN file_terms t ON t.file_id = f.id"
        ).fetchall():
            terms.setdefault(kind, {}).setdefault(fid, Counter())[term] = cnt
        all_files: dict[str, list[str]] = {}
        for fid, kind in con.execute("SELECT id, kind FROM files").fetchall():
            all_files.setdefault(kind, []).append(fid)

        node_rows, member_rows = [], []
        skipped: dict[str, int] = {}
        for kind, file_ids in sorted(all_files.items()):
            if len(file_ids) > MAX_FILES_PER_KIND:
                skipped[kind] = len(file_ids)
                continue
            terms_by_file = terms.get(kind, {})
            vectors = _vectors({fid: terms_by_file.get(fid, Counter())
                                for fid in file_ids})
            corpus_total: dict[str, float] = {}
            for vec in vectors.values():
                for t, v in vec.items():
                    corpus_total[t] = corpus_total.get(t, 0.0) + v
            tree = build_tree_for_kind(_agglomerate(vectors), kind,
                                       corpus_total)
            for node in tree:
                node_rows.append((
                    node["id"], node["parent"], kind, node["label"],
                    node["depth"], len(node["members"]), node["cohesion"],
                ))
                member_rows.extend(
                    (node["id"], fid) for fid in node["members"])
        con.execute("BEGIN TRANSACTION")
        con.execute("DELETE FROM topic_nodes")
        con.execute("DELETE FROM topic_node_files")
        if node_rows:
            con.executemany("INSERT INTO topic_nodes VALUES (?,?,?,?,?,?,?)",
                            node_rows)
        if member_rows:
            con.executemany("INSERT INTO topic_node_files VALUES (?,?)",
                            member_rows)
        con.execute("COMMIT")
        return {"nodes": len(node_rows), "kinds": sorted(all_files),
                "skipped_too_large": skipped}
    finally:
        con.close()
