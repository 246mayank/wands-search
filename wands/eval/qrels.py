"""Human relevance labels as {query_id: {product_id: gain}}."""

from __future__ import annotations

import polars as pl

GAINS = {"Exact": 2, "Partial": 1, "Irrelevant": 0}
EXACT = GAINS["Exact"]

Qrels = dict[int, dict[int, int]]


def build_qrels(labels: pl.DataFrame) -> Qrels:
    """A pair labelled twice keeps its highest gain (14 pairs disagree; see docs/DATA_NOTES.md)."""
    qrels: Qrels = {}
    for query_id, product_id, label in labels.select("query_id", "product_id", "label").iter_rows():
        judged = qrels.setdefault(query_id, {})
        judged[product_id] = max(judged.get(product_id, 0), GAINS[label])
    return qrels
