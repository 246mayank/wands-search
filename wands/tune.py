"""Small dev-only grids. Every config tried is logged to results/, so a best row can't be cherry-picked."""

from __future__ import annotations

import csv
import itertools
from pathlib import Path

import numpy as np
import yaml

from wands.catalog import FIELDS
from wands.eval.metrics import ndcg_at
from wands.pipeline import World
from wands.retrievers.bm25f import top_k
from wands.text import tokenize

BM25_GRID = {
    "name": [1, 2, 3, 4],
    "class": [0, 1, 2],
    "hierarchy": [0, 0.5, 1],
    "features": [0, 0.5, 1],
    "description": [0, 0.25, 0.5],
}


def tune_bm25(
    world: World, query_ids: list[int], k1: float = 1.5, b: float = 0.75, stem: bool = True
) -> list[dict]:
    """Mean dev NDCG@10 of every weight combination, best first."""
    indexes = {f: world.store.get(f, k1, b, stem) for f in FIELDS}
    per_query = []
    for qid in query_ids:
        judged = world.qrels.get(qid, {})
        if not any(judged.values()):
            continue
        tokens = tokenize(world.queries[qid], stem)
        field_scores = {f: index.scores(tokens) for f, index in indexes.items()}
        # Only products that match in some field can rank, which is far fewer than the whole catalogue.
        keep = np.flatnonzero(np.sum([s > 0 for s in field_scores.values()], axis=0))
        kept_scores = {f: s[keep] for f, s in field_scores.items()}
        per_query.append((judged, world.catalog.product_ids[keep], kept_scores))

    rows = []
    for values in itertools.product(*BM25_GRID.values()):
        weights = dict(zip(BM25_GRID, values, strict=True))
        total = 0.0
        for judged, ids, scores in per_query:
            combined = sum(weights[f] * scores[f] for f in FIELDS)
            total += ndcg_at(ids[top_k(combined, 10)].tolist(), judged, 10)
        rows.append({**weights, "ndcg@10": round(total / len(per_query), 6)})
    return sorted(rows, key=lambda row: row["ndcg@10"], reverse=True)


def write_grid(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_bm25_config(best: dict, path: Path, name: str = "bm25f", stem: bool = True) -> None:
    weights = {f: best[f] for f in FIELDS}
    config = {"name": name, "pipeline": {"type": "bm25f", "field_weights": weights, "stem": stem}}
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8", newline="\n")
