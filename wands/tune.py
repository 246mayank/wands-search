"""Small dev-only grids. Every config tried is logged to results/, so a best row can't be cherry-picked."""

from __future__ import annotations

import csv
import itertools
from pathlib import Path

import numpy as np
import yaml

from wands.catalog import FIELDS
from wands.eval.metrics import ndcg_at
from wands.pipeline import World, build, load_config
from wands.retrievers.bm25f import top_k
from wands.retrievers.fusion import order, weighted_scores
from wands.text import tokenize
from wands.types import Query

BM25_GRID = {
    "name": [1, 2, 3, 4],
    "class": [0, 1, 2],
    "hierarchy": [0, 0.5, 1],
    "features": [0, 0.5, 1],
    "description": [0, 0.25, 0.5],
}
ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
W_CLASS = [0.05, 0.1, 0.2, 0.3, 0.5]
W_ATTR = [0, 0.05, 0.1, 0.2]


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


def mean_ndcg(ranked: dict[int, list[int]], world: World, query_ids: list[int]) -> float:
    scores = [ndcg_at(ranked[q], world.qrels.get(q, {}), 10) for q in query_ids]
    return round(float(np.nanmean(scores)), 6)


def tune_fusion(world: World, query_ids: list[int], config: Path) -> list[dict]:
    """Every alpha for the weighted sum, on the two lists of a fusion config. RRF has nothing to tune here."""
    fusion = build(load_config(config)["pipeline"], world)
    lists = {
        qid: [r.search(Query(text=world.queries[qid]), fusion.depth) for r in fusion.retrievers]
        for qid in query_ids
    }
    rows = []
    for alpha in ALPHAS:
        ranked = {qid: [pid for pid, _ in order(weighted_scores(ls, alpha), 10)] for qid, ls in lists.items()}
        rows.append({"alpha": alpha, "ndcg@10": mean_ndcg(ranked, world, query_ids)})
    return sorted(rows, key=lambda row: row["ndcg@10"], reverse=True)


def tune_qu(world: World, query_ids: list[int], config: Path) -> list[dict]:
    """Every (w_class, w_attr) for a soft-boost config, reusing one set of fused results."""
    booster = build(load_config(config)["pipeline"], world)
    hits = {qid: booster.inner.search(Query(text=world.queries[qid]), booster.depth) for qid in query_ids}
    rows = []
    for w_class, w_attr in itertools.product(W_CLASS, W_ATTR):
        booster.w_class, booster.w_attr = w_class, w_attr
        ranked = {
            qid: [h.product_id for h in booster.rescore(Query(text=world.queries[qid]), found)[:10]]
            for qid, found in hits.items()
        }
        rows.append({"w_class": w_class, "w_attr": w_attr, "ndcg@10": mean_ndcg(ranked, world, query_ids)})
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
