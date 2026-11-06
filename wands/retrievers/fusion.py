"""Combine ranked lists: reciprocal rank fusion, or a min-max normalised weighted sum."""

from __future__ import annotations

from wands.types import Hit, Query, Retriever

RRF_K = 60


def rrf_scores(lists: list[list[Hit]], k: int = RRF_K) -> dict[int, float]:
    """Each list gives a product 1 / (k + rank). Only ranks are used, so score scales don't matter."""
    scores: dict[int, float] = {}
    for hits in lists:
        for rank, hit in enumerate(hits, start=1):
            scores[hit.product_id] = scores.get(hit.product_id, 0.0) + 1 / (k + rank)
    return scores


def minmax(hits: list[Hit]) -> dict[int, float]:
    """Scale a list's scores to 0..1. The list's own best and worst set the scale."""
    if not hits:
        return {}
    values = [hit.score for hit in hits]
    low, high = min(values), max(values)
    span = high - low
    return {hit.product_id: (hit.score - low) / span if span > 0 else 1.0 for hit in hits}


def weighted_scores(lists: list[list[Hit]], alpha: float) -> dict[int, float]:
    """alpha x first list + (1 - alpha) x second, each normalised. A missing product counts as 0."""
    first, second = minmax(lists[0]), minmax(lists[1])
    return {
        pid: alpha * first.get(pid, 0.0) + (1 - alpha) * second.get(pid, 0.0)
        for pid in first.keys() | second.keys()
    }


def order(scores: dict[int, float], k: int) -> list[tuple[int, float]]:
    """Best first, with product id breaking ties so runs are repeatable."""
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]


class FusionRetriever:
    def __init__(
        self,
        retrievers: list[Retriever],
        method: str = "rrf",
        k: int = RRF_K,
        alpha: float = 0.5,
        depth: int = 200,
        name: str = "fusion",
    ):
        if method not in ("rrf", "weighted"):
            raise ValueError(f"unknown fusion method {method!r}")
        if method == "weighted" and len(retrievers) != 2:
            raise ValueError("weighted fusion takes exactly two retrievers")
        self.name = name
        self.retrievers = retrievers
        self.method = method
        self.k = k
        self.alpha = alpha
        self.depth = depth

    def fuse(self, lists: list[list[Hit]], k: int) -> list[Hit]:
        scores = rrf_scores(lists, self.k) if self.method == "rrf" else weighted_scores(lists, self.alpha)
        ranks = [{hit.product_id: rank for rank, hit in enumerate(hits, start=1)} for hits in lists]
        hits = []
        for pid, score in order(scores, k):
            debug: dict[str, float | str] = {
                f"{r.name}_rank": rank[pid]
                for r, rank in zip(self.retrievers, ranks, strict=True)
                if pid in rank
            }
            hits.append(Hit(product_id=pid, score=score, debug=debug))
        return hits

    def search(self, q: Query, k: int) -> list[Hit]:
        return self.fuse([r.search(q, self.depth) for r in self.retrievers], k)
