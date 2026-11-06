"""Ranking metrics for one query. All take a ranked list of product ids and that query's labels.

A metric with no possible score for a query (nothing relevant to find) returns NaN, and means skip
it, so such a query neither helps nor hurts. Unjudged products count as gain 0; judged_at reports
how often that happened.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from wands.eval.qrels import EXACT

NAN = float("nan")

METRICS = ["ndcg@10", "ndcg_exact@10", "mrr@10", "recall@100", "capped_recall@100", "judged@10"]


def dcg(gains: Sequence[int]) -> float:
    """Linear gain, log2 discount: rank 1 divides by log2(2) = 1."""
    return sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))


def ndcg_at(ranked: Sequence[int], judged: dict[int, int], k: int = 10, exact_only: bool = False) -> float:
    """Graded NDCG@k. The ideal comes from every label of the query, not only from what was retrieved."""
    if exact_only:
        judged = {pid: 1 for pid, gain in judged.items() if gain == EXACT}
    ideal = dcg(sorted(judged.values(), reverse=True)[:k])
    if ideal == 0:
        return NAN
    return dcg([judged.get(pid, 0) for pid in ranked[:k]]) / ideal


def mrr_at(ranked: Sequence[int], judged: dict[int, int], k: int = 10) -> float:
    """Reciprocal rank of the first Exact product in the top k."""
    if EXACT not in judged.values():
        return NAN
    for rank, pid in enumerate(ranked[:k], start=1):
        if judged.get(pid, 0) == EXACT:
            return 1 / rank
    return 0.0


def recall_at(ranked: Sequence[int], judged: dict[int, int], k: int = 100) -> float:
    """Share of the query's Exact + Partial products found in the top k."""
    relevant = sum(gain > 0 for gain in judged.values())
    if relevant == 0:
        return NAN
    return sum(judged.get(pid, 0) > 0 for pid in ranked[:k]) / relevant


def capped_recall_at(ranked: Sequence[int], judged: dict[int, int], k: int = 100) -> float:
    """Recall where 1.0 is reachable: a query with 800 relevant products is divided by k, not by 800."""
    relevant = sum(gain > 0 for gain in judged.values())
    if relevant == 0:
        return NAN
    return sum(judged.get(pid, 0) > 0 for pid in ranked[:k]) / min(relevant, k)


def judged_at(ranked: Sequence[int], judged: dict[int, int], k: int = 10) -> float:
    """Share of the top k that has a human label."""
    top = ranked[:k]
    if not top:
        return NAN
    return sum(pid in judged for pid in top) / len(top)


def score_query(ranked: Sequence[int], judged: dict[int, int]) -> dict[str, float]:
    return {
        "ndcg@10": ndcg_at(ranked, judged, 10),
        "ndcg_exact@10": ndcg_at(ranked, judged, 10, exact_only=True),
        "mrr@10": mrr_at(ranked, judged, 10),
        "recall@100": recall_at(ranked, judged, 100),
        "capped_recall@100": capped_recall_at(ranked, judged, 100),
        "judged@10": judged_at(ranked, judged, 10),
    }
