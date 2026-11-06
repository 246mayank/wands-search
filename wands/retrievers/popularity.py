"""A popularity prior: does adding a product's rating count help, on top of relevance?"""

from __future__ import annotations

import numpy as np

from wands.catalog import Catalog
from wands.retrievers.fusion import minmax
from wands.types import Hit, Query, Retriever


class PopularityPrior:
    """score = inner_norm + weight x log(1 + rating_count) scaled to 0..1. No ratings counts as 0."""

    def __init__(
        self, inner: Retriever, catalog: Catalog, weight: float, depth: int = 200, name: str = "pop"
    ):
        self.name = name
        self.inner = inner
        self.weight = weight
        self.depth = depth
        counts = np.nan_to_num(catalog.numeric["log_rating_count"], nan=0.0)
        top = counts.max()
        scaled = counts / top if top > 0 else counts
        self.popularity = {int(pid): float(v) for pid, v in zip(catalog.product_ids, scaled, strict=True)}

    def search(self, q: Query, k: int) -> list[Hit]:
        hits = self.inner.search(q, max(k, self.depth))
        norm = minmax(hits)
        scored = [
            hit.model_copy(
                update={"score": norm[hit.product_id] + self.weight * self.popularity[hit.product_id]}
            )
            for hit in hits
        ]
        return sorted(scored, key=lambda h: (-h.score, h.product_id))[:k]
