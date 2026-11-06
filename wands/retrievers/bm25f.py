"""Field-weighted BM25: one index per field, and a query's score is a weighted sum across fields.

This is not BM25F proper, which merges term frequencies across fields before saturating them. The
name stays because the design calls it that; docs/adr/001 says what the difference costs.
"""

from __future__ import annotations

import bm25s
import numpy as np

from wands.catalog import Catalog
from wands.text import tokenize
from wands.types import Hit, Query


class FieldIndex:
    """A bm25s index over one list of texts."""

    def __init__(self, texts: list[str], k1: float, b: float, stem: bool):
        self.stem = stem
        self.size = len(texts)
        self.bm25 = bm25s.BM25(k1=k1, b=b)
        self.bm25.index([tokenize(t, stem) or ["__empty__"] for t in texts], show_progress=False)

    def scores(self, tokens: list[str]) -> np.ndarray:
        ids = self.bm25.get_tokens_ids(tokens)
        if not ids:
            return np.zeros(self.size, dtype=np.float32)
        return self.bm25.get_scores_from_ids(ids)


class IndexStore:
    """Builds each (field, k1, b, stem) index once, because configs share them."""

    def __init__(self, catalog: Catalog):
        self.catalog = catalog
        self._built: dict[tuple, FieldIndex] = {}

    def get(self, field: str, k1: float, b: float, stem: bool) -> FieldIndex:
        key = (field, k1, b, stem)
        if key not in self._built:
            texts = self.catalog.all_text() if field == "all" else self.catalog.fields[field]
            self._built[key] = FieldIndex(texts, k1, b, stem)
        return self._built[key]


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Positions of the k best scores, best first, with ties broken by position so runs are repeatable."""
    k = min(k, scores.size)
    candidates = np.argpartition(-scores, k - 1)[:k]
    return candidates[np.lexsort((candidates, -scores[candidates]))]


class BM25FRetriever:
    def __init__(
        self,
        store: IndexStore,
        field_weights: dict[str, float],
        k1: float = 1.5,
        b: float = 0.75,
        stem: bool = True,
        name: str = "bm25f",
    ):
        self.name = name
        self.stem = stem
        self.product_ids = store.catalog.product_ids
        self.indexes = {f: store.get(f, k1, b, stem) for f, w in field_weights.items() if w > 0}
        self.weights = {f: w for f, w in field_weights.items() if w > 0}

    def score_all(self, text: str) -> np.ndarray:
        tokens = tokenize(text, self.stem)
        total = np.zeros(len(self.product_ids), dtype=np.float32)
        for field, index in self.indexes.items():
            total += self.weights[field] * index.scores(tokens)
        return total

    def search(self, q: Query, k: int) -> list[Hit]:
        scores = self.score_all(q.text)
        return [
            Hit(product_id=int(self.product_ids[i]), score=float(scores[i]))
            for i in top_k(scores, k)
            if scores[i] > 0
        ]
