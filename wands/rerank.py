"""Cross-encoder reranking of the top of a ranked list."""

from __future__ import annotations

import json
from pathlib import Path

from wands.catalog import Catalog
from wands.retrievers.fusion import RRF_K, order
from wands.types import Hit, Query, Retriever

CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CE_CACHE_PATH = Path("data/ce_scores.jsonl")


class CrossEncoderScorer:
    """Scores (query, product) pairs. Scores are kept on disk, because tuning reruns the same pairs."""

    def __init__(self, catalog: Catalog, model_name: str = CE_MODEL, cache_path: Path | None = CE_CACHE_PATH):
        self.model_name = model_name
        self.positions = {int(pid): i for i, pid in enumerate(catalog.product_ids)}
        self.catalog = catalog
        self.model = None
        self.cache_path = cache_path
        self.seen: dict[str, dict[int, float]] = {}
        if cache_path and cache_path.exists():
            with cache_path.open(encoding="utf-8") as f:
                for line in f:
                    query, pid, score = json.loads(line)
                    self.seen.setdefault(query, {})[pid] = score

    def text(self, product_id: int) -> str:
        i = self.positions[product_id]
        parts = (
            self.catalog.fields["name"][i],
            self.catalog.fields["class"][i],
            self.catalog.attribute_text(i),
        )
        return " | ".join(p for p in parts if p)

    def score(self, query: str, product_ids: list[int]) -> list[float]:
        known = self.seen.setdefault(query, {})
        missing = [pid for pid in product_ids if pid not in known]
        if missing:
            if self.model is None:
                from sentence_transformers import CrossEncoder

                self.model = CrossEncoder(self.model_name, device="cpu")
            scores = self.model.predict(
                [(query, self.text(pid)) for pid in missing], batch_size=64, show_progress_bar=False
            )
            new = {pid: float(score) for pid, score in zip(missing, scores, strict=True)}
            known.update(new)
            if self.cache_path:
                with self.cache_path.open("a", encoding="utf-8", newline="\n") as f:
                    for pid, score in new.items():
                        f.write(json.dumps([query, pid, score]) + "\n")
        return [known[pid] for pid in product_ids]


class CrossEncoderReranker:
    """Reorder the top `depth` hits; the rest keep their order below them.

    mode "score" trusts the cross-encoder alone. Mode "rrf" blends its order with the incoming order, for
    a model that wasn't trained on this domain.
    """

    def __init__(
        self,
        inner: Retriever,
        scorer: CrossEncoderScorer,
        depth: int = 50,
        mode: str = "score",
        name: str = "rerank",
    ):
        if mode not in ("score", "rrf"):
            raise ValueError(f"unknown rerank mode {mode!r}")
        self.name = name
        self.inner = inner
        self.scorer = scorer
        self.depth = depth
        self.mode = mode

    def search(self, q: Query, k: int) -> list[Hit]:
        hits = self.inner.search(q, max(k, self.depth))
        head, tail = hits[: self.depth], hits[self.depth :]
        scores = self.scorer.score(q.text, [h.product_id for h in head])
        return (self.reorder(head, scores) + tail)[:k]

    def reorder(self, head: list[Hit], scores: list[float]) -> list[Hit]:
        by_id = {hit.product_id: hit for hit in head}
        ce_score = {hit.product_id: s for hit, s in zip(head, scores, strict=True)}
        ce_order = [pid for pid, _ in order(ce_score, len(head))]
        if self.mode == "score":
            final = dict(order(ce_score, len(head)))
        else:
            blended = {hit.product_id: 1 / (RRF_K + rank) for rank, hit in enumerate(head, start=1)}
            for rank, pid in enumerate(ce_order, start=1):
                blended[pid] += 1 / (RRF_K + rank)
            final = dict(order(blended, len(head)))
        return [
            Hit(
                product_id=pid,
                score=score,
                debug={**by_id[pid].debug, "rerank_score": ce_score[pid], "input_score": by_id[pid].score},
            )
            for pid, score in final.items()
        ]
