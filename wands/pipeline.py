"""Build a retriever pipeline from a YAML config, and run it over a split."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from wands import data
from wands.catalog import Catalog
from wands.eval.qrels import Qrels, build_qrels
from wands.qu.cache import QUCache
from wands.qu.rewrite import REWRITES_PATH, RewriteRetriever, load_rewrites
from wands.qu.understand import load_table
from wands.rerank import CrossEncoderReranker, CrossEncoderScorer
from wands.retrievers.bm25f import BM25FRetriever, IndexStore
from wands.retrievers.booster import QUBooster
from wands.retrievers.dense import EMBEDDINGS_PATH, DenseRetriever, QueryEncoder
from wands.retrievers.fusion import RRF_K, FusionRetriever
from wands.retrievers.popularity import PopularityPrior
from wands.types import Hit, Query, Retriever


@dataclass
class World:
    """Everything a pipeline is built from. Loaded once per command."""

    catalog: Catalog
    store: IndexStore
    qrels: Qrels
    queries: dict[int, str]
    query_classes: dict[int, str]
    _embeddings: np.ndarray | None = None
    _encoder: QueryEncoder | None = None
    _qu_cache: QUCache | None = None
    _scorer: CrossEncoderScorer | None = None

    @classmethod
    def load(cls) -> World:
        wands = data.load()
        catalog = Catalog.from_products(wands.products)
        queries = dict(wands.queries.select("query_id", "query").iter_rows())
        classes = dict(wands.queries.select("query_id", "query_class").iter_rows())
        return cls(catalog, IndexStore(catalog), build_qrels(wands.labels), queries, classes)

    @property
    def embeddings(self) -> np.ndarray:
        if self._embeddings is None:
            self._embeddings = np.load(EMBEDDINGS_PATH)
        return self._embeddings

    @property
    def encoder(self) -> QueryEncoder:
        if self._encoder is None:
            self._encoder = QueryEncoder()
        return self._encoder

    @property
    def scorer(self) -> CrossEncoderScorer:
        if self._scorer is None:
            self._scorer = CrossEncoderScorer(self.catalog)
        return self._scorer

    @property
    def qu_cache(self) -> QUCache:
        if self._qu_cache is None:
            self._qu_cache = QUCache()
        return self._qu_cache


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build(spec: dict, world: World, name: str = "pipeline") -> Retriever:
    kind = spec["type"]
    if kind == "ref":
        # Pull in another config, so a tuned stage has one definition.
        return build(load_config(Path(spec["path"]))["pipeline"], world, spec["name"])
    if kind == "bm25f":
        return BM25FRetriever(
            world.store,
            spec["field_weights"],
            k1=spec.get("k1", 1.5),
            b=spec.get("b", 0.75),
            stem=spec.get("stem", True),
            name=spec.get("name", name),
        )
    if kind == "dense":
        return DenseRetriever(
            world.encoder, world.embeddings, world.catalog.product_ids, name=spec.get("name", name)
        )
    if kind == "fusion":
        return FusionRetriever(
            [build(child, world) for child in spec["retrievers"]],
            method=spec.get("method", "rrf"),
            k=spec.get("k", RRF_K),
            alpha=spec.get("alpha", 0.5),
            depth=spec.get("depth", 200),
            name=name,
        )
    if kind == "qu_boost":
        return QUBooster(
            build(spec["inner"], world),
            load_table(world.qu_cache, spec["model"], spec["prompt"]),
            world.catalog,
            w_class=spec.get("w_class", 0.2),
            w_attr=spec.get("w_attr", 0.1),
            mode=spec.get("mode", "soft"),
            depth=spec.get("depth", 200),
            name=name,
        )
    if kind == "rewrite":
        rewrites = load_rewrites(Path(spec.get("path", REWRITES_PATH)))
        return RewriteRetriever(build(spec["inner"], world), rewrites, name=name)
    if kind == "popularity":
        return PopularityPrior(build(spec["inner"], world), world.catalog, spec["weight"], name=name)
    if kind == "ltr":
        from wands.ltr import LTRRetriever, builder_for

        return LTRRetriever(builder_for(spec["inner"], world), name=name)
    if kind == "rerank":
        return CrossEncoderReranker(
            build(spec["inner"], world),
            world.scorer,
            depth=spec.get("depth", 50),
            mode=spec.get("mode", "score"),
            name=name,
        )
    raise ValueError(f"unknown pipeline type {kind!r}")


def find_booster(retriever: Retriever) -> QUBooster | None:
    """The query-understanding stage of a pipeline, if it has one."""
    node = retriever
    while node is not None:
        if isinstance(node, QUBooster):
            return node
        node = getattr(node, "inner", None)
    return None


def run_pipeline(retriever: Retriever, world: World, query_ids: list[int], k: int) -> dict[int, list[Hit]]:
    return {qid: retriever.search(Query(text=world.queries[qid], query_id=qid), k) for qid in query_ids}
