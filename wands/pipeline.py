"""Build a retriever pipeline from a YAML config, and run it over a split."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from wands import data
from wands.catalog import Catalog
from wands.eval.qrels import Qrels, build_qrels
from wands.retrievers.bm25f import BM25FRetriever, IndexStore
from wands.types import Hit, Query, Retriever


@dataclass
class World:
    """Everything a pipeline is built from. Loaded once per command."""

    catalog: Catalog
    store: IndexStore
    qrels: Qrels
    queries: dict[int, str]
    query_classes: dict[int, str]

    @classmethod
    def load(cls) -> World:
        wands = data.load()
        catalog = Catalog.from_products(wands.products)
        queries = dict(wands.queries.select("query_id", "query").iter_rows())
        classes = dict(wands.queries.select("query_id", "query_class").iter_rows())
        return cls(catalog, IndexStore(catalog), build_qrels(wands.labels), queries, classes)


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build(spec: dict, world: World, name: str = "pipeline") -> Retriever:
    kind = spec["type"]
    if kind == "bm25f":
        return BM25FRetriever(
            world.store,
            spec["field_weights"],
            k1=spec.get("k1", 1.5),
            b=spec.get("b", 0.75),
            stem=spec.get("stem", True),
            name=name,
        )
    raise ValueError(f"unknown pipeline type {kind!r}")


def run_pipeline(retriever: Retriever, world: World, query_ids: list[int], k: int) -> dict[int, list[Hit]]:
    return {qid: retriever.search(Query(text=world.queries[qid], query_id=qid), k) for qid in query_ids}
