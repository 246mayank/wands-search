"""Learning to rank over the fused candidates, trained on dev labels only.

Dev scores come from 5-fold cross-validation by query, so no query is scored by a model that saw it.
Training on all of dev, for the single test run, happens only after that.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from wands.pipeline import World, find_booster
from wands.retrievers.booster import QUBooster
from wands.text import tokenize
from wands.types import Hit, Query

MODEL_PATH = Path("cache/ltr_model.txt")
SEED = 20261106
FOLDS = 5
FEATURES = [
    "bm25_name",
    "bm25_class",
    "bm25_hierarchy",
    "bm25_features",
    "bm25_description",
    "bm25_weighted",
    "dense",
    "rrf",
    "bm25_rank",
    "dense_rank",
    "qu_class_match",
    "qu_attr_share",
    "qu_confidence",
    "qu_answered",
    "name_overlap",
    "query_tokens",
    "log_rating_count",
    "average_rating",
    "log_review_count",
]
CONFIDENCE = {"low": 1.0, "medium": 2.0, "high": 3.0}
MISSING_RANK = 1000.0


class FeatureBuilder:
    """One feature row per candidate of a query. Candidates are the fused top 200, before any boost."""

    def __init__(self, world: World, booster: QUBooster):
        self.world = world
        self.booster = booster
        self.fusion = booster.inner
        self.bm25, self.dense = self.fusion.retrievers
        self.positions = {int(pid): i for i, pid in enumerate(world.catalog.product_ids)}
        self.numeric = world.catalog.numeric

    def rows(self, text: str) -> tuple[list[int], np.ndarray]:
        q = Query(text=text)
        hits = self.fusion.search(q, self.fusion.depth)
        ids = [hit.product_id for hit in hits]
        if not ids:
            return ids, np.empty((0, len(FEATURES)))
        at = np.array([self.positions[pid] for pid in ids])
        tokens = tokenize(text, self.bm25.stem)
        token_set = set(tokens)
        qu, _ = self.booster.table.lookup(text)
        wanted = set(qu.product_classes) if qu else set()
        field_scores = {field: index.scores(tokens)[at] for field, index in self.bm25.indexes.items()}
        dense = self.dense.embeddings[at] @ self.dense.encoder.encode(text)
        rows = np.zeros((len(ids), len(FEATURES)), dtype=np.float64)
        column = {name: j for j, name in enumerate(FEATURES)}
        for field, values in field_scores.items():
            rows[:, column[f"bm25_{field}"]] = values
        rows[:, column["bm25_weighted"]] = sum(self.bm25.weights[f] * v for f, v in field_scores.items())
        rows[:, column["dense"]] = dense
        rows[:, column["qu_answered"]] = 1.0 if qu else 0.0
        rows[:, column["qu_confidence"]] = CONFIDENCE[qu.confidence] if qu else 0.0
        rows[:, column["query_tokens"]] = len(tokens)
        for name in ("log_rating_count", "average_rating", "log_review_count"):
            rows[:, column[name]] = self.numeric[name][at]
        for i, hit in enumerate(hits):
            rows[i, column["rrf"]] = hit.score
            rows[i, column["bm25_rank"]] = hit.debug.get(f"{self.bm25.name}_rank", MISSING_RANK)
            rows[i, column["dense_rank"]] = hit.debug.get(f"{self.dense.name}_rank", MISSING_RANK)
            name_tokens = set(tokenize(self.world.catalog.fields["name"][at[i]], self.bm25.stem))
            rows[i, column["name_overlap"]] = len(token_set & name_tokens) / max(1, len(token_set))
            if qu:
                rows[i, column["qu_class_match"]] = float(bool(self.booster.classes[hit.product_id] & wanted))
                if qu.attributes:
                    matched = self.booster.matcher.matches(hit.product_id, qu.attributes)
                    rows[i, column["qu_attr_share"]] = matched / len(qu.attributes)
        return ids, rows


def make_ranker():
    import lightgbm as lgb

    return lgb.LGBMRanker(
        objective="lambdarank",
        label_gain=[0, 1, 2],  # the same linear gains as the metric, not LightGBM's 2^label - 1
        lambdarank_truncation_level=10,
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=SEED,
        deterministic=True,
        force_row_wise=True,
        n_jobs=8,
        verbose=-1,
    )


def training_set(data: dict[int, tuple[list[int], np.ndarray]], world: World, query_ids: list[int]):
    """Judged candidates only: an unjudged product isn't known to be irrelevant."""
    xs, ys, groups = [], [], []
    for qid in query_ids:
        ids, rows = data[qid]
        judged = world.qrels.get(qid, {})
        keep = [i for i, pid in enumerate(ids) if pid in judged]
        if len(keep) < 2:
            continue
        xs.append(rows[keep])
        ys.append(np.array([judged[ids[i]] for i in keep]))
        groups.append(len(keep))
    return np.vstack(xs), np.concatenate(ys), groups


def fit(data, world: World, query_ids: list[int]):
    x, y, groups = training_set(data, world, query_ids)
    return make_ranker().fit(x, y, group=groups, feature_name=FEATURES)


def ordered(model, ids: list[int], rows: np.ndarray) -> list[tuple[int, float]]:
    if not ids:
        return []
    scores = model.predict(rows)
    return sorted(zip(ids, map(float, scores), strict=True), key=lambda item: (-item[1], item[0]))


def folds_of(query_ids: list[int]) -> list[list[int]]:
    shuffled = sorted(query_ids)
    random.Random(SEED).shuffle(shuffled)
    return [shuffled[i::FOLDS] for i in range(FOLDS)]


def cross_validate(data, world: World, query_ids: list[int]) -> tuple[dict[int, list[int]], np.ndarray]:
    """Out-of-fold rankings for every query, and feature importance summed over the folds."""
    ranked: dict[int, list[int]] = {}
    importance = np.zeros(len(FEATURES))
    for held_out in folds_of(query_ids):
        train = [qid for qid in query_ids if qid not in set(held_out)]
        model = fit(data, world, train)
        importance += model.booster_.feature_importance(importance_type="gain")
        for qid in held_out:
            ids, rows = data[qid]
            ranked[qid] = [pid for pid, _ in ordered(model, ids, rows)]
    return ranked, importance / importance.sum()


class LTRRetriever:
    """The model trained on all of dev. Never score dev with it: that is in-sample."""

    def __init__(self, features: FeatureBuilder, model_path: Path = MODEL_PATH, name: str = "ltr"):
        import lightgbm as lgb

        self.name = name
        self.features = features
        self.model = lgb.Booster(model_file=str(model_path))

    def search(self, q: Query, k: int) -> list[Hit]:
        ids, rows = self.features.rows(q.text)
        found = ordered(self.model, ids, rows)[:k]
        return [Hit(product_id=pid, score=score) for pid, score in found]


def builder_for(config_pipeline: dict, world: World) -> FeatureBuilder:
    from wands.pipeline import build

    booster = find_booster(build(config_pipeline, world))
    if booster is None:
        raise ValueError("the LTR features need a pipeline with a QU stage")
    return FeatureBuilder(world, booster)
