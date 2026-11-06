"""The frozen dev/test split of the 480 queries, stratified by query_class where a class is big enough."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

SPLITS_PATH = Path("splits.json")
SPLIT_SEED = 20261106
N_DEV = 320


def make_split(queries: Iterable[tuple[int, str]], n_dev: int = N_DEV, seed: int = SPLIT_SEED) -> dict:
    """Walk the classes in random order and give each its share of dev, carrying the rounding forward.

    Every class ends within one query of its 2/3 share, and dev has exactly n_dev queries. Most of the
    189 classes are too small for true stratification (98 have one query), hence "where possible".
    """
    by_class: dict[str, list[int]] = defaultdict(list)
    for query_id, query_class in queries:
        by_class[query_class].append(query_id)
    total = sum(len(ids) for ids in by_class.values())

    rng = random.Random(seed)
    classes = sorted(by_class)
    rng.shuffle(classes)
    dev: list[int] = []
    test: list[int] = []
    seen = 0
    for query_class in classes:
        ids = sorted(by_class[query_class])
        rng.shuffle(ids)
        seen += len(ids)
        take = (seen * n_dev + total // 2) // total - len(dev)
        dev.extend(ids[:take])
        test.extend(ids[take:])
    return {"seed": seed, "dev": sorted(dev), "test": sorted(test)}


def write_split(split: dict, path: Path = SPLITS_PATH) -> None:
    path.write_text(json.dumps(split) + "\n", encoding="utf-8", newline="\n")


def load_split(name: str, path: Path = SPLITS_PATH) -> list[int]:
    return json.loads(path.read_text(encoding="utf-8"))[name]
