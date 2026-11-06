"""An LLM relevance judge, calibrated against the human labels.

It measures agreement with annotators, and is used only to estimate what unjudged results would have
scored. It never replaces a human label: where a human label exists, that label wins.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wands.catalog import Catalog
from wands.eval.metrics import ndcg_at
from wands.eval.qrels import GAINS
from wands.qu.client import LLMClient, LLMTimeout
from wands.qu.understand import PROMPTS_DIR

CACHE_PATH = Path("cache/judge_cache.jsonl")
SEED = 20261106
PER_LABEL = 200
BATCH_SIZE = 20
LABELS = ["Irrelevant", "Partial", "Exact"]  # index = gain


@dataclass(frozen=True)
class Pair:
    query: str
    product_id: int
    product: str


def product_card(catalog: Catalog, i: int, description_chars: int = 300) -> str:
    fields = catalog.fields
    parts = (
        f"name: {fields['name'][i]}",
        f"class: {fields['class'][i]}",
        f"category: {fields['hierarchy'][i]}",
        f"attributes: {catalog.attribute_text(i)}",
        f"description: {fields['description'][i][:description_chars]}",
    )
    return " | ".join(parts)


def sample_labelled(
    qrels: dict, query_ids: list[int], per_label: int = PER_LABEL
) -> list[tuple[int, int, int]]:
    """(query_id, product_id, gain) triples: the same number per label, so Exact isn't swamped by Partial."""
    rng = random.Random(SEED)
    chosen = []
    for gain in sorted(GAINS.values()):
        pool = sorted(
            (qid, pid, g) for qid in query_ids for pid, g in qrels.get(qid, {}).items() if g == gain
        )
        chosen += rng.sample(pool, per_label)
    return chosen


def pair_key(model: str, prompt: str, pair: Pair) -> str:
    payload = json.dumps([model, prompt, pair.query, pair.product], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class JudgeCache:
    def __init__(self, path: Path = CACHE_PATH):
        self.path = path
        self.entries: dict[str, dict] = {}
        if path.exists():
            with path.open(encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line)
                    self.entries[entry["key"]] = entry

    def put(self, entry: dict) -> None:
        self.entries[entry["key"]] = entry
        with self.path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")


def _ask(client: LLMClient, template: str, pairs: list[Pair]) -> dict[int, str]:
    lines = [
        json.dumps({"id": i, "query": p.query, "product": p.product}, ensure_ascii=False)
        for i, p in enumerate(pairs)
    ]
    try:
        data = json.loads(client.complete(template.replace("{{items}}", "\n".join(lines))))
    except (LLMTimeout, ValueError):
        return {}
    rows = data.get("results", []) if isinstance(data, dict) else data
    return {
        row["id"]: row["label"]
        for row in rows
        if isinstance(row, dict) and row.get("label") in LABELS and isinstance(row.get("id"), int)
    }


def judge(
    client: LLMClient,
    version: str,
    pairs: list[Pair],
    cache: JudgeCache,
    batch_size: int = BATCH_SIZE,
    prompts_dir: Path = PROMPTS_DIR,
) -> dict[str, int]:
    template = (prompts_dir / f"judge_{version}.md").read_text(encoding="utf-8")

    def key(pair: Pair) -> str:
        return pair_key(client.model, template, pair)

    todo = [p for p in pairs if key(p) not in cache.entries]
    stats = Counter(asked=len(todo))
    for start in range(0, len(todo), batch_size):
        batch = todo[start : start + batch_size]
        stats["calls"] += 1
        answers = _ask(client, template, batch)
        for i, pair in enumerate(batch):
            label = answers.get(i)
            if label is None:
                stats["retried"] += 1
                stats["calls"] += 1
                label = _ask(client, template, [pair]).get(0)
            stats[label or "failed"] += 1
            cache.put(
                {
                    "key": key(pair),
                    "model": client.model,
                    "prompt": version,
                    "query": pair.query,
                    "product_id": pair.product_id,
                    "label": label,
                }
            )
        print(f"{start + len(batch)}/{len(todo)} {dict(stats)}", flush=True)
    return dict(stats)


def confusion(truth: list[int], predicted: list[int], classes: int = 3) -> np.ndarray:
    """Rows are the human label, columns the LLM's."""
    matrix = np.zeros((classes, classes), dtype=int)
    for t, p in zip(truth, predicted, strict=True):
        matrix[t, p] += 1
    return matrix


def weighted_kappa(matrix: np.ndarray) -> float:
    """Cohen's kappa with linear weights: Exact vs Irrelevant counts as twice the miss of a neighbour."""
    n = matrix.sum()
    k = matrix.shape[0]
    weights = np.abs(np.subtract.outer(np.arange(k), np.arange(k))) / (k - 1)
    expected = np.outer(matrix.sum(axis=1), matrix.sum(axis=0)) / n
    return float(1 - (weights * matrix).sum() / (weights * expected).sum())


def fill_ndcg(
    ranked: dict[int, list[int]], qrels: dict, llm: dict[tuple[int, int], int], ids: list[int]
) -> dict:
    """NDCG@10 before and after adding LLM gains for unjudged products. Human labels are never changed."""
    before, after, judged_before, judged_after = [], [], 0, 0
    for qid in ids:
        human = qrels.get(qid, {})
        filled = {**human, **{pid: g for (q, pid), g in llm.items() if q == qid and pid not in human}}
        top = ranked[qid][:10]
        judged_before += sum(pid in human for pid in top)
        judged_after += sum(pid in filled for pid in top)
        before.append(ndcg_at(ranked[qid], human, 10))
        after.append(ndcg_at(ranked[qid], filled, 10))
    shown = sum(len(ranked[qid][:10]) for qid in ids)
    return {
        "ndcg_before": float(np.nanmean(before)),
        "ndcg_after": float(np.nanmean(after)),
        "judged_before": judged_before / shown,
        "judged_after": judged_after / shown,
        "scores_before": np.array(before),
        "scores_after": np.array(after),
    }
