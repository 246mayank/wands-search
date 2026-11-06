"""LLM query rewriting, only for queries query understanding was unsure about. Offline and cached, like QU."""

from __future__ import annotations

import json
from pathlib import Path

from wands.qu.cache import QUTable
from wands.qu.client import LLMClient, LLMTimeout
from wands.qu.understand import PROMPTS_DIR
from wands.types import Hit, Query, Retriever

REWRITES_PATH = Path("cache/rewrites.jsonl")
BATCH_SIZE = 20


def eligible(table: QUTable, text: str) -> bool:
    """Chosen from QU's own output, never from labels, so a query's score can't pick it."""
    qu, _ = table.lookup(text)
    return qu is None or qu.confidence == "low" or not qu.product_classes


def rewrite_all(
    client: LLMClient, version: str, queries: list[str], prompts_dir: Path = PROMPTS_DIR
) -> dict[str, str]:
    """{query: rewrite}. A query the model doesn't answer stays as it is."""
    template = (prompts_dir / f"rewrite_{version}.md").read_text(encoding="utf-8")
    out = {}
    for start in range(0, len(queries), BATCH_SIZE):
        batch = queries[start : start + BATCH_SIZE]
        lines = [json.dumps({"id": i, "query": q}, ensure_ascii=False) for i, q in enumerate(batch)]
        try:
            data = json.loads(client.complete(template.replace("{{items}}", "\n".join(lines))))
        except (LLMTimeout, ValueError):
            continue
        rows = data.get("results", []) if isinstance(data, dict) else data
        answers = {r.get("id"): r.get("rewrite") for r in rows if isinstance(r, dict)}
        for i, query in enumerate(batch):
            answer = answers.get(i)
            if isinstance(answer, str) and answer.strip():
                out[query] = answer.strip()
    return out


def load_rewrites(path: Path = REWRITES_PATH) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return {row["query"]: row["rewrite"] for row in rows}


def save_rewrites(rewrites: dict[str, str], path: Path = REWRITES_PATH) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as f:
        for query, rewrite in rewrites.items():
            f.write(json.dumps({"query": query, "rewrite": rewrite}, ensure_ascii=False) + "\n")


class RewriteRetriever:
    """Retrieve with the rewrite. The QU stage above it still looks up the shopper's original words."""

    def __init__(self, inner: Retriever, rewrites: dict[str, str], name: str = "rewrite"):
        self.name = name
        self.inner = inner
        self.rewrites = rewrites

    def search(self, q: Query, k: int) -> list[Hit]:
        return self.inner.search(Query(text=self.rewrites.get(q.text, q.text), query_id=q.query_id), k)
