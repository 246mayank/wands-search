"""Run the LLM over queries in batches: validate each answer, re-ask once, otherwise record why it failed."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from wands.qu.cache import QUCache, cache_key
from wands.qu.client import LLMClient, LLMTimeout
from wands.qu.schema import MALFORMED_JSON, OK, TIMEOUT, parse_item

PROMPTS_DIR = Path("prompts")
BATCH_SIZE = 20


@dataclass(frozen=True)
class Item:
    id: int
    query: str
    candidates: list[str]


def prompt_text(version: str, prompts_dir: Path = PROMPTS_DIR) -> str:
    return (prompts_dir / f"qu_{version}.md").read_text(encoding="utf-8")


def render(template: str, items: list[Item]) -> str:
    lines = [
        json.dumps({"id": i.id, "query": i.query, "candidates": i.candidates}, ensure_ascii=False)
        for i in items
    ]
    return template.replace("{{items}}", "\n".join(lines))


def extract_answers(text: str) -> dict[int, object]:
    """{id: answer} from a reply: `{"results": [...]}`, a bare array, or one answer alone."""
    data = json.loads(text)
    if isinstance(data, dict):
        data = data.get("results", [data])
    if not isinstance(data, list):
        raise ValueError("reply is neither an object nor an array")
    return {row["id"]: row for row in data if isinstance(row, dict) and isinstance(row.get("id"), int)}


def ask(client: LLMClient, template: str, items: list[Item]) -> dict[int, tuple[dict | None, str]]:
    """One call. Returns {id: (validated answer, reason)} for every item asked."""
    try:
        reply = client.complete(render(template, items))
    except LLMTimeout:
        return {item.id: (None, TIMEOUT) for item in items}
    try:
        answers = extract_answers(reply)
    except (ValueError, TypeError, KeyError):
        return {item.id: (None, MALFORMED_JSON) for item in items}
    out: dict[int, tuple[dict | None, str]] = {}
    for item in items:
        if item.id not in answers:
            out[item.id] = (None, MALFORMED_JSON)
            continue
        qu, reason = parse_item(answers[item.id], item.candidates)
        out[item.id] = (qu.model_dump() if qu else None, reason)
    return out


def understand(
    client: LLMClient,
    version: str,
    items: list[Item],
    cache: QUCache,
    batch_size: int = BATCH_SIZE,
    prompts_dir: Path = PROMPTS_DIR,
) -> dict[str, int]:
    """Answer every item not already cached. Returns counts of calls and outcomes."""
    template = prompt_text(version, prompts_dir)
    todo = [i for i in items if cache.get(cache_key(client.model, template, i.query, i.candidates)) is None]
    stats = {"asked": len(todo), "calls": 0, "retried": 0, OK: 0}

    def save(item: Item, result: dict | None, reason: str) -> None:
        stats[reason] = stats.get(reason, 0) + 1
        cache.put(
            {
                "key": cache_key(client.model, template, item.query, item.candidates),
                "model": client.model,
                "prompt": version,
                "query": item.query,
                "result": result,
                "reason": reason,
            }
        )

    for start in range(0, len(todo), batch_size):
        batch = todo[start : start + batch_size]
        stats["calls"] += 1
        results = ask(client, template, batch)
        for item in batch:
            result, reason = results[item.id]
            if reason != OK:
                stats["retried"] += 1
                stats["calls"] += 1
                result, reason = ask(client, template, [item])[item.id]
            save(item, result, reason)
        print(f"{start + len(batch)}/{len(todo)} {stats}", flush=True)
    return stats
