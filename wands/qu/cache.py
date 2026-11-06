"""The committed answer cache. Evaluation, CI and the API read it, and only `wands qu run` writes it."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from wands.qu.schema import CACHE_MISS, OK, QueryUnderstanding

CACHE_PATH = Path("cache/qu_cache.jsonl")


def cache_key(model: str, prompt_text: str, query: str, candidates: list[str]) -> str:
    """A change to the model, prompt, query or candidates misses the cache, which forces a re-record."""
    payload = json.dumps([model, prompt_text, query, candidates], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class QUCache:
    def __init__(self, path: Path = CACHE_PATH):
        self.path = path
        self.entries: dict[str, dict] = {}
        if path.exists():
            with path.open(encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line)
                    self.entries[entry["key"]] = entry

    def get(self, key: str) -> dict | None:
        return self.entries.get(key)

    def put(self, entry: dict) -> None:
        self.entries[entry["key"]] = entry
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")


class QUTable:
    """Answers for one model + prompt, looked up by query text. This is the only QU the ranker sees."""

    def __init__(self, entries: dict[str, dict]):
        self.entries = entries

    @classmethod
    def from_cache(
        cls, cache: QUCache, model: str, template: str, candidates: dict[str, list[str]]
    ) -> QUTable:
        """Only answers recorded for exactly this prompt text and these candidates count.

        Looking up by key, not by version name, means editing a prompt in place can't serve stale
        answers: every query misses, the fallback count rises, and the gate fails until a re-record.
        """
        keys = {query: cache_key(model, template, query, c) for query, c in candidates.items()}
        return cls({query: cache.entries[key] for query, key in keys.items() if key in cache.entries})

    def lookup(self, query: str) -> tuple[QueryUnderstanding | None, str]:
        entry = self.entries.get(query)
        if entry is None:
            return None, CACHE_MISS
        if entry["reason"] != OK:
            return None, entry["reason"]
        return QueryUnderstanding.model_validate(entry["result"]), OK

    def fallback_counts(self, queries: list[str]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for query in queries:
            _, reason = self.lookup(query)
            if reason != OK:
                counts[reason] = counts.get(reason, 0) + 1
        return counts
