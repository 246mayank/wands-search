"""The types every pipeline stage shares."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel


class Query(BaseModel):
    text: str
    query_id: int | None = None


class Hit(BaseModel):
    product_id: int
    score: float
    debug: dict[str, float | str] = {}


class Retriever(Protocol):
    name: str

    def search(self, q: Query, k: int) -> list[Hit]: ...
