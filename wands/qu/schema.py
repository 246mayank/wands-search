"""What the LLM must return for one query, and the check every answer goes through."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from wands.qu.attrs import AttrKey

Intent = Literal["specific_item", "category_browse", "style_or_room", "unclear"]
Confidence = Literal["high", "medium", "low"]

OK = "OK"
MALFORMED_JSON = "MALFORMED_JSON"
INVALID_CLASS = "INVALID_CLASS"
INVALID_ATTR = "INVALID_ATTR"
TIMEOUT = "TIMEOUT"
CACHE_MISS = "CACHE_MISS"
FALLBACKS = (MALFORMED_JSON, INVALID_CLASS, INVALID_ATTR, TIMEOUT)


class QueryUnderstanding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_classes: list[str] = Field(max_length=3)
    attributes: dict[AttrKey, str]
    intent: Intent
    confidence: Confidence


def parse_item(raw: Any, candidates: list[str]) -> tuple[QueryUnderstanding | None, str]:
    """Validate one answer. A class outside the query's candidate list is refused, never trusted."""
    if not isinstance(raw, dict):
        return None, MALFORMED_JSON
    item = {key: value for key, value in raw.items() if key != "id"}
    if isinstance(item.get("attributes"), dict):
        # "color": null means the same as leaving color out, which the prompt asks for.
        item["attributes"] = {k: v for k, v in item["attributes"].items() if v not in (None, "")}
    try:
        qu = QueryUnderstanding.model_validate(item)
    except ValidationError as error:
        about_attrs = any(e["loc"] and e["loc"][0] == "attributes" for e in error.errors())
        return None, INVALID_ATTR if about_attrs else MALFORMED_JSON
    if any(c not in candidates for c in qu.product_classes):
        return None, INVALID_CLASS
    return qu, OK
