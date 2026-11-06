import json

import polars as pl
import pytest

from wands.catalog import Catalog
from wands.qu.attrs import product_attributes
from wands.qu.cache import QUCache, QUTable, cache_key
from wands.qu.client import LLMTimeout
from wands.qu.schema import (
    CACHE_MISS,
    INVALID_ATTR,
    INVALID_CLASS,
    MALFORMED_JSON,
    OK,
    TIMEOUT,
    QueryUnderstanding,
    parse_item,
)
from wands.qu.understand import Item, extract_answers, understand
from wands.retrievers.booster import AttributeMatcher, QUBooster, match_tokens
from wands.types import Hit, Query

CANDIDATES = ["Beds", "Sofas"]
GOOD = {
    "product_classes": ["Beds"],
    "attributes": {"color": "gray"},
    "intent": "category_browse",
    "confidence": "high",
}


def test_parse_accepts_a_good_answer():
    qu, reason = parse_item({**GOOD, "id": 3}, CANDIDATES)
    assert reason == OK and qu.product_classes == ["Beds"]


def test_parse_refuses_a_class_outside_the_candidates():
    assert parse_item({**GOOD, "product_classes": ["Lamps"]}, CANDIDATES) == (None, INVALID_CLASS)


def test_parse_refuses_an_unknown_attribute():
    assert parse_item({**GOOD, "attributes": {"brand": "moen"}}, CANDIDATES) == (None, INVALID_ATTR)


def test_parse_drops_null_attributes():
    qu, reason = parse_item({**GOOD, "attributes": {"color": None, "style": ""}}, CANDIDATES)
    assert reason == OK and qu.attributes == {}


def test_parse_refuses_a_wrong_shape():
    assert parse_item("Beds", CANDIDATES) == (None, MALFORMED_JSON)
    assert parse_item({**GOOD, "intent": "shopping"}, CANDIDATES) == (None, MALFORMED_JSON)
    assert parse_item({**GOOD, "extra": 1}, CANDIDATES) == (None, MALFORMED_JSON)
    assert parse_item({**GOOD, "product_classes": ["Beds"] * 4}, CANDIDATES) == (None, MALFORMED_JSON)


def test_extract_answers_takes_object_array_or_single():
    row = {"id": 1, "x": 1}
    assert extract_answers(json.dumps({"results": [row]})) == {1: row}
    assert extract_answers(json.dumps([row])) == {1: row}
    assert extract_answers(json.dumps(row)) == {1: row}


def test_cache_key_changes_with_every_input():
    base = cache_key("m", "p", "q", ["a"])
    assert base == cache_key("m", "p", "q", ["a"])
    for other in (
        ("n", "p", "q", ["a"]),
        ("m", "o", "q", ["a"]),
        ("m", "p", "r", ["a"]),
        ("m", "p", "q", ["b"]),
    ):
        assert cache_key(*other) != base


class Scripted:
    model = "scripted"

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def complete(self, prompt):
        self.prompts.append(prompt)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.fixture
def prompts(tmp_path):
    (tmp_path / "qu_v1.md").write_text("Q:\n{{items}}\n", encoding="utf-8")
    return tmp_path


ITEMS = [Item(1, "gray bed", CANDIDATES), Item(2, "sofa", CANDIDATES)]


def reply(*rows):
    return json.dumps({"results": list(rows)})


def test_understand_caches_a_batch_and_does_not_ask_twice(prompts, tmp_path):
    client = Scripted([reply({**GOOD, "id": 1}, {**GOOD, "id": 2})])
    cache = QUCache(tmp_path / "c.jsonl")
    stats = understand(client, "v1", ITEMS, cache, prompts_dir=prompts)
    assert stats["calls"] == 1 and stats[OK] == 2
    assert understand(client, "v1", ITEMS, cache, prompts_dir=prompts)["asked"] == 0
    reloaded = QUTable.from_cache(QUCache(tmp_path / "c.jsonl"), "scripted", "v1")
    assert reloaded.lookup("sofa")[1] == OK


def test_understand_reasks_a_failed_item_once_then_falls_back(prompts, tmp_path):
    bad = {**GOOD, "id": 2, "product_classes": ["Lamps"]}
    client = Scripted([reply({**GOOD, "id": 1}, bad), reply(bad)])
    cache = QUCache(tmp_path / "c.jsonl")
    stats = understand(client, "v1", ITEMS, cache, prompts_dir=prompts)
    assert stats["retried"] == 1 and stats[INVALID_CLASS] == 1 and len(client.prompts) == 2
    assert QUTable.from_cache(cache, "scripted", "v1").lookup("sofa") == (None, INVALID_CLASS)


def test_understand_recovers_on_the_reask(prompts, tmp_path):
    client = Scripted(["not json", reply({**GOOD, "id": 1}), reply({**GOOD, "id": 2})])
    cache = QUCache(tmp_path / "c.jsonl")
    stats = understand(client, "v1", ITEMS, cache, prompts_dir=prompts)
    assert stats == {"asked": 2, "calls": 3, "retried": 2, OK: 2}


def test_understand_records_a_timeout(prompts, tmp_path):
    client = Scripted([LLMTimeout("x"), LLMTimeout("x")])
    cache = QUCache(tmp_path / "c.jsonl")
    understand(client, "v1", ITEMS[:1], cache, prompts_dir=prompts)
    assert QUTable.from_cache(cache, "scripted", "v1").lookup("gray bed") == (None, TIMEOUT)


def test_table_counts_misses_and_fallbacks():
    table = QUTable({"a": {"reason": INVALID_CLASS, "result": None}})
    assert table.lookup("b") == (None, CACHE_MISS)
    assert table.fallback_counts(["a", "b"]) == {INVALID_CLASS: 1, CACHE_MISS: 1}


def catalog() -> Catalog:
    return Catalog.from_products(
        pl.DataFrame(
            {
                "product_id": [1, 2, 3],
                "product_name": ["alder frame", "grey linen bed", "oak table"],
                "product_class": ["Beds", "Beds", "Dining Tables"],
                "category_hierarchy": ["", "", ""],
                "product_description": ["", "", ""],
                "product_features": [
                    "basecolor : gray|primarymaterial : oak",
                    "color : white",
                    "color : gray",
                ],
            }
        )
    )


def test_product_attributes_pool_keys_into_groups():
    assert product_attributes("basecolor : gray|topcolor : white|color : gray|mirror : no") == {
        "color": ["gray", "white"]
    }


def test_match_tokens_unifies_spelling():
    assert match_tokens("Grey") == match_tokens("gray")


def test_attribute_matcher_reads_features_and_name():
    matcher = AttributeMatcher(catalog())
    assert matcher.matches(1, {"color": "grey"}) == 1  # in features, other spelling
    assert matcher.matches(2, {"color": "gray"}) == 1  # only in the name
    assert matcher.matches(3, {"material": "oak"}) == 1  # oak is in the name, not in material
    assert matcher.matches(2, {"color": "red", "material": "linen"}) == 1


class Fixed:
    name = "inner"

    def __init__(self, hits):
        self.hits = hits

    def search(self, q, k):
        return self.hits[:k]


INNER = [Hit(product_id=3, score=3.0), Hit(product_id=1, score=2.0), Hit(product_id=2, score=1.0)]


def booster(answer: dict | None, **kwargs) -> QUBooster:
    entries = {} if answer is None else {"q": {"reason": OK, "result": answer}}
    return QUBooster(Fixed(INNER), QUTable(entries), catalog(), **kwargs)


def answer(classes, attributes=None, confidence="high"):
    return QueryUnderstanding(
        product_classes=classes, attributes=attributes or {}, intent="category_browse", confidence=confidence
    ).model_dump()


def test_booster_lifts_a_matching_class_over_a_better_scored_product():
    found = booster(answer(["Beds"]), w_class=0.6, w_attr=0).search(Query(text="q"), 3)
    assert [h.product_id for h in found] == [1, 3, 2]
    assert found[0].debug["class_match"] == 1.0


def test_booster_scores_by_the_formula():
    found = booster(answer(["Beds"], {"color": "gray"}), w_class=0.2, w_attr=0.1).search(Query(text="q"), 3)
    by_id = {h.product_id: h.score for h in found}
    assert by_id[1] == pytest.approx(0.5 + 0.2 + 0.1)  # class and color match
    assert by_id[3] == pytest.approx(1.0 + 0.1)  # color only
    assert by_id[2] == pytest.approx(0.0 + 0.2 + 0.1)  # class, and gray via "grey" in the name


def test_booster_halves_low_confidence():
    found = booster(answer(["Beds"], confidence="low"), w_class=0.2, w_attr=0).search(Query(text="q"), 3)
    assert {h.product_id: h.score for h in found}[1] == pytest.approx(0.5 + 0.1)


def test_booster_leaves_a_cache_miss_alone():
    found = booster(None).search(Query(text="q"), 3)
    assert [h.product_id for h in found] == [3, 1, 2]
    assert found[0].debug["qu"] == CACHE_MISS


def test_filter_drops_other_classes():
    found = booster(answer(["Beds"]), mode="filter").search(Query(text="q"), 3)
    assert [h.product_id for h in found] == [1, 2]


def test_filter_that_would_empty_the_page_is_ignored():
    found = booster(answer(["Lamps"]), mode="filter").search(Query(text="q"), 3)
    assert [h.product_id for h in found] == [3, 1, 2]
