import polars as pl
import pytest

from wands.catalog import Catalog
from wands.eval.gate import check, make_report
from wands.rerank import CrossEncoderReranker, CrossEncoderScorer
from wands.types import Hit, Query


def catalog() -> Catalog:
    return Catalog.from_products(
        pl.DataFrame(
            {
                "product_id": [1, 2, 3, 4],
                "product_name": ["a", "b", "c", "d"],
                "product_class": ["X", "", "Y", "Z"],
                "category_hierarchy": ["", "", "", ""],
                "product_description": ["", "", "", ""],
                "product_features": ["color : gray", "", "", ""],
            }
        )
    )


class FakeModel:
    def __init__(self, scores: dict[str, float]):
        self.scores = scores
        self.pairs = []

    def predict(self, pairs, batch_size, show_progress_bar):
        self.pairs += pairs
        return [self.scores[text.split(" | ")[0]] for _, text in pairs]


class Fixed:
    name = "inner"

    def search(self, q, k):
        return [Hit(product_id=pid, score=10.0 - pid) for pid in (1, 2, 3, 4)][:k]


def reranker(mode: str = "score", depth: int = 3):
    scorer = CrossEncoderScorer(catalog(), cache_path=None)
    scorer.model = FakeModel({"a": 0.1, "b": 0.2, "c": 0.9})
    return CrossEncoderReranker(Fixed(), scorer, depth=depth, mode=mode), scorer


def test_text_joins_name_class_and_attributes():
    scorer = CrossEncoderScorer(catalog(), cache_path=None)
    assert scorer.text(1) == "a | X | gray"
    assert scorer.text(2) == "b"


def test_scorer_never_scores_a_pair_twice():
    scorer = CrossEncoderScorer(catalog(), cache_path=None)
    scorer.model = FakeModel({"a": 0.1, "b": 0.2})
    scorer.score("q", [1, 2])
    assert scorer.score("q", [2, 1]) == [0.2, 0.1]
    assert len(scorer.model.pairs) == 2


def test_score_mode_reorders_only_the_head():
    rr, _ = reranker()
    found = rr.search(Query(text="q"), 4)
    assert [h.product_id for h in found] == [3, 2, 1, 4]
    assert found[0].debug["rerank_score"] == 0.9


def test_rrf_mode_blends_both_orders():
    # Incoming 1,2,3 and cross-encoder 3,2,1: product 2 is second in both, so RRF ties 1 and 3 with it
    # below them: 1/61 + 1/63 beats 2/62.
    rr, _ = reranker("rrf")
    found = rr.search(Query(text="q"), 3)
    assert [h.product_id for h in found] == [1, 3, 2]


def test_unknown_mode_is_refused():
    with pytest.raises(ValueError):
        CrossEncoderReranker(Fixed(), None, mode="vote")


def report(ndcg: float, **fallbacks: int) -> dict:
    return make_report("configs/final.yaml", "dev", "abc", {"ndcg@10": ndcg}, fallbacks)


def test_gate_passes_on_equal_and_on_better():
    assert check(report(0.75), report(0.75))[0]
    assert check(report(0.80), report(0.75))[0]


def test_gate_tolerates_a_drop_within_the_limit():
    assert check(report(0.7485), report(0.75))[0]


def test_gate_fails_on_a_larger_drop():
    ok, lines = check(report(0.74), report(0.75))
    assert not ok and lines[0].startswith("FAIL")


def test_gate_fails_when_a_fallback_count_rises():
    ok, lines = check(report(0.75, INVALID_CLASS=3), report(0.75, INVALID_CLASS=2))
    assert not ok and any("INVALID_CLASS" in line for line in lines)


def test_gate_fails_on_a_new_fallback_reason():
    assert not check(report(0.75, CACHE_MISS=1), report(0.75))[0]


def test_gate_allows_fewer_fallbacks():
    assert check(report(0.75), report(0.75, TIMEOUT=2))[0]
