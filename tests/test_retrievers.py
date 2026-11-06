import numpy as np
import polars as pl
import pytest

from wands.catalog import Catalog
from wands.retrievers.bm25f import BM25FRetriever, IndexStore, top_k
from wands.retrievers.dense import DenseRetriever
from wands.retrievers.fusion import FusionRetriever, minmax, order, rrf_scores, weighted_scores
from wands.text import feature_text, fold, parse_features, split_classes, tokenize
from wands.types import Hit, Query


def catalog() -> Catalog:
    products = pl.DataFrame(
        {
            "product_id": [10, 11, 12, 13],
            "product_name": ["oak dining table", "red velvet sofa", "blue area rug", "sofa table"],
            "product_class": ["Dining Tables", "Sofas", "Area Rugs", "Sofa & Console Tables|Accent Tables"],
            "category_hierarchy": ["Furniture / Tables", "Furniture / Sofas", "Rugs", "Furniture / Tables"],
            "product_description": ["a solid table", "a soft sofa", "", "narrow"],
            "product_features": [
                "color : brown|mirror : no",
                "color : red",
                "color : blue",
                "overallheight:30",
            ],
        }
    )
    return Catalog.from_products(products)


def hits(*ids: int) -> list[Hit]:
    return [Hit(product_id=pid, score=float(len(ids) - i)) for i, pid in enumerate(ids)]


def test_fold_drops_accents_and_case():
    assert fold("Wall Décor") == "wall decor"


def test_tokenize_splits_on_punctuation_and_stems():
    assert tokenize('Moen® Hooks, 48"', stem=False) == ["moen", "hooks", "48"]
    assert tokenize("chairs") == tokenize("chair")


def test_tokenize_keeps_stopwords():
    assert "with" in tokenize("table with drawers", stem=False)


def test_parse_features_handles_both_separators_and_empty_keys():
    assert parse_features("Color : White|overallheight:30| : no") == [
        ("color", "white"),
        ("overallheight", "30"),
    ]


def test_feature_text_keeps_descriptive_values_only():
    assert (
        feature_text("color : white|mirror : no|drawers : yes|overallheight:30|style : rustic")
        == "white rustic"
    )


def test_split_classes():
    assert split_classes("Sofas | Beds") == ("Sofas", "Beds")
    assert split_classes("") == ()


def test_catalog_fields_line_up():
    cat = catalog()
    assert len(cat) == 4
    assert cat.classes[3] == ("Sofa & Console Tables", "Accent Tables")
    assert cat.fields["features"][0] == "brown"
    assert "oak dining table" in cat.embedding_text()[0]


def test_top_k_is_ordered_and_breaks_ties_by_position():
    assert top_k(np.array([0.2, 0.9, 0.9, 0.1]), 3).tolist() == [1, 2, 0]


def test_bm25f_finds_the_named_product_and_skips_non_matches():
    bm25 = BM25FRetriever(IndexStore(catalog()), {"name": 1.0})
    found = bm25.search(Query(text="velvet sofa"), 10)
    assert found[0].product_id == 11
    assert 12 not in [hit.product_id for hit in found]


def test_bm25f_field_weight_changes_the_order():
    query = Query(text="sofa")
    store = IndexStore(catalog())
    by_name = BM25FRetriever(store, {"name": 1.0, "class": 0}).search(query, 2)
    by_class = BM25FRetriever(store, {"name": 0, "class": 1.0}).search(query, 2)
    assert {hit.product_id for hit in by_name} == {11, 13}
    assert {hit.product_id for hit in by_class} == {11, 13}
    both = BM25FRetriever(store, {"name": 1.0, "class": 1.0}).score_all("sofa")
    alone = BM25FRetriever(store, {"name": 1.0}).score_all("sofa")
    assert both[1] > alone[1]


def test_bm25f_query_with_unknown_words_returns_nothing():
    assert BM25FRetriever(IndexStore(catalog()), {"name": 1.0}).search(Query(text="zzzz"), 5) == []


def test_rrf_by_hand():
    scores = rrf_scores([hits(1, 2), hits(2, 3)], k=60)
    assert scores[2] == pytest.approx(1 / 62 + 1 / 61)
    assert scores[1] == pytest.approx(1 / 61)
    assert scores[3] == pytest.approx(1 / 62)
    assert [pid for pid, _ in order(scores, 3)] == [2, 1, 3]


def test_rrf_ignores_score_scale():
    big = [Hit(product_id=1, score=1000.0), Hit(product_id=2, score=999.0)]
    small = [Hit(product_id=1, score=0.2), Hit(product_id=2, score=0.1)]
    assert rrf_scores([big], 60) == rrf_scores([small], 60)


def test_minmax_scales_to_unit_range():
    scores = minmax(
        [Hit(product_id=1, score=4.0), Hit(product_id=2, score=2.0), Hit(product_id=3, score=3.0)]
    )
    assert scores == {1: 1.0, 2: 0.0, 3: 0.5}


def test_minmax_of_equal_scores_does_not_divide_by_zero():
    assert minmax([Hit(product_id=1, score=2.0), Hit(product_id=2, score=2.0)]) == {1: 1.0, 2: 1.0}


def test_weighted_sum_counts_a_missing_product_as_zero():
    first = [Hit(product_id=1, score=2.0), Hit(product_id=2, score=1.0)]
    second = [Hit(product_id=2, score=5.0), Hit(product_id=3, score=1.0)]
    scores = weighted_scores([first, second], alpha=0.25)
    assert scores == {1: 0.25, 2: 0.75, 3: 0.0}


class Fixed:
    def __init__(self, name: str, found: list[Hit]):
        self.name = name
        self.found = found

    def search(self, q: Query, k: int) -> list[Hit]:
        return self.found[:k]


def test_fusion_records_each_retrievers_rank():
    fused = FusionRetriever([Fixed("bm25", hits(1, 2)), Fixed("dense", hits(2, 3))]).search(
        Query(text="x"), 2
    )
    assert fused[0].product_id == 2
    assert fused[0].debug == {"bm25_rank": 2, "dense_rank": 1}


def test_weighted_fusion_needs_two_lists():
    with pytest.raises(ValueError):
        FusionRetriever([Fixed("a", [])], method="weighted")


class FakeEncoder:
    def encode(self, text: str) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)


def test_dense_ranks_by_dot_product():
    vectors = np.array([[0.0, 1.0], [1.0, 0.0], [0.6, 0.8]], dtype=np.float32)
    found = DenseRetriever(FakeEncoder(), vectors, np.array([5, 6, 7])).search(Query(text="x"), 2)
    assert [hit.product_id for hit in found] == [6, 7]
