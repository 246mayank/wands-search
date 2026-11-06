import polars as pl
import pytest

from wands.catalog import Catalog
from wands.retrievers.popularity import PopularityPrior
from wands.types import Hit, Query


def catalog() -> Catalog:
    return Catalog.from_products(
        pl.DataFrame(
            {
                "product_id": [1, 2, 3],
                "product_name": ["a", "b", "c"],
                "product_class": ["", "", ""],
                "category_hierarchy": ["", "", ""],
                "product_description": ["", "", ""],
                "product_features": ["", "", ""],
                "rating_count": [None, 0.0, 99.0],
                "average_rating": [None, 4.0, 5.0],
                "review_count": [None, 0.0, 1.0],
            }
        )
    )


class Fixed:
    name = "inner"

    def search(self, q, k):
        return [Hit(product_id=1, score=2.0), Hit(product_id=2, score=1.0), Hit(product_id=3, score=0.0)][:k]


def test_catalog_keeps_missing_ratings_as_nan():
    cat = catalog()
    assert cat.numeric["average_rating"][0] != cat.numeric["average_rating"][0]
    assert cat.numeric["log_rating_count"][2] == pytest.approx(4.60517, abs=1e-4)


def test_popularity_lifts_a_rated_product():
    found = PopularityPrior(Fixed(), catalog(), weight=2.0).search(Query(text="q"), 3)
    assert [h.product_id for h in found] == [3, 1, 2]


def test_zero_weight_changes_nothing():
    found = PopularityPrior(Fixed(), catalog(), weight=0.0).search(Query(text="q"), 3)
    assert [h.product_id for h in found] == [1, 2, 3]
