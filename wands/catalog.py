"""The product catalogue in the shape the retrievers need: one text per field, per product."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from wands.qu.attrs import product_attributes
from wands.text import feature_text, split_classes

FIELDS = ["name", "class", "hierarchy", "features", "description"]


@dataclass
class Catalog:
    product_ids: np.ndarray
    fields: dict[str, list[str]]  # field -> text per product, aligned with product_ids
    classes: list[tuple[str, ...]]
    raw_features: list[str]

    def __len__(self) -> int:
        return len(self.product_ids)

    @classmethod
    def from_products(cls, products: pl.DataFrame) -> Catalog:
        features = products["product_features"].to_list()
        classes = [split_classes(c) for c in products["product_class"].to_list()]
        fields = {
            "name": products["product_name"].to_list(),
            "class": [" ".join(c) for c in classes],
            "hierarchy": products["category_hierarchy"].to_list(),
            "features": [feature_text(f) for f in features],
            "features_raw": features,  # for the ablation against the cleaned text above
            "description": products["product_description"].to_list(),
        }
        return cls(products["product_id"].to_numpy(), fields, classes, features)

    def all_text(self) -> list[str]:
        return [" ".join(parts) for parts in zip(*(self.fields[f] for f in FIELDS), strict=True)]

    def attribute_text(self, i: int, top_values: int = 12) -> str:
        """Shopper-facing attribute values of one product, such as `gray velvet modern`."""
        groups = product_attributes(self.raw_features[i]).values()
        return " ".join([value for values in groups for value in values][:top_values])

    def embedding_text(self) -> list[str]:
        """`name | class | hierarchy | attributes`: the text embedded for dense retrieval."""
        rows = zip(self.fields["name"], self.fields["class"], self.fields["hierarchy"], strict=True)
        return [
            " | ".join(part for part in (*row, self.attribute_text(i)) if part) for i, row in enumerate(rows)
        ]
