"""Use cached query understanding in ranking: a soft boost, or a hard class filter for comparison."""

from __future__ import annotations

from wands.catalog import Catalog
from wands.qu.attrs import product_attributes
from wands.qu.cache import QUTable
from wands.qu.schema import OK, QueryUnderstanding
from wands.retrievers.fusion import minmax
from wands.text import tokenize
from wands.types import Hit, Query, Retriever

LOW_CONFIDENCE_FACTOR = 0.5
_SPELLINGS = {"grey": "gray"}


def match_tokens(text: str) -> set[str]:
    return {_SPELLINGS.get(token, token) for token in tokenize(text)}


class AttributeMatcher:
    """Does a product state an attribute value? It may say so in that attribute's features or in its name."""

    def __init__(self, catalog: Catalog):
        self.positions = {int(pid): i for i, pid in enumerate(catalog.product_ids)}
        self.catalog = catalog
        self._tokens: dict[int, dict[str, set[str]]] = {}

    def _product_tokens(self, product_id: int) -> dict[str, set[str]]:
        if product_id not in self._tokens:
            i = self.positions[product_id]
            groups = product_attributes(self.catalog.raw_features[i])
            tokens = {group: match_tokens(" ".join(values)) for group, values in groups.items()}
            tokens["__name__"] = match_tokens(self.catalog.fields["name"][i])
            self._tokens[product_id] = tokens
        return self._tokens[product_id]

    def matches(self, product_id: int, attributes: dict[str, str]) -> int:
        product = self._product_tokens(product_id)
        count = 0
        for group, value in attributes.items():
            wanted = match_tokens(value)
            if wanted and wanted <= (product.get(group, set()) | product["__name__"]):
                count += 1
        return count


class QUBooster:
    def __init__(
        self,
        inner: Retriever,
        table: QUTable,
        catalog: Catalog,
        w_class: float = 0.2,
        w_attr: float = 0.1,
        mode: str = "soft",
        depth: int = 200,
        name: str = "qu",
    ):
        if mode not in ("soft", "filter"):
            raise ValueError(f"unknown booster mode {mode!r}")
        self.name = name
        self.inner = inner
        self.table = table
        self.w_class = w_class
        self.w_attr = w_attr
        self.mode = mode
        self.depth = depth
        self.classes = {int(pid): set(c) for pid, c in zip(catalog.product_ids, catalog.classes, strict=True)}
        self.matcher = AttributeMatcher(catalog)

    def search(self, q: Query, k: int) -> list[Hit]:
        return self.rescore(q, self.inner.search(q, max(k, self.depth)))[:k]

    def rescore(self, q: Query, hits: list[Hit]) -> list[Hit]:
        qu, reason = self.table.lookup(q.text)
        if qu is None:
            return [h.model_copy(update={"debug": {**h.debug, "qu": reason}}) for h in hits]
        return self.filter(hits, qu) if self.mode == "filter" else self.boost(hits, qu)

    def filter(self, hits: list[Hit], qu: QueryUnderstanding) -> list[Hit]:
        """Keep products of a predicted class. If that leaves nothing, the prediction is ignored."""
        wanted = set(qu.product_classes)
        kept = [h for h in hits if self.classes[h.product_id] & wanted]
        return [h.model_copy(update={"debug": {**h.debug, "qu": OK}}) for h in kept or hits]

    def boost(self, hits: list[Hit], qu: QueryUnderstanding) -> list[Hit]:
        """score = fused_norm + w_class x class match + w_attr x share of attributes matched."""
        norm = minmax(hits)
        scale = LOW_CONFIDENCE_FACTOR if qu.confidence == "low" else 1.0
        wanted = set(qu.product_classes)
        boosted = []
        for hit in hits:
            class_match = 1.0 if self.classes[hit.product_id] & wanted else 0.0
            attr_share = 0.0
            if qu.attributes:
                attr_share = self.matcher.matches(hit.product_id, qu.attributes) / len(qu.attributes)
            bonus = scale * (self.w_class * class_match + self.w_attr * attr_share)
            debug = {
                **hit.debug,
                "qu": OK,
                "class_match": class_match,
                "attr_share": attr_share,
                "boost": bonus,
            }
            boosted.append(Hit(product_id=hit.product_id, score=norm[hit.product_id] + bonus, debug=debug))
        return sorted(boosted, key=lambda h: (-h.score, h.product_id))
