"""The fixed attribute vocabulary: shopper-facing groups, each pooling several `product_features` keys."""

from __future__ import annotations

from collections import defaultdict
from typing import Literal

from wands.text import is_descriptive, parse_features

AttrKey = Literal["color", "material", "style", "shape", "pattern", "room", "size", "theme"]

GROUP_KEYS: dict[str, tuple[str, ...]] = {
    "color": (
        "color",
        "basecolor",
        "topcolor",
        "upholsterycolor",
        "framecolor",
        "primarycolor",
        "maincolor",
        "seatcolor",
        "cushioncolor",
        "legcolor",
        "backcolor",
        "shadecolor",
        "dswoodtone",
    ),
    "material": (
        "primarymaterial",
        "material",
        "framematerial",
        "upholsterymaterial",
        "basematerial",
        "topmaterial",
        "legmaterial",
    ),
    "style": ("style", "dsprimaryproductstyle", "dssecondaryproductstyle"),
    "shape": ("shape", "topshape", "rugshape", "tableshape", "headboardshape", "sinkshape", "shadeshape"),
    "pattern": ("pattern",),
    "room": ("roomuse", "location", "installationlocation"),
    "size": ("size", "mattresssize", "rugsize", "bedsize", "beddingsize"),
    "theme": ("theme",),
}

KEY_TO_GROUP = {key: group for group, keys in GROUP_KEYS.items() for key in keys}


def product_attributes(raw_features: str) -> dict[str, list[str]]:
    """{group: values} for one product, keeping feature order and dropping flags and bare numbers."""
    groups: dict[str, list[str]] = defaultdict(list)
    for key, value in parse_features(raw_features):
        group = KEY_TO_GROUP.get(key)
        if group and is_descriptive(value) and value not in groups[group]:
            groups[group].append(value)
    return dict(groups)
