"""Tokenising, and turning a product row into the text each retriever sees."""

from __future__ import annotations

import re
import unicodedata

import Stemmer

_STEMMER = Stemmer.Stemmer("english")
_TOKEN = re.compile(r"[a-z0-9]+")
_NUMBER = re.compile(r"[\d.,/ -]+")
_FLAGS = {"yes", "no", ""}


def fold(text: str) -> str:
    """Lowercase and drop accents, so "Décor" and "decor" tokenise alike."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower()


def tokenize(text: str, stem: bool = True) -> list[str]:
    """Split on non-alphanumerics, with no stopword removal: "with" and "for" are rare in product names."""
    tokens = _TOKEN.findall(fold(text))
    return _STEMMER.stemWords(tokens) if stem else tokens


def parse_features(raw: str) -> list[tuple[str, str]]:
    """`color : white|overallheight:30` becomes [("color", "white"), ("overallheight", "30")]."""
    pairs = []
    for item in raw.split("|"):
        key, sep, value = item.partition(":")
        key, value = key.strip().lower(), value.strip().lower()
        if sep and key:
            pairs.append((key, value))
    return pairs


def is_descriptive(value: str) -> bool:
    return value not in _FLAGS and not _NUMBER.fullmatch(value)


def feature_text(raw: str) -> str:
    """Values that describe the product. Dropped: keys, numbers, and yes/no flags.

    A key isn't searchable text: `mirror : no` must not match the query "mirror".
    """
    return " ".join(value for _, value in parse_features(raw) if is_descriptive(value))


def split_classes(raw: str) -> tuple[str, ...]:
    """product_class is `|`-delimited on about 5% of products."""
    return tuple(c.strip() for c in raw.split("|") if c.strip())
