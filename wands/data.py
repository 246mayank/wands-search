"""Fetch WANDS pinned to one upstream commit, verify it, and load it with Polars."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import httpx
import polars as pl

WANDS_COMMIT = "3b74dcf4ba29ab8ff3e6a50b5b09fc627cb882b5"
BASE_URL = f"https://raw.githubusercontent.com/wayfair/WANDS/{WANDS_COMMIT}/dataset"

# SHA-256 of each file at WANDS_COMMIT. See docs/DATA_NOTES.md for how these were cross-checked.
CHECKSUMS = {
    "label.csv": "c11fe81ad62f17f56f316b0ec9630ebe8fbe1393578cb0ca4f05c17253a180ef",
    "product.csv": "d993926254572e6eba96c8fd87cc549a17fb91ad3748308036eee4cf92b10ac6",
    "query.csv": "63b61660560fecc33ec490804c7e2b81402ee3e7c31a9cbb5e03736639f68e95",
}

DATA_DIR = Path("data")
TEXT_COLUMNS = [
    "product_name",
    "product_class",
    "category_hierarchy",
    "product_description",
    "product_features",
]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    """The id git gives this file, to compare with the blob ids GitHub lists."""
    digest = hashlib.sha1()
    digest.update(f"blob {path.stat().st_size}\0".encode())
    digest.update(path.read_bytes())
    return digest.hexdigest()


def fetch(data_dir: Path = DATA_DIR) -> dict[str, str]:
    """Download any file that is missing or wrong, and fail if a checksum doesn't match."""
    data_dir.mkdir(parents=True, exist_ok=True)
    seen = {}
    for name, expected in CHECKSUMS.items():
        path = data_dir / name
        if not (path.exists() and sha256_of(path) == expected):
            tmp = path.with_suffix(".part")
            with httpx.stream("GET", f"{BASE_URL}/{name}", timeout=120, follow_redirects=True) as r:
                r.raise_for_status()
                with tmp.open("wb") as f:
                    for chunk in r.iter_bytes():
                        f.write(chunk)
            tmp.replace(path)
        seen[name] = sha256_of(path)
        if seen[name] != expected:
            raise ValueError(f"{name}: sha256 {seen[name]} does not match pinned {expected}")
    return seen


def _read(path: Path, quoted: bool) -> pl.DataFrame:
    quote_char = '"' if quoted else None
    return pl.read_csv(path, separator="\t", quote_char=quote_char, infer_schema_length=10_000)


@dataclass(frozen=True)
class Wands:
    products: pl.DataFrame
    queries: pl.DataFrame
    labels: pl.DataFrame


def load(data_dir: Path = DATA_DIR) -> Wands:
    # All tab-separated. product.csv has raw inch marks inside fields, so quoting stays off there;
    # query.csv is CSV-quoted (3 queries contain an escaped inch mark), so it needs quoting on.
    products = (
        _read(data_dir / "product.csv", quoted=False)
        .rename({"category hierarchy": "category_hierarchy"})
        .with_columns(pl.col(TEXT_COLUMNS).fill_null(""))
    )
    queries = _read(data_dir / "query.csv", quoted=True).with_columns(pl.col("query_class").fill_null(""))
    labels = _read(data_dir / "label.csv", quoted=False)
    return Wands(products=products, queries=queries, labels=labels)
