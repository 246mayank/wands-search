"""A short list of plausible classes per query: the LLM may only choose from it.

The list is the union of classes close to the query in embedding space and classes of the top lexical
hits. Sorted by name, so its order doesn't hint at an answer.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from wands.retrievers.bm25f import BM25FRetriever, top_k
from wands.retrievers.dense import QueryEncoder, load_model
from wands.types import Query

CANDIDATES_PATH = Path("cache/qu_candidates.json")
CLASS_VECTORS_PATH = Path("cache/class_embeddings.npz")
EMBED_TOP = 15
LEXICAL_TOP = 15
LEXICAL_HITS = 50


def write_class_vectors(classes: list[str], path: Path = CLASS_VECTORS_PATH) -> None:
    vectors = load_model().encode(classes, normalize_embeddings=True, show_progress_bar=False)
    np.savez_compressed(path, classes=np.array(classes), vectors=vectors.astype(np.float32))


def load_class_vectors(path: Path = CLASS_VECTORS_PATH) -> tuple[list[str], np.ndarray]:
    stored = np.load(path, allow_pickle=False)
    return stored["classes"].tolist(), stored["vectors"]


def candidates_for(
    text: str,
    bm25: BM25FRetriever,
    encoder: QueryEncoder,
    class_names: list[str],
    class_vectors: np.ndarray,
    product_classes: dict[int, tuple[str, ...]],
    embed_top: int = EMBED_TOP,
    lexical_top: int = LEXICAL_TOP,
) -> list[str]:
    near = class_vectors @ encoder.encode(text)
    chosen = {class_names[i] for i in top_k(near, embed_top)}
    counts: Counter[str] = Counter()
    for hit in bm25.search(Query(text=text), LEXICAL_HITS):
        counts.update(product_classes[hit.product_id])
    chosen.update(name for name, _ in counts.most_common(lexical_top))
    return sorted(chosen)


def write_candidates(table: dict[str, list[str]], path: Path = CANDIDATES_PATH) -> None:
    path.write_text(json.dumps(table, ensure_ascii=False, indent=0) + "\n", encoding="utf-8", newline="\n")


def load_candidates(path: Path = CANDIDATES_PATH) -> dict[str, list[str]]:
    return json.loads(path.read_text(encoding="utf-8"))
