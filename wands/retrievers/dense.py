"""Dense retrieval: brute-force dot product over normalised embeddings. 43K x 384 needs no ANN index."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from wands.retrievers.bm25f import top_k
from wands.types import Hit, Query

MODEL_NAME = "BAAI/bge-small-en-v1.5"
# From the model card: queries (not passages) take this prefix for short-query-to-passage retrieval.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

EMBEDDINGS_PATH = Path("data/embeddings.npy")
QUERY_CACHE_PATH = Path("cache/query_embeddings.npz")


def load_model(model_name: str = MODEL_NAME):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device="cpu")


def encode_products(texts: list[str], model_name: str = MODEL_NAME, batch_size: int = 128) -> np.ndarray:
    model = load_model(model_name)
    vectors = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
    return vectors.astype(np.float32)


class QueryEncoder:
    """Query vectors from the committed cache, so evaluation needs no model. Unseen text loads the model."""

    def __init__(self, cache_path: Path = QUERY_CACHE_PATH, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.model = None
        self.cached: dict[str, np.ndarray] = {}
        if cache_path.exists():
            stored = np.load(cache_path, allow_pickle=False)
            self.cached = dict(zip(stored["texts"].tolist(), stored["vectors"], strict=True))

    def encode(self, text: str) -> np.ndarray:
        if text in self.cached:
            return self.cached[text]
        if self.model is None:
            self.model = load_model(self.model_name)
        return self.model.encode(QUERY_PREFIX + text, normalize_embeddings=True).astype(np.float32)


def write_query_cache(texts: list[str], path: Path = QUERY_CACHE_PATH, model_name: str = MODEL_NAME) -> None:
    model = load_model(model_name)
    prefixed = [QUERY_PREFIX + t for t in texts]
    vectors = model.encode(prefixed, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, texts=np.array(texts), vectors=vectors.astype(np.float32))


class DenseRetriever:
    def __init__(
        self, encoder: QueryEncoder, embeddings: np.ndarray, product_ids: np.ndarray, name: str = "dense"
    ):
        self.name = name
        self.encoder = encoder
        self.embeddings = embeddings
        self.product_ids = product_ids

    def search(self, q: Query, k: int) -> list[Hit]:
        scores = self.embeddings @ self.encoder.encode(q.text)
        return [Hit(product_id=int(self.product_ids[i]), score=float(scores[i])) for i in top_k(scores, k)]
