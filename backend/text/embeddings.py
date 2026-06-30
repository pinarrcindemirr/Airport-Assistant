from __future__ import annotations

import numpy as np

from backend.kb.kb import get_all_records, build_text_for_embedding

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class TextEmbedder:
    """Encodes text into L2-normalised vectors and builds the KB matrix."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, model=None):
        if model is not None:
            # Injected model (e.g. a test stand-in). Must expose .encode(list[str]) -> ndarray.
            self.model = model
            self.model_name = getattr(model, "name", "injected")
        else:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a list of strings into an (n, d) array of unit vectors."""
        vectors = np.asarray(self.model.encode(texts), dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors[None, :]
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # guard against a zero vector
        return vectors / norms

    def embed_query(self, query: str) -> np.ndarray:
        """Encode a single query into one unit vector of shape (d,)."""
        return self.encode([query])[0]

    def embed_kb(self, include_keywords: bool = False) -> tuple[list[str], np.ndarray]:

        records = get_all_records()
        ids = [r["id"] for r in records]
        texts = [build_text_for_embedding(r, include_keywords=include_keywords) for r in records]
        matrix = self.encode(texts)
        return ids, matrix
