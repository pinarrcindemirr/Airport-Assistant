from __future__ import annotations

from dataclasses import dataclass

import faiss
import numpy as np

from backend.utils.config import CONFIDENCE_THRESHOLD, TOP_K
from backend.kb.kb import get_record_by_id
from backend.text.embeddings import TextEmbedder


@dataclass
class Candidate:
    """One retrieval hit: the record plus its similarity score."""
    record_id: str
    score: float
    record: dict


@dataclass
class Answer:
    """Outcome of answer."""
    answered: bool          # False -> hand over to a human
    candidates: list[Candidate]
    top_score: float

    @property
    def record(self) -> dict | None:
        return self.candidates[0].record if (self.answered and self.candidates) else None


class TextRetriever:
    def __init__(self, embedder: TextEmbedder, include_keywords: bool = True):
        self.embedder = embedder
        self.include_keywords = include_keywords
        self.ids, self.matrix = embedder.embed_kb(include_keywords=include_keywords)

        self.dimension = self.matrix.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(np.ascontiguousarray(self.matrix, dtype=np.float32))

    def search(self, query: str, top_k: int = TOP_K) -> list[Candidate]:
        """Return the top_k most similar records, highest score first."""
        q = self.embedder.embed_query(query)
        q = np.ascontiguousarray(q, dtype=np.float32).reshape(1, -1)

        sims, indices = self.index.search(q, top_k)  # both shape (1, top_k)
        sims, indices = sims[0], indices[0]

        return [
            Candidate(record_id=self.ids[i], score=float(sims[rank]),
                      record=get_record_by_id(self.ids[i]))
            for rank, i in enumerate(indices)
            if i != -1  # FAISS pads with -1 if top_k > number of indexed vectors
        ]

    def answer(self, query: str, top_k: int = TOP_K) -> Answer:
        candidates = self.search(query, top_k=top_k)
        top_score = candidates[0].score if candidates else 0.0
        return Answer(
            answered=top_score >= CONFIDENCE_THRESHOLD,
            candidates=candidates,
            top_score=top_score,
        )