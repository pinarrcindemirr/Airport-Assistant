from __future__ import annotations

from dataclasses import dataclass

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
    """Outcome of answer(): either a confident hit or an abstention."""
    answered: bool          # False -> hand over to a human
    candidates: list[Candidate]
    top_score: float

    @property
    def record(self) -> dict | None:
        return self.candidates[0].record if (self.answered and self.candidates) else None


class TextRetriever:
    """Embeds the KB once, then answers queries against it."""

    def __init__(self, embedder: TextEmbedder, include_keywords: bool = False):
        self.embedder = embedder
        self.include_keywords = include_keywords
        self.ids, self.matrix = embedder.embed_kb(include_keywords=include_keywords)

    def search(self, query: str, top_k: int = TOP_K) -> list[Candidate]:
        """Return the top_k most similar records, highest score first."""
        q = self.embedder.embed_query(query)
        # matrix rows and q are unit vectors, so the dot product is cosine sim.
        sims = self.matrix @ q
        order = np.argsort(-sims)[:top_k]
        return [
            Candidate(record_id=self.ids[i], score=float(sims[i]),
                      record=get_record_by_id(self.ids[i]))
            for i in order
        ]

    def answer(self, query: str, top_k: int = TOP_K) -> Answer:
        """Search, then abstain if the best score is below the threshold."""
        candidates = self.search(query, top_k=top_k)
        top_score = candidates[0].score if candidates else 0.0
        return Answer(
            answered=top_score >= CONFIDENCE_THRESHOLD,
            candidates=candidates,
            top_score=top_score,
        )
