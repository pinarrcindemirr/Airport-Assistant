from __future__ import annotations

from dataclasses import dataclass

import torch

from backend.kb.kb import get_record_by_id
from backend.image.embeddings import ImageEmbedder

IMAGE_CONFIDENCE_THRESHOLD = 0.23  # calibrated against real CLIP scores - see docstring above
IMAGE_TOP_K = 3


@dataclass
class ImageCandidate:
    """One retrieval hit: the record plus its similarity score."""
    record_id: str
    score: float
    record: dict


@dataclass
class ImageAnswer:
    """Outcome of answer(): either a confident hit or an abstention."""
    answered: bool          # False -> hand over to a human
    candidates: list[ImageCandidate]
    top_score: float

    @property
    def record(self) -> dict | None:
        return self.candidates[0].record if (self.answered and self.candidates) else None


class ImageRetriever:
    """Embeds the KB's visual descriptions once, then answers image queries against them."""

    def __init__(self, embedder: ImageEmbedder):
        self.embedder = embedder
        self.ids, self.text_matrix = embedder.embed_kb_visual_descriptions()  # (n_records, d)

    def search(self, image_tensor: torch.Tensor, top_k: int = IMAGE_TOP_K) -> list[ImageCandidate]:
        """image_tensor: a single preprocessed image, shape (3, 224, 224) or (1, 3, 224, 224)."""
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_emb = self.embedder.encode_images(image_tensor)[0]  # (d,)
        sims = self.text_matrix @ image_emb  # (n_records,) cosine sim, both unit vectors
        order = torch.argsort(sims, descending=True)[:top_k]
        return [
            ImageCandidate(record_id=self.ids[i], score=sims[i].item(),
                            record=get_record_by_id(self.ids[i]))
            for i in order.tolist()
        ]

    def answer(self, image_tensor: torch.Tensor, top_k: int = IMAGE_TOP_K) -> ImageAnswer:
        """Search, then abstain if the best score is below the (placeholder) threshold."""
        candidates = self.search(image_tensor, top_k=top_k)
        top_score = candidates[0].score if candidates else 0.0
        return ImageAnswer(
            answered=top_score >= IMAGE_CONFIDENCE_THRESHOLD,
            candidates=candidates,
            top_score=top_score,
        )