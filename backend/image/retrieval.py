"""
Image retrieval over the knowledge base.

Given a preprocessed passenger photo, return the most relevant KB record(s)
by cosine similarity against every record's CLIP-encoded visual_description,
and decide whether the system is confident enough to answer. This mirrors
backend/text/retrieval.py's search()/answer() split exactly, so both
modalities expose the same interface to the multimodal fusion layer.

On the confidence threshold: CLIP image-to-text cosine similarities and
sentence-transformer text-to-text similarities are NOT on the same scale
(CLIP scores typically sit lower - often 0.15-0.35 even for a correct match,
versus sentence-transformers' 0.3-0.6 seen in the text pipeline).

IMAGE_CONFIDENCE_THRESHOLD was calibrated against real evaluation data
(80 images, ViT-B/32): correct top-1 predictions scored 0.235-0.376
(mean 0.306), while incorrect top-1 predictions scored 0.238-0.324
(mean 0.285) - the two distributions almost completely overlap. Unlike the
text pipeline, where the confidence threshold cleanly separated answered
from abstained queries, a similarity threshold provides very little
discriminative power here: CLIP is often just as "confident" when wrong as
when right, because most errors come from genuinely near-duplicate KB
content (e.g. gate_a05 vs gate_c22 differ only by a short alphanumeric code
CLIP reads unreliably; several categories originally shared an identical
sign colour) rather than from genuine uncertainty. The threshold is set just
below the observed floor of both distributions (~0.235) so the system rarely
abstains on this dataset; abstention here is a weak safety net, and the real
fix for the observed errors is disambiguating KB content, not threshold
tuning (see backend/image/evaluation.py output and the report's error
analysis for specifics).
"""

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