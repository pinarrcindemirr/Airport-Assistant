"""
CLIP embedding layer for the vision pipeline.

Wraps OpenAI's CLIP (ViT-B/32, the same model used in the course notebook)
to produce image and text embeddings in CLIP's shared multimodal space.
This is what makes "matching images to knowledge base descriptions" (per
the brief) possible without training anything: a KB record's
`visual_description` text and an uploaded passenger photo can be compared
directly by cosine similarity, because CLIP's image and text towers were
jointly trained to place matching pairs close together in the same space.

Design notes:
  - CLIP is used strictly as a FROZEN pretrained model, per the brief
    ("Pretrained models such as CLIP and Whisper should be used as frozen
    models and do not require training").
  - The model is injectable (the `model` argument), mirroring
    backend/text/embeddings.py's TextEmbedder: this lets the retrieval logic
    be exercised with a lightweight stand-in, without requiring the ~350MB
    CLIP checkpoint download every time the code is imported or tested.
  - Image tensors passed in are expected to already be preprocessed by
    backend/image/preprocess.py (resize/crop/normalise matching CLIP's own
    conventions) - CLIP's bundled `preprocess` transform is intentionally
    not used, so there is a single, explicit preprocessing implementation
    rather than two that must be kept in sync.
"""

from __future__ import annotations

import torch

from backend.kb.kb import get_all_records

DEFAULT_CLIP_MODEL = "ViT-B/32"


class ImageEmbedder:
    """Encodes images and text into CLIP's shared embedding space."""

    def __init__(self, model_name: str = DEFAULT_CLIP_MODEL, device: str | None = None, model=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if model is not None:
            # Injected model (e.g. a test stand-in exposing encode_image/encode_text
            # with the same signatures as a real CLIP model).
            self.model = model
            self.model_name = getattr(model, "name", "injected")
        else:
            # Lazy import: only needed when actually loading real CLIP weights,
            # so this module can still be imported (and tested with an
            # injected model) without the `clip` package installed.
            import clip
            self.model, _ = clip.load(model_name, device=self.device)
            self.model.eval()
            self.model_name = model_name

    def encode_images(self, image_batch: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of already-preprocessed images (N, 3, 224, 224) into
        L2-normalised CLIP embeddings, so cosine similarity reduces to a
        plain dot product (same convention as backend/text/embeddings.py).
        """
        with torch.no_grad():
            features = self.model.encode_image(image_batch.to(self.device)).float()
        return features / features.norm(dim=-1, keepdim=True)

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """Encode a list of strings into L2-normalised CLIP text embeddings."""
        import clip  # lazy: only the tokenizer is needed here, kept import-light
        tokens = clip.tokenize(texts, truncate=True).to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(tokens).float()
        return features / features.norm(dim=-1, keepdim=True)

    def embed_kb_visual_descriptions(self) -> tuple[list[str], torch.Tensor]:
        """
        Encode every KB record's `visual_description` field with CLIP's text
        tower. This is the anchor an uploaded image is matched against - the
        image is compared to TEXT, never to another image, which is what
        makes zero-shot retrieval possible without a labelled image gallery
        or any training step.
        """
        records = get_all_records()
        ids = [r["id"] for r in records]
        texts = [r["visual_description"] for r in records]
        matrix = self.encode_text(texts)
        return ids, matrix