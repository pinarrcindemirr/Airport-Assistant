from __future__ import annotations

import torch

from backend.kb.kb import get_all_records

DEFAULT_CLIP_MODEL = "ViT-B/32"


class ImageEmbedder:
    """Encodes images and text into CLIP's shared embedding space."""

    def __init__(self, model_name: str = DEFAULT_CLIP_MODEL, device: str | None = None, model=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if model is not None:
            self.model = model
            self.model_name = getattr(model, "name", "injected")
        else:
            import clip
            self.model, _ = clip.load(model_name, device=self.device)
            self.model.eval()
            self.model_name = model_name

    def encode_images(self, image_batch: torch.Tensor) -> torch.Tensor:

        with torch.no_grad():
            features = self.model.encode_image(image_batch.to(self.device)).float()
        return features / features.norm(dim=-1, keepdim=True)

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """Encode a list of strings into L2-normalised CLIP text embeddings."""
        import clip  
        tokens = clip.tokenize(texts, truncate=True).to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(tokens).float()
        return features / features.norm(dim=-1, keepdim=True)

    def embed_kb_visual_descriptions(self) -> tuple[list[str], torch.Tensor]:

        records = get_all_records()
        ids = [r["id"] for r in records]
        texts = [r["visual_description"] for r in records]
        matrix = self.encode_text(texts)
        return ids, matrix