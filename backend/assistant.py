from __future__ import annotations

import torch

from backend.schemas import FusionResponse
from backend.fusion.router import process_query as _fusion_process_query

_text_retriever = None
_image_retriever = None
_transcriber = None


def _get_text_retriever():
    global _text_retriever
    if _text_retriever is None:
        from backend.text.embeddings import TextEmbedder
        from backend.text.retrieval import TextRetriever
        _text_retriever = TextRetriever(TextEmbedder())
    return _text_retriever


def _get_image_retriever():
    global _image_retriever
    if _image_retriever is None:
        from backend.image.embeddings import ImageEmbedder
        from backend.image.retrieval import ImageRetriever
        _image_retriever = ImageRetriever(ImageEmbedder())
    return _image_retriever


def _get_transcriber():
    global _transcriber
    if _transcriber is None:
        from backend.audio.whisper_model import SpeechTranscriber
        _transcriber = SpeechTranscriber()
    return _transcriber


def process_query(text: str | None = None, image_path: str | None = None,
                   audio_path: str | None = None) -> FusionResponse:

    text_retriever = _get_text_retriever() if (text or audio_path) else None
    transcriber = _get_transcriber() if audio_path else None

    image_tensor = None
    image_retriever = None
    if image_path is not None:
        from backend.image.preprocess import preprocess
        image_tensor = preprocess(image_path)
        image_retriever = _get_image_retriever()

    return _fusion_process_query(
        text_retriever=text_retriever,
        image_retriever=image_retriever,
        transcriber=transcriber,
        text=text,
        image_tensor=image_tensor,
        audio_path=audio_path,
    )