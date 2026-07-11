"""
Single entry point for the whole assistant: process_query(text, image_path, audio_path).

This is the ONLY module the UI (Streamlit today, potentially FastAPI/React
later) needs to import. It hides all model-loading and pipeline wiring
behind one function call, which is what makes the backend genuinely
UI-agnostic - swapping Streamlit for a React+FastAPI frontend later only
means writing a new thin layer that calls this same function.

Models are loaded lazily (on first use) and cached as module-level
singletons, so CLIP/Whisper/the sentence-transformer are each loaded ONCE
per process, not once per request - important for a Streamlit app, where the
script re-runs on every user interaction.
"""

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
    """
    The one function the UI calls. Pass whichever of text / image_path /
    audio_path the passenger actually provided this turn; leave the rest as
    None (not empty string/None-equivalent placeholders - genuinely absent).

    image_path is a file path, not a preprocessed tensor: this function
    owns turning a raw upload into the tensor backend/image/retrieval.py
    expects, so the UI layer never has to know CLIP's input format.
    """
    # text_retriever is needed whenever there's a text query to score - either
    # typed directly, OR derived from audio transcription inside router.py.
    # Loading it only when `text` was given (ignoring audio_path) was a bug:
    # a voice-only request would transcribe fine but then crash inside
    # router.process_query with "no text_retriever provided".
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