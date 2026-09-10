from __future__ import annotations

import torch

from backend.schemas import FusionResponse
from backend.fusion.router import process_query as _fusion_process_query

_text_retriever = None
_image_retriever = None
_transcriber = None
_ocr_reader = None


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


def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        from backend.image.ocr_reader import SignTextReader
        _ocr_reader = SignTextReader()
    return _ocr_reader

def warm_up() -> None:
    
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(_get_text_retriever),
            executor.submit(_get_image_retriever),
            executor.submit(_get_transcriber),
            executor.submit(_get_ocr_reader),
        ]
        for f in futures:
            f.result()

def process_query(text: str | None = None, image_path: str | None = None,
                   audio_path: str | None = None) -> FusionResponse:
    text_retriever = _get_text_retriever() if (text or audio_path or image_path) else None
    transcriber = _get_transcriber() if audio_path else None
    ocr_reader = _get_ocr_reader() if image_path else None

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
        ocr_reader=ocr_reader,
        text=text,
        image_tensor=image_tensor,
        image_path=image_path,
        audio_path=audio_path,
    )