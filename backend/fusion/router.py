from __future__ import annotations

import torch

from backend.kb.kb import get_record_by_id
from backend.schemas import ModalityResult, FusionResponse
from backend.fusion.confidence import (
    text_confidence, image_confidence, audio_confidence, ocr_confidence, combined_confidence,
)

FUSION_CONFIDENCE_THRESHOLD = 0.5


def process_query(text_retriever=None, image_retriever=None, transcriber=None, ocr_reader=None,
                   text: str | None = None, image_tensor: torch.Tensor | None = None,
                   image_path: str | None = None, audio_path: str | None = None) -> FusionResponse:

    modality_results: list[ModalityResult] = []

    resolved_text = text
    if audio_path is not None:
        if transcriber is None:
            raise ValueError("audio_path given but no transcriber provided")
        transcription = transcriber.transcribe_file_with_confidence(audio_path)
        modality_results.append(ModalityResult(
            modality="audio",
            answered=transcription.answered,
            confidence=audio_confidence(transcription.avg_logprob),
            record_id=None,  
            raw_score=transcription.avg_logprob,
            extra={"transcribed_text": transcription.text, "no_speech_prob": transcription.no_speech_prob},
        ))
        if resolved_text is None:
            resolved_text = transcription.text  

    if image_path is not None:
        if ocr_reader is None:
            raise ValueError("image_path given but no ocr_reader provided")
        ocr_result = ocr_reader.read(image_path)
        modality_results.append(ModalityResult(
            modality="ocr",
            answered=ocr_result.answered,
            confidence=ocr_confidence(ocr_result.confidence),
            record_id=None,  
            raw_score=ocr_result.confidence,
            extra={"ocr_text": ocr_result.text, "num_detections": ocr_result.num_detections},
        ))
        if resolved_text is None and ocr_result.text:
            resolved_text = ocr_result.text 

    if resolved_text:
        if text_retriever is None:
            raise ValueError("text given but no text_retriever provided")
        text_answer = text_retriever.answer(resolved_text)
        modality_results.append(ModalityResult(
            modality="text",
            answered=text_answer.answered,
            confidence=text_confidence(text_answer.top_score),
            record_id=text_answer.record["id"] if text_answer.record else None,
            raw_score=text_answer.top_score,
            extra={"query": resolved_text},
        ))

    if image_tensor is not None:
        if image_retriever is None:
            raise ValueError("image_tensor given but no image_retriever provided")
        image_answer = image_retriever.answer(image_tensor)
        modality_results.append(ModalityResult(
            modality="image",
            answered=image_answer.answered,
            confidence=image_confidence(image_answer.top_score),
            record_id=image_answer.record["id"] if image_answer.record else None,
            raw_score=image_answer.top_score,
        ))

    return _combine(modality_results)


def _combine(modality_results: list[ModalityResult]) -> FusionResponse:
    if not modality_results:
        return FusionResponse(answered=False, record=None, confidence=0.0,
                               modality_results=[], agreement=None,
                               message="No input was provided.")

    candidates = [m for m in modality_results if m.record_id is not None]
    agreement = None
    if len(candidates) > 1:
        agreement = len({m.record_id for m in candidates}) == 1

    confident_candidates = [m for m in candidates if m.answered]

    if not confident_candidates:
        combined_conf = combined_confidence([m.confidence for m in modality_results])
        return FusionResponse(
            answered=False, record=None, confidence=combined_conf,
            modality_results=modality_results, agreement=agreement,
            message="I'm not confident enough to answer that - please check "
                     "with airport staff or an information desk.",
        )

    if agreement:
        chosen_id = confident_candidates[0].record_id
        combined_conf = combined_confidence([m.confidence for m in confident_candidates])
    else:
        best = max(confident_candidates, key=lambda m: m.confidence)
        chosen_id = best.record_id
        combined_conf = best.confidence

    record = get_record_by_id(chosen_id)
    answered = combined_conf >= FUSION_CONFIDENCE_THRESHOLD

    if answered:
        message = f"Here's what I found: {record['name']}."
        if agreement is True:
            message += " (confirmed by more than one input)"
        elif agreement is False:
            alt_ids = {m.record_id for m in confident_candidates} - {chosen_id}
            if alt_ids:
                message += f" Note: another input suggested a different match ({', '.join(alt_ids)})."
    else:
        message = "I found a possible match but I'm not confident enough -please double-check with airport staff."

    return FusionResponse(
        answered=answered,
        record=record if answered else None,
        confidence=combined_conf,
        modality_results=modality_results,
        agreement=agreement,
        message=message,
    )