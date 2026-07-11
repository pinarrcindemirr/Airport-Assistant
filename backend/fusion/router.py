"""
Rule-based multimodal fusion / routing layer.

Combines whichever of text/image/audio inputs are present into a single
FusionResponse, handling every input scenario the brief requires: text-only,
voice-only, image-only, image+text together, and voice+image together.

Design: RULE-BASED, not a trained fusion network. Every one of CLIP, Whisper,
and the sentence-transformer used elsewhere in this project is a frozen,
pretrained model (per the brief). A learned fusion layer (e.g. an attention
MLP deciding modality weights) would need its own training data and training
loop to produce meaningful weights - without one, its weights would be
random, not learned, regardless of how sophisticated the architecture looks.
Rule-based routing is explicitly accepted by the brief as sufficient
("Rule-based routing" is listed alongside more advanced optional strategies),
and has the advantage that every decision is traceable to an explicit,
inspectable rule rather than an opaque, untrained set of weights.

The core rule, in order:
  1. If audio is given, transcribe it first - voice is treated as a route
     INTO the text pipeline, not a separate destination. If explicit typed
     text is ALSO given, the typed text takes precedence as the query (audio
     is still scored and reported on its own, e.g. for confidence display).
  2. Whichever of text/image are present each produce their OWN answer()
     decision, at their OWN already-calibrated threshold (backend/text/retrieval.py,
     backend/image/retrieval.py, backend/audio/whisper_model.py).
  3. Raw, incompatible-scale scores are normalised to a common [0, 1] via
     backend/fusion/confidence.py.
  4. If more than one modality is present AND they point to the SAME KB
     record -> treat that agreement as reinforcing evidence.
     If they point to DIFFERENT records -> the modality with the higher
     normalised confidence wins; the other is kept as a visible alternative,
     never silently discarded.
  5. The fusion layer NEVER answers with a record that no individual
     modality was confident enough to support on its own - it only
     arbitrates BETWEEN modalities that already passed their own bar, or
     confirms agreement between them. If nothing passed its own bar, the
     system abstains and hands over to a human, regardless of how the raw
     numbers might look combined.
"""

from __future__ import annotations

import torch

from backend.kb.kb import get_record_by_id
from backend.schemas import ModalityResult, FusionResponse
from backend.fusion.confidence import text_confidence, image_confidence, audio_confidence, combined_confidence

# Placeholder, like the per-modality thresholds were before their own real
# calibration: this needs revisiting once real multi-modality test scenarios
# (brief's required 5 structured scenarios) have been run and their combined
# confidence numbers observed - see backend/fusion/evaluation.py (deployment
# stage) for that data once it exists.
FUSION_CONFIDENCE_THRESHOLD = 0.5


def process_query(text_retriever=None, image_retriever=None, transcriber=None,
                   text: str | None = None, image_tensor: torch.Tensor | None = None,
                   audio_path: str | None = None) -> FusionResponse:
    """
    Route whichever of text/image_tensor/audio_path are provided (None =
    "not provided by the user this turn", not "provided but empty") through
    their respective pipelines, then combine the results.

    text_retriever / image_retriever / transcriber are injected (not
    constructed here) so this function can be unit-tested with lightweight
    stand-ins, exactly like each pipeline's own embedder classes - and so
    backend/assistant.py (the real entry point) only has to load the actual
    heavy models once, not on every call.
    """
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
            record_id=None,  # audio itself doesn't point to a KB record - only the resulting text does
            raw_score=transcription.avg_logprob,
            extra={"transcribed_text": transcription.text, "no_speech_prob": transcription.no_speech_prob},
        ))
        if resolved_text is None:
            resolved_text = transcription.text  # voice becomes the query only if no typed text was given

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

    # Only modalities that point at an actual KB record can participate in
    # agreement/arbitration (audio's own ModalityResult has record_id=None -
    # it contributes a confidence signal but never a candidate record itself).
    candidates = [m for m in modality_results if m.record_id is not None]
    agreement = None
    if len(candidates) > 1:
        agreement = len({m.record_id for m in candidates}) == 1

    # Never answer with a record that no individual modality was itself
    # confident enough to support - only arbitrate between/confirm among
    # modalities that already passed their OWN calibrated threshold.
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
        message = "I found a possible match but I'm not confident enough - please double-check with airport staff."

    return FusionResponse(
        answered=answered,
        record=record if answered else None,
        confidence=combined_conf,
        modality_results=modality_results,
        agreement=agreement,
        message=message,
    )