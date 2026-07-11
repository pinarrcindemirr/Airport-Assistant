"""
Shared confidence normalisation for the fusion layer.

Each modality's underlying model produces a confidence-like number on a
different, mathematically incompatible scale:
  - Text  (backend/text/retrieval.py Answer.top_score):    cosine similarity, higher = better
  - Image (backend/image/retrieval.py ImageAnswer.top_score): cosine similarity, higher = better
  - Audio (backend/audio/whisper_model.py Transcription.avg_logprob): log-probability,
    LESS NEGATIVE = better (inverted relative to the other two)

None of these are directly averageable as-is: a text score of 0.35 and an
audio avg_logprob of -0.35 are not comparable, so blindly averaging raw
scores would be mathematically meaningless (and audio's sign convention is
even inverted relative to the other two). This module rescales each
modality's raw score onto a common [0, 1] "normalised confidence" using the
observed floor/ceiling from that modality's own real evaluation data (see
backend/text/evaluation, backend/image/evaluation.py,
backend/audio/evaluation.py), so the fusion layer can combine confidences
from ANY subset of available modalities - text-only, image-only, voice-only,
image+text together, voice+image together, etc. - via a simple mean once
everything is on the same scale.

The floor/ceiling constants below are calibration values based on observed
score distributions during evaluation, not universal constants - like the
per-modality confidence thresholds, they should be revisited if the KB,
image set, or audio dataset changes meaningfully.
"""

from __future__ import annotations

# (floor, ceiling) observed from each pipeline's real evaluation runs.
# floor: roughly where that modality's own abstention threshold sits.
# ceiling: roughly the best score seen on a clean, confident, correct match.
TEXT_SCORE_RANGE = (0.30, 0.60)
IMAGE_SCORE_RANGE = (0.22, 0.38)
AUDIO_LOGPROB_RANGE = (-1.5, -0.05)  # inverted: less negative is better


def _normalize(value: float, floor: float, ceiling: float) -> float:
    """Linearly rescale value from [floor, ceiling] to [0, 1], clipped at both ends."""
    if ceiling == floor:
        return 0.0
    return max(0.0, min(1.0, (value - floor) / (ceiling - floor)))


def text_confidence(cosine_score: float) -> float:
    """Normalise a TextRetriever Answer.top_score to [0, 1]."""
    return _normalize(cosine_score, *TEXT_SCORE_RANGE)


def image_confidence(cosine_score: float) -> float:
    """Normalise an ImageRetriever ImageAnswer.top_score to [0, 1]."""
    return _normalize(cosine_score, *IMAGE_SCORE_RANGE)


def audio_confidence(avg_logprob: float) -> float:
    """Normalise a Transcription.avg_logprob to [0, 1] (handles the inverted sign)."""
    return _normalize(avg_logprob, *AUDIO_LOGPROB_RANGE)


def combined_confidence(confidences: list[float]) -> float:
    """
    Mean of whichever normalised per-modality confidences are actually
    available for THIS request - e.g. just [text_confidence(...)] for a
    text-only query, or [text_confidence(...), image_confidence(...)] for a
    combined image+text query.

    Callers must only pass confidences for modalities that were genuinely
    used for this request - never pad a missing modality with a default
    value (e.g. 0.0 or 0.5), since that would silently drag the combined
    score up or down based on something that was never actually evaluated.
    """
    if not confidences:
        return 0.0
    return sum(confidences) / len(confidences)


if __name__ == "__main__":
    # Demo: python -m backend.fusion.confidence
    print("Text-only query, top_score=0.42:")
    print(f"  confidence = {combined_confidence([text_confidence(0.42)]):.3f}\n")

    print("Image-only query, top_score=0.30:")
    print(f"  confidence = {combined_confidence([image_confidence(0.30)]):.3f}\n")

    print("Voice-only query, avg_logprob=-0.20:")
    print(f"  confidence = {combined_confidence([audio_confidence(-0.20)]):.3f}\n")

    print("Combined image+text query, image=0.30, text=0.42:")
    combined = combined_confidence([image_confidence(0.30), text_confidence(0.42)])
    print(f"  confidence = {combined:.3f}\n")

    print("Combined voice+image query, voice=-0.20, image=0.22 (near its floor):")
    combined = combined_confidence([audio_confidence(-0.20), image_confidence(0.22)])
    print(f"  confidence = {combined:.3f}")