"""
Shared data structures returned by the fusion layer (backend/fusion/router.py)
and consumed by the UI. Having one common shape here - rather than the UI
reaching into TextRetriever's Answer, ImageRetriever's ImageAnswer, and
SpeechTranscriber's Transcription separately - is what makes the backend
genuinely UI-agnostic: Streamlit today, a future FastAPI/React frontend
later, both just read a single FusionResponse.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModalityResult:
    """What one single modality (text / image / audio) contributed to this request."""
    modality: str            # "text" | "image" | "audio"
    answered: bool           # this modality's OWN answer()/abstain decision, at its own calibrated threshold
    confidence: float        # normalised to [0, 1] via backend/fusion/confidence.py - comparable across modalities
    record_id: str | None    # the KB record id this modality pointed to, or None if it abstained
    raw_score: float         # the native, un-normalised score (cosine similarity or avg_logprob) - kept for debugging/reporting
    extra: dict = field(default_factory=dict)  # e.g. {"transcribed_text": "..."} for audio


@dataclass
class FusionResponse:
    """The single object the UI (or any future frontend) needs to render a reply."""
    answered: bool
    record: dict | None                # full KB record dict if answered, else None
    confidence: float                  # combined, normalised confidence behind the final decision
    modality_results: list[ModalityResult]
    agreement: bool | None             # True/False if >1 modality was used and could be compared; None if only one modality was used
    message: str                       # human-readable summary, ready to show in the UI