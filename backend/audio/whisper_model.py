from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_WHISPER_MODEL = "base"

AIRPORT_VOCABULARY_PROMPT = (
    "Airport terms: gate, check-in, baggage claim, security, lounge, "
    "restroom, prayer room, customs, currency exchange, information desk, "
    "boarding pass, terminal, special assistance."
)

AUDIO_CONFIDENCE_THRESHOLD = -0.65


@dataclass
class Transcription:

    text: str
    avg_logprob: float     # mean log-probability Whisper assigned its own tokens (closer to 0 = more confident)
    no_speech_prob: float  # Whisper's own estimate that this segment was not speech at all
    answered: bool         # False -> hand over to a human / ask the passenger to repeat


def _segment_confidence(result: dict) -> tuple[float, float]:

    segments = result.get("segments", [])
    if not segments:
        return float("nan"), float("nan")
    avg_logprob = sum(s["avg_logprob"] for s in segments) / len(segments)
    no_speech_prob = max(s["no_speech_prob"] for s in segments)  # worst case, not averaged
    return avg_logprob, no_speech_prob


class SpeechTranscriber:
    """Wraps a Whisper model to transcribe preprocessed waveforms into text."""

    def __init__(self, model_name: str = DEFAULT_WHISPER_MODEL, model=None):
        if model is not None:
            # Injected model
            self.model = model
            self.model_name = getattr(model, "name", "injected")
        else:
            import whisper
            self.model = whisper.load_model(model_name)
            self.model_name = model_name

    def transcribe(self, waveform: np.ndarray, language: str | None = "en",
                    initial_prompt: str | None = AIRPORT_VOCABULARY_PROMPT) -> dict:

        return self.model.transcribe(waveform.astype(np.float32), language=language,
                                      initial_prompt=initial_prompt)

    def transcribe_file(self, path: str | Path, language: str | None = "en",
                         initial_prompt: str | None = AIRPORT_VOCABULARY_PROMPT) -> dict:
        from backend.audio.preprocess import preprocess_audio
        waveform = preprocess_audio(path)
        return self.transcribe(waveform, language=language, initial_prompt=initial_prompt)

    def transcribe_with_confidence(self, waveform: np.ndarray, language: str | None = "en",
                                    initial_prompt: str | None = AIRPORT_VOCABULARY_PROMPT) -> Transcription:

        result = self.transcribe(waveform, language=language, initial_prompt=initial_prompt)
        avg_logprob, no_speech_prob = _segment_confidence(result)
        answered = (avg_logprob >= AUDIO_CONFIDENCE_THRESHOLD) and (no_speech_prob < 0.6)
        return Transcription(text=result["text"].strip(), avg_logprob=avg_logprob,
                              no_speech_prob=no_speech_prob, answered=answered)

    def transcribe_file_with_confidence(self, path: str | Path, language: str | None = "en",
                                         initial_prompt: str | None = AIRPORT_VOCABULARY_PROMPT) -> Transcription:
        from backend.audio.preprocess import preprocess_audio
        waveform = preprocess_audio(path)
        return self.transcribe_with_confidence(waveform, language=language, initial_prompt=initial_prompt)