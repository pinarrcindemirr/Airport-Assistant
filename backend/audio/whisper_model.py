"""
Speech-to-text wrapper around OpenAI's Whisper (frozen, per the brief -
"Pretrained models such as CLIP and Whisper should be used as frozen models
and do not require training").

The recommended default model size is "base": small enough to run on a CPU
in reasonable time for a proof-of-concept (no GPU assumption), while still
handling accented/noisy speech noticeably better than "tiny". This can be
raised to "small"/"medium" if accuracy matters more than latency for the
final deployed app - see the trade-off note in transcribe().

The model is injectable (the `model` argument), mirroring TextEmbedder and
ImageEmbedder in the other two pipelines: this lets the surrounding
preprocessing/evaluation logic be exercised with a lightweight stand-in,
without downloading a real Whisper checkpoint every time the code is
imported or tested.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_WHISPER_MODEL = "base"

# Passed as Whisper's initial_prompt: a short list of domain vocabulary that
# biases decoding towards airport-specific terms. Added directly in response
# to real evaluation evidence (see backend/audio/evaluation.py output) that
# "check-in" and "lounge" were misheard as "chicken" and "legge" - both
# uncommon words in Whisper's general training distribution.
AIRPORT_VOCABULARY_PROMPT = (
    "Airport terms: gate, check-in, baggage claim, security, lounge, "
    "restroom, prayer room, customs, currency exchange, information desk, "
    "boarding pass, terminal, special assistance."
)

# Placeholder - see calibration note in SpeechTranscriber.transcribe_with_confidence.
# Calibrated against 15 real recordings (see backend/audio/evaluation.py output):
# correct transcriptions scored -0.624 to -0.410, incorrect ones -4.794 to -0.441.
# -0.65 sits just below the worst correct case (-0.624) while still catching
# both real errors (-0.653 and -4.794), fixing a false-abstain that occurred
# at the original -0.6 cutoff without losing either genuine catch.
AUDIO_CONFIDENCE_THRESHOLD = -0.65


@dataclass
class Transcription:
    """
    Result of a confidence-aware transcription, mirroring the answered/
    top_score pattern used by TextRetriever.answer() and ImageRetriever.answer()
    so all three modalities expose the same shape to the fusion layer.
    """
    text: str
    avg_logprob: float     # mean log-probability Whisper assigned its own tokens (closer to 0 = more confident)
    no_speech_prob: float  # Whisper's own estimate that this segment was not speech at all
    answered: bool         # False -> hand over to a human / ask the passenger to repeat


def _segment_confidence(result: dict) -> tuple[float, float]:
    """
    Aggregate Whisper's per-segment avg_logprob/no_speech_prob into single
    clip-level numbers. Segments are averaged (not just the first/last),
    since even a short utterance can be split into more than one segment.
    """
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
            # Injected model (e.g. a test stand-in exposing .transcribe(audio) -> dict).
            self.model = model
            self.model_name = getattr(model, "name", "injected")
        else:
            # Lazy import: only needed when actually loading real Whisper weights,
            # so this module can still be imported (and tested with an injected
            # model) without the `whisper` package installed.
            import whisper
            self.model = whisper.load_model(model_name)
            self.model_name = model_name

    def transcribe(self, waveform: np.ndarray, language: str | None = "en",
                    initial_prompt: str | None = AIRPORT_VOCABULARY_PROMPT) -> dict:
        """
        Transcribe a preprocessed waveform (mono, 16kHz float32 - see
        backend/audio/preprocess.py) into text.

        initial_prompt biases Whisper's decoding towards the given
        vocabulary without any fine-tuning; defaults to
        AIRPORT_VOCABULARY_PROMPT. Pass None to disable (e.g. to reproduce
        the baseline numbers without the vocabulary hint, for comparison).

        Returns the full Whisper result dict (text, segments, detected
        language, etc.) rather than just the string, so confidence-relevant
        fields (e.g. per-segment avg_logprob) remain available for the
        evaluation script and for a future confidence/abstention mechanism
        matching the text and image pipelines.
        """
        # Whisper's own API accepts either a file path or a float32 numpy array
        # directly; passing the array avoids a redundant disk round-trip since
        # preprocess.py already returns one.
        return self.model.transcribe(waveform.astype(np.float32), language=language,
                                      initial_prompt=initial_prompt)

    def transcribe_file(self, path: str | Path, language: str | None = "en",
                         initial_prompt: str | None = AIRPORT_VOCABULARY_PROMPT) -> dict:
        """Convenience path: preprocess a raw audio file on disk, then transcribe it."""
        from backend.audio.preprocess import preprocess_audio
        waveform = preprocess_audio(path)
        return self.transcribe(waveform, language=language, initial_prompt=initial_prompt)

    def transcribe_with_confidence(self, waveform: np.ndarray, language: str | None = "en",
                                    initial_prompt: str | None = AIRPORT_VOCABULARY_PROMPT) -> Transcription:
        """
        Same as transcribe(), but returns a Transcription with an explicit
        answered/abstain decision, mirroring TextRetriever.answer() and
        ImageRetriever.answer(). AUDIO_CONFIDENCE_THRESHOLD (-0.6 on the
        avg_logprob scale) is a PLACEHOLDER: unlike the text and image
        thresholds, which were calibrated against 40-80 examples each, only
        15 recordings exist for audio so far - not enough to reliably locate
        a correct/incorrect score boundary. Recalibrate the same way once a
        larger recorded set is evaluated (see backend/audio/evaluation.py,
        which now prints avg_logprob/no_speech_prob per example for exactly
        this purpose).
        """
        result = self.transcribe(waveform, language=language, initial_prompt=initial_prompt)
        avg_logprob, no_speech_prob = _segment_confidence(result)
        answered = (avg_logprob >= AUDIO_CONFIDENCE_THRESHOLD) and (no_speech_prob < 0.6)
        return Transcription(text=result["text"].strip(), avg_logprob=avg_logprob,
                              no_speech_prob=no_speech_prob, answered=answered)

    def transcribe_file_with_confidence(self, path: str | Path, language: str | None = "en",
                                         initial_prompt: str | None = AIRPORT_VOCABULARY_PROMPT) -> Transcription:
        """File-path convenience wrapper around transcribe_with_confidence()."""
        from backend.audio.preprocess import preprocess_audio
        waveform = preprocess_audio(path)
        return self.transcribe_with_confidence(waveform, language=language, initial_prompt=initial_prompt)