"""
Audio preprocessing pipeline for the speech component.

Covers every subtask the brief asks for: audio loading, format conversion,
noise/silence handling, and optional MFCC extraction (via Librosa). The
output of this module feeds directly into backend/audio/whisper_model.py
for transcription.

Design notes:
  - Whisper expects mono audio at 16kHz. Passenger recordings (phone voice
    memos, browser mic capture) arrive in all sorts of sample rates and
    channel counts, so resampling/mono-conversion is not optional - skipping
    it silently degrades transcription quality rather than raising an error.
  - Silence trimming (librosa.effects.trim) removes leading/trailing dead
    air from a recording (e.g. the button-press delay before a passenger
    starts speaking) without altering the speech itself, since Whisper's
    internal voice-activity handling works better on tightly-cropped audio.
  - MFCC extraction is implemented and exposed because the brief explicitly
    lists it as an optional subtask, but it is NOT on the transcription path:
    Whisper takes raw waveform (or its own log-mel spectrogram internally),
    not MFCCs. MFCCs are provided here only for exploratory analysis (e.g.
    comparing the acoustic profile of a quiet vs a noisy recording) if an
    audio classification side-component is added later.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf


def _ensure_ffmpeg_on_path() -> None:
    """
    librosa falls back to ffmpeg (via audioread) for formats libsndfile can't
    read directly - notably .m4a/.mp3 phone recordings, which is exactly what
    passengers are expected to upload/record. Rather than requiring a manual
    system-wide ffmpeg install (awkward on Windows without admin rights), we
    use the `imageio-ffmpeg` package, which ships a self-contained ffmpeg
    binary as a normal pip install.

    Gotcha this works around: imageio-ffmpeg's binary is NOT named
    "ffmpeg"/"ffmpeg.exe" (e.g. it's "ffmpeg-win64-v4.2.2.exe"), but audioread
    invokes the literal command "ffmpeg" via subprocess - just adding the
    binary's folder to PATH is not enough, since nothing on PATH matches that
    exact name. We copy the real binary once into a local ".ffmpeg_bin" folder
    under the alias "ffmpeg"/"ffmpeg.exe" and put THAT folder on PATH instead.

    If imageio-ffmpeg isn't installed, this silently does nothing (a system
    ffmpeg already on PATH will still work as before).
    """
    try:
        import shutil
        import sys
        import imageio_ffmpeg

        real_exe = Path(imageio_ffmpeg.get_ffmpeg_exe())
        alias_dir = Path(__file__).resolve().parent / ".ffmpeg_bin"
        alias_dir.mkdir(exist_ok=True)
        alias_name = "ffmpeg.exe" if sys.platform.startswith("win") else "ffmpeg"
        alias_path = alias_dir / alias_name

        if not alias_path.exists():
            shutil.copy2(real_exe, alias_path)
            alias_path.chmod(0o755)

        alias_dir_str = str(alias_dir)
        if alias_dir_str not in os.environ.get("PATH", ""):
            os.environ["PATH"] = alias_dir_str + os.pathsep + os.environ.get("PATH", "")
    except ImportError:
        pass  # falls back to a system ffmpeg on PATH, if any


_ensure_ffmpeg_on_path()

WHISPER_SAMPLE_RATE = 16_000


def load_audio(path: str | Path, target_sr: int = WHISPER_SAMPLE_RATE) -> np.ndarray:
    """
    Load an audio file from disk, converting to mono and resampling to
    `target_sr` in one step. Accepts wav/mp3/m4a/flac - whatever librosa's
    backend (soundfile / audioread) can decode.
    """
    waveform, _ = librosa.load(str(path), sr=target_sr, mono=True)
    return waveform


def trim_silence(waveform: np.ndarray, top_db: float = 30.0) -> np.ndarray:
    """
    Trim leading/trailing silence below `top_db` relative to the loudest
    part of the clip. Does not touch silence between words mid-sentence.
    """
    trimmed, _ = librosa.effects.trim(waveform, top_db=top_db)
    return trimmed


def normalize_volume(waveform: np.ndarray) -> np.ndarray:
    """Peak-normalise so the loudest sample hits ~0.95, correcting for quiet recordings."""
    peak = np.abs(waveform).max()
    if peak < 1e-6:
        return waveform  # silent clip - leave as is rather than dividing by ~0
    return waveform * (0.95 / peak)


def extract_mfcc(waveform: np.ndarray, sr: int = WHISPER_SAMPLE_RATE, n_mfcc: int = 13) -> np.ndarray:
    """
    Extract MFCCs for exploratory acoustic analysis only (see module
    docstring) - not used on the Whisper transcription path.
    """
    return librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=n_mfcc)


def preprocess_audio(path: str | Path, trim: bool = True, normalize: bool = True) -> np.ndarray:
    """Full pipeline: load -> resample/mono -> (optional trim) -> (optional normalise)."""
    waveform = load_audio(path)
    if trim:
        waveform = trim_silence(waveform)
    if normalize:
        waveform = normalize_volume(waveform)
    return waveform


def save_wav(waveform: np.ndarray, path: str | Path, sr: int = WHISPER_SAMPLE_RATE) -> None:
    """Write a processed waveform back to disk (useful for sanity-checking the pipeline by ear)."""
    sf.write(str(path), waveform, sr)


if __name__ == "__main__":
    # Demo / sanity check: python -m backend.audio.preprocess <path_to_wav>
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m backend.audio.preprocess <path_to_audio_file>")
        raise SystemExit(1)

    src = sys.argv[1]
    raw = load_audio(src)
    processed = preprocess_audio(src)
    mfcc = extract_mfcc(processed)

    print(f"Raw duration:       {len(raw) / WHISPER_SAMPLE_RATE:.2f}s  ({len(raw)} samples)")
    print(f"Processed duration: {len(processed) / WHISPER_SAMPLE_RATE:.2f}s  ({len(processed)} samples)")
    print(f"MFCC shape:         {mfcc.shape}  (n_mfcc, n_frames)")

    out_path = Path(src).with_name(Path(src).stem + "_preprocessed.wav")
    save_wav(processed, out_path)
    print(f"Preprocessed audio saved to {out_path}")