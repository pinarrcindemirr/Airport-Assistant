from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from backend.utils.config import ROOT_DIR

N_RUNS = 10
SAMPLE_IMAGE = ROOT_DIR / "data" / "images" / "gate" / "gate_b12_v1.png"
SAMPLE_AUDIO = ROOT_DIR / "data" / "audio" / "raw" / "a01.m4a"
SAMPLE_TEXT = "where is gate B12"


def _time_calls(fn, n: int = N_RUNS) -> dict:
    """Call fn() n times (after one untimed warm-up), return timing stats in ms."""
    fn()  
    times = []
    for _ in range(n):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)  # seconds -> ms
    return {
        "mean_ms": float(np.mean(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "n": n,
    }


def benchmark_text() -> dict:
    from backend.text.embeddings import TextEmbedder
    from backend.text.retrieval import TextRetriever

    embedder = TextEmbedder()
    retriever = TextRetriever(embedder)
    return _time_calls(lambda: retriever.answer(SAMPLE_TEXT))


def benchmark_image() -> dict:
    from backend.image.embeddings import ImageEmbedder
    from backend.image.retrieval import ImageRetriever
    from backend.image.preprocess import preprocess

    embedder = ImageEmbedder()
    retriever = ImageRetriever(embedder)
    image_tensor = preprocess(SAMPLE_IMAGE)
    return _time_calls(lambda: retriever.answer(image_tensor))


def benchmark_audio() -> dict:
    from backend.audio.whisper_model import SpeechTranscriber

    transcriber = SpeechTranscriber()
    return _time_calls(lambda: transcriber.transcribe_file_with_confidence(SAMPLE_AUDIO))


def benchmark_ocr() -> dict:
    from backend.image.ocr_reader import SignTextReader

    reader = SignTextReader()
    return _time_calls(lambda: reader.read(SAMPLE_IMAGE))


BENCHMARKS = {
    "text (MiniLM + FAISS)": benchmark_text,
    "image (CLIP)": benchmark_image,
    "audio (Whisper base)": benchmark_audio,
    "ocr (EasyOCR)": benchmark_ocr,
}


def _print_table(results: dict) -> None:
    print(f"\nCPU latency benchmark (n={N_RUNS} timed calls per model, after 1 warm-up call)\n")
    print(f"{'component':<24}{'mean (ms)':>12}{'min (ms)':>12}{'max (ms)':>12}")
    print("-" * 60)
    for name, stats in results.items():
        print(f"{name:<24}{stats['mean_ms']:>12.1f}{stats['min_ms']:>12.1f}{stats['max_ms']:>12.1f}")


def main() -> None:
    results = {}
    for name, fn in BENCHMARKS.items():
        print(f"Benchmarking {name}...")
        results[name] = fn()
    _print_table(results)


if __name__ == "__main__":
    main()