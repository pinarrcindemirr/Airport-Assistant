"""
Speech pipeline evaluation.

Transcribes every recording listed in data/audio/audio_queries.csv with
Whisper and compares the output against the known ground-truth text (what
the passenger actually said when the recording was made), reporting Word
Error Rate (WER) overall and broken down by recording condition (quiet /
noisy / fast / accented) - mirroring the condition-based breakdown used in
the text and image pipelines' evaluations, rather than a single averaged
number.

WER is implemented directly (word-level Levenshtein distance) rather than
pulling in the `jiwer` package, to keep this module dependency-light; the
formula is standard: (substitutions + deletions + insertions) / reference_length.

Run:  python -m backend.audio.evaluation
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from backend.audio.whisper_model import SpeechTranscriber

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT_DIR / "data" / "audio" / "audio_queries.csv"
DEFAULT_AUDIO_ROOT = ROOT_DIR / "data" / "audio" / "raw"


def _normalize(text: str) -> list[str]:
    """
    Normalise text before WER comparison: lowercase, drop apostrophes
    (so "I'm" -> "im", matching contraction-less ground-truth spelling as a
    single token rather than splitting into "i"/"m"), then strip remaining
    punctuation. Without this, Whisper's correctly-punctuated output (e.g.
    "I'm", "B12?") would be counted as wrong against plain ground-truth
    spelling (e.g. "im", "B12") purely due to formatting, not genuine content
    errors. This matches standard ASR evaluation practice (WER is a content
    metric, not a punctuation/capitalisation metric).
    """
    text = text.lower().replace("'", "")
    return re.findall(r"[a-z0-9]+", text)


def _word_error_rate(reference: str, hypothesis: str) -> float:
    """
    Word-level Levenshtein distance / len(reference words), computed on
    normalised tokens (see _normalize). Standard WER definition used to
    report Whisper transcription quality (brief's "basic Word Error Rate
    where possible").
    """
    ref = _normalize(reference)
    hyp = _normalize(hypothesis)
    n, m = len(ref), len(hyp)
    if n == 0:
        return 0.0 if m == 0 else 1.0

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[n][m] / n


def load_manifest(path: str | Path = DEFAULT_MANIFEST) -> list[dict]:
    """Read the recording manifest with pandas; returned as list[dict] for existing callers."""
    return pd.read_csv(path, dtype=str, keep_default_na=False).to_dict(orient="records")


def evaluate(transcriber: SpeechTranscriber, manifest_path: str | Path = DEFAULT_MANIFEST,
             audio_root: str | Path = DEFAULT_AUDIO_ROOT) -> tuple[dict, list[dict]]:
    rows = load_manifest(manifest_path)
    results = []

    for row in rows:
        audio_path = Path(audio_root) / row["filename"]
        transcription = transcriber.transcribe_file_with_confidence(audio_path)
        hypothesis = transcription.text
        wer = _word_error_rate(row["expected_text"], hypothesis)

        results.append({
            "audio_id": row["audio_id"],
            "intent": row["intent"],
            "condition": row["condition"],
            "expected_text": row["expected_text"],
            "transcribed_text": hypothesis,
            "wer": wer,
            "avg_logprob": transcription.avg_logprob,
            "no_speech_prob": transcription.no_speech_prob,
            "answered": transcription.answered,
        })

    df = pd.DataFrame(results)
    summary = df.groupby("condition")["wer"].mean().to_dict()  # per-condition mean WER
    summary["overall"] = df["wer"].mean()
    return summary, results


def _print_report(summary: dict, results: list[dict]) -> None:
    df = pd.DataFrame(results)

    print("Speech pipeline evaluation (Word Error Rate - lower is better)\n")
    for cond, wer in summary.items():
        if cond == "overall":
            continue
        n = (df["condition"] == cond).sum()
        print(f"  {cond:<10} WER={wer:.3f}  (n={n})")
    print(f"  {'overall':<10} WER={summary['overall']:.3f}  (n={len(df)})")

    correct = df[df["wer"] == 0]
    incorrect = df[df["wer"] > 0]
    print(f"\nConfidence distribution (avg_logprob - needed to calibrate AUDIO_CONFIDENCE_THRESHOLD):")
    for label, group in (("correct  ", correct), ("incorrect", incorrect)):
        if group.empty:
            print(f"  {label} predictions: n=0")
            continue
        print(f"  {label} predictions: n={len(group):<3} min={group['avg_logprob'].min():.3f}  "
              f"mean={group['avg_logprob'].mean():.3f}  max={group['avg_logprob'].max():.3f}")

    print("\nPer-example transcriptions:")
    for r in results:
        flag = "OK  " if r["wer"] == 0 else f"WER={r['wer']:.2f}"
        answered_flag = "answered" if r["answered"] else "ABSTAIN"
        print(f"  [{r['condition']:<8}] {flag}  [{answered_flag}]  "
              f"logprob={r['avg_logprob']:.3f}  no_speech={r['no_speech_prob']:.3f}")
        print(f"              expected:    \"{r['expected_text']}\"")
        print(f"              transcribed: \"{r['transcribed_text']}\"")


def main() -> None:
    transcriber = SpeechTranscriber()
    summary, results = evaluate(transcriber)
    _print_report(summary, results)


if __name__ == "__main__":
    main()