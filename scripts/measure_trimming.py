from pathlib import Path

from backend.audio.preprocess import load_audio, preprocess_audio, WHISPER_SAMPLE_RATE

AUDIO_DIR = Path("data") / "audio" / "raw"


def main() -> None:
    files = sorted(AUDIO_DIR.glob("*.m4a"))
    if not files:
        print(f"No .m4a files found in {AUDIO_DIR.resolve()}")
        return

    print(f"{'file':<10}{'raw (s)':>10}{'trimmed (s)':>14}{'removed (s)':>14}{'removed %':>12}")
    print("-" * 60)

    total_raw = 0.0
    total_trimmed = 0.0
    for f in files:
        raw = load_audio(f)
        # trim only (normalize doesn't change length) to isolate the trim effect
        processed = preprocess_audio(f, trim=True, normalize=False)
        raw_s = len(raw) / WHISPER_SAMPLE_RATE
        trim_s = len(processed) / WHISPER_SAMPLE_RATE
        removed = raw_s - trim_s
        pct = (removed / raw_s * 100) if raw_s > 0 else 0.0
        total_raw += raw_s
        total_trimmed += trim_s
        print(f"{f.name:<10}{raw_s:>10.2f}{trim_s:>14.2f}{removed:>14.2f}{pct:>11.1f}%")

    print("-" * 60)
    tot_removed = total_raw - total_trimmed
    tot_pct = (tot_removed / total_raw * 100) if total_raw > 0 else 0.0
    print(f"{'TOTAL':<10}{total_raw:>10.2f}{total_trimmed:>14.2f}{tot_removed:>14.2f}{tot_pct:>11.1f}%")
    print(f"\nMean removed per clip: {tot_removed / len(files):.2f}s")


if __name__ == "__main__":
    main()