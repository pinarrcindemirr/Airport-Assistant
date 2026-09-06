from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

OCR_ANSWER_THRESHOLD = 0.55  


@dataclass
class OCRResult:
    """Outcome of reading text from an airport sign image."""
    text: str             
    confidence: float       
    num_detections: int
    answered: bool        


class SignTextReader:
    """read text off airport sign/gate-board photos."""

    def __init__(self, languages: list[str] | None = None, reader=None):
        if reader is not None:
            self.reader = reader
        else:
            import easyocr
            self.reader = easyocr.Reader(languages or ["en"], gpu=False)

    def read(self, image_path: str | Path) -> OCRResult:
        detections = self.reader.readtext(str(image_path))  # list of (bbox, text, conf)

        if not detections:
            return OCRResult(text="", confidence=0.0, num_detections=0, answered=False)

        texts = [d[1] for d in detections]
        confs = [d[2] for d in detections]
        combined_text = " ".join(texts).strip()
        mean_conf = sum(confs) / len(confs)

        return OCRResult(
            text=combined_text,
            confidence=mean_conf,
            num_detections=len(detections),
            answered=bool(combined_text) and mean_conf >= OCR_ANSWER_THRESHOLD,
        )


if __name__ == "__main__":
    import csv
    from backend.utils.config import ROOT_DIR

    reader = SignTextReader()
    manifest = ROOT_DIR / "data" / "images" / "image_labels.csv"
    rows = list(csv.DictReader(open(manifest, encoding="utf-8")))

    for row in rows[:10]:
        path = ROOT_DIR / row["filename"]
        result = reader.read(path)
        print(f"{row['filename']:<40} -> text={result.text!r:40} "
              f"conf={result.confidence:.3f}  answered={result.answered}")