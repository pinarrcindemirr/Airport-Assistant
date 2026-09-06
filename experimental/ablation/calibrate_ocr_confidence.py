import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import csv
from collections import defaultdict
from pathlib import Path

from backend.image.ocr_reader import SignTextReader
from backend.utils.config import ROOT_DIR

reader = SignTextReader()
manifest = ROOT_DIR / "data" / "images" / "image_labels.csv"
rows = list(csv.DictReader(open(manifest, encoding="utf-8")))

by_category = defaultdict(list)

for row in rows:
    path = ROOT_DIR / row["filename"]
    result = reader.read(path)
    by_category[row["category"]].append(result.confidence if result.num_detections else 0.0)
    print(f"{row['filename']:<40} cat={row['category']:<15} "
          f"text={result.text!r:35} conf={result.confidence:.3f} n_det={result.num_detections}")

print("\n--- Per-category mean confidence ---")
for cat, confs in by_category.items():
    print(f"{cat:<20} mean={sum(confs)/len(confs):.3f}  n={len(confs)}")