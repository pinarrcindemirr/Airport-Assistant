"""
Quick manual smoke test for the fusion layer, using real models.
Run from the project root:  python test_fusion.py
"""

from backend.assistant import process_query

test_queries = [
    "where is gate B12",
    "where do I pick up my luggage",
    "can I rest somewhere quiet before my flight",
    "is my flight on time",
    "coffee near gate A",
]

print("=== Text-only, multiple queries (easy/hard/out-of-scope) ===\n")
for q in test_queries:
    r = process_query(text=q)
    record_id = r.record["id"] if r.record else None
    print(f"query: {q!r}")
    print(f"  answered={r.answered}  confidence={r.confidence:.3f}  record={record_id}")
    print()

print("=== Voice-only ===")
r = process_query(audio_path="data/audio/raw/a01.m4a")
print("answered:", r.answered, "confidence:", r.confidence, "record:", r.record["id"] if r.record else None)
print()

print("=== Image-only ===")
r = process_query(image_path="data/images/gate/gate_b12_v1.png")
print("answered:", r.answered, "confidence:", r.confidence, "record:", r.record["id"] if r.record else None)
print()

print("=== Image + Text together, SAME record (expect agreement=True) ===")
r = process_query(text="where is gate B12", image_path="data/images/gate/gate_b12_v1.png")
print("answered:", r.answered, "confidence:", r.confidence, "record:", r.record["id"] if r.record else None,
      "agreement:", r.agreement)
for m in r.modality_results:
    print(f"   [{m.modality}] answered={m.answered} confidence={m.confidence:.3f} record_id={m.record_id}")
print()

print("=== Image + Text together, DIFFERENT records (expect agreement=False) ===")
r = process_query(text="where do I pick up my luggage", image_path="data/images/gate/gate_b12_v1.png")
print("answered:", r.answered, "confidence:", r.confidence, "record:", r.record["id"] if r.record else None,
      "agreement:", r.agreement)
for m in r.modality_results:
    print(f"   [{m.modality}] answered={m.answered} confidence={m.confidence:.3f} record_id={m.record_id}")
print()

print("=== Voice + Image together, SAME record (expect agreement=True) ===")
r = process_query(audio_path="data/audio/raw/a01.m4a", image_path="data/images/gate/gate_b12_v1.png")
print("answered:", r.answered, "confidence:", r.confidence, "record:", r.record["id"] if r.record else None,
      "agreement:", r.agreement)
for m in r.modality_results:
    print(f"   [{m.modality}] answered={m.answered} confidence={m.confidence:.3f} record_id={m.record_id}")