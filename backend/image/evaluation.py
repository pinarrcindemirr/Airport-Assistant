"""
Vision pipeline evaluation.

Tests whether each generated airport sign image (the clean reference plus
the three degraded query conditions produced by data/images/generate_mockups.py)
retrieves its own KB record when matched against every record's CLIP-encoded
visual_description. Reports top-1 and top-3 accuracy overall AND broken down
by condition (clean/angle/dark/blur) - mirroring how the text pipeline's
evaluation broke results down by query difficulty (easy/hard) rather than
reporting one averaged number, so degradation robustness is visible directly.

Run:  python -m backend.image.evaluation
"""

from __future__ import annotations

from collections import defaultdict

import torch

from backend.image.embeddings import ImageEmbedder
from backend.image.retrieval import ImageRetriever, IMAGE_TOP_K
from backend.image.preprocess import build_dataloader

CONDITION_ORDER = ["clean", "angle", "dark", "blur"]


def evaluate(retriever: ImageRetriever, batch_size: int = 8) -> tuple[dict, dict]:
    """
    Run every image in data/images/image_labels.csv through the retriever in
    batches, compare the top prediction against the image's own record_id
    (ground truth: each generated image is a photo of exactly one record),
    and aggregate accuracy per condition plus a list of concrete examples.
    """
    loader = build_dataloader(batch_size=batch_size)
    per_condition = defaultdict(lambda: {"top1": 0, "top3": 0, "n": 0})
    examples = {"correct": [], "incorrect": []}

    for tensors, records in loader:
        image_embs = retriever.embedder.encode_images(tensors)          # (B, d)
        sims = image_embs @ retriever.text_matrix.T                     # (B, n_records)

        for row, rec in zip(sims, records):
            order = torch.argsort(row, descending=True)[:IMAGE_TOP_K]
            ranked_ids = [retriever.ids[i] for i in order.tolist()]
            top1_hit = ranked_ids[0] == rec.record_id
            top3_hit = rec.record_id in ranked_ids

            bucket = per_condition[rec.condition]
            bucket["top1"] += int(top1_hit)
            bucket["top3"] += int(top3_hit)
            bucket["n"] += 1

            entry = {
                "filename": rec.filename, "record_id": rec.record_id,
                "condition": rec.condition, "predicted": ranked_ids[0],
                "score": row[order[0]].item(),
            }
            (examples["correct"] if top1_hit else examples["incorrect"]).append(entry)

    summary = {
        cond: {
            "top1_accuracy": v["top1"] / v["n"],
            "top3_accuracy": v["top3"] / v["n"],
            "n": v["n"],
        }
        for cond, v in per_condition.items()
    }
    overall_n = sum(v["n"] for v in per_condition.values())
    summary["overall"] = {
        "top1_accuracy": sum(v["top1"] for v in per_condition.values()) / overall_n,
        "top3_accuracy": sum(v["top3"] for v in per_condition.values()) / overall_n,
        "n": overall_n,
    }
    return summary, examples


def _score_stats(scores: list[float]) -> str:
    if not scores:
        return "n=0"
    return f"n={len(scores):<3} min={min(scores):.3f}  mean={sum(scores)/len(scores):.3f}  max={max(scores):.3f}"


def _print_report(summary: dict, examples: dict) -> None:
    print(f"Vision pipeline evaluation (top_k={IMAGE_TOP_K})\n")
    print(f"{'condition':<12}{'top1_acc':>10}{'top3_acc':>10}{'n':>6}")
    print("-" * 40)
    for cond in CONDITION_ORDER + ["overall"]:
        if cond not in summary:
            continue
        s = summary[cond]
        print(f"{cond:<12}{s['top1_accuracy']:>10.3f}{s['top3_accuracy']:>10.3f}{s['n']:>6}")

    correct_scores = [e["score"] for e in examples["correct"]]
    incorrect_scores = [e["score"] for e in examples["incorrect"]]
    print(f"\nTop-1 score distribution (needed to calibrate IMAGE_CONFIDENCE_THRESHOLD):")
    print(f"  correct   predictions: {_score_stats(correct_scores)}")
    print(f"  incorrect predictions: {_score_stats(incorrect_scores)}")

    print(f"\nIncorrect top-1 examples ({len(examples['incorrect'])} total, showing up to 15):")
    for e in examples["incorrect"][:15]:
        print(f"  [{e['condition']:<6}] {e['record_id']:<22} -> predicted {e['predicted']:<22} score={e['score']:.3f}")


def main() -> None:
    embedder = ImageEmbedder()
    retriever = ImageRetriever(embedder)
    summary, examples = evaluate(retriever)
    _print_report(summary, examples)


if __name__ == "__main__":
    main()