from __future__ import annotations

import pandas as pd
import torch

from backend.image.embeddings import ImageEmbedder
from backend.image.retrieval import ImageRetriever, IMAGE_TOP_K
from backend.image.preprocess import build_dataloader

CONDITION_ORDER = ["clean", "angle", "dark", "blur"]


def evaluate(retriever: ImageRetriever, batch_size: int = 8) -> tuple[dict, dict]:

    loader = build_dataloader(batch_size=batch_size)
    rows = []

    for tensors, records in loader:
        image_embs = retriever.embedder.encode_images(tensors)      
        sims = image_embs @ retriever.text_matrix.T                 

        for row, rec in zip(sims, records):
            order = torch.argsort(row, descending=True)[:IMAGE_TOP_K]
            ranked_ids = [retriever.ids[i] for i in order.tolist()]
            rows.append({
                "filename": rec.filename,
                "record_id": rec.record_id,
                "condition": rec.condition,
                "predicted": ranked_ids[0],
                "score": row[order[0]].item(),
                "top1_hit": ranked_ids[0] == rec.record_id,
                "top3_hit": rec.record_id in ranked_ids,
            })

    df = pd.DataFrame(rows)

    grouped = df.groupby("condition").agg(
        top1_accuracy=("top1_hit", "mean"),
        top3_accuracy=("top3_hit", "mean"),
        n=("top1_hit", "size"),
    )
    summary = {cond: row.to_dict() for cond, row in grouped.iterrows()}
    summary["overall"] = {
        "top1_accuracy": df["top1_hit"].mean(),
        "top3_accuracy": df["top3_hit"].mean(),
        "n": len(df),
    }

    examples = {
        "correct": df[df["top1_hit"]].to_dict(orient="records"),
        "incorrect": df[~df["top1_hit"]].to_dict(orient="records"),
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
        print(f"{cond:<12}{s['top1_accuracy']:>10.3f}{s['top3_accuracy']:>10.3f}{int(s['n']):>6}")

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