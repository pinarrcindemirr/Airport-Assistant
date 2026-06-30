"""
Keywords ablation for the text retrieval pipeline.

Question being answered: does appending the curated `keywords` list to each
record's embedding text help or hurt retrieval? SentenceTransformer is trained
on natural sentences, so a comma-separated keyword tail might add useful signal
or just add noise. We measure it instead of guessing.

How: build the retriever twice from the SAME code path, once with
include_keywords=False and once True (the flag is forwarded all the way down to
kb.build_text_for_embedding), then score both against data/text/airport_queries.csv.

Metrics:
  In-scope queries (have a correct record):
    - hit@1      : top result is an acceptable record
    - recall@k   : an acceptable record appears in the top-K
    - MRR        : mean reciprocal rank of the first acceptable record
  Out-of-scope queries (no record exists -> system should stay silent):
    - abstain_acc: fraction where the top score fell below CONFIDENCE_THRESHOLD
  Plus an easy-vs-hard breakdown of hit@1, because keywords are most likely to
  matter on the harder, less literal queries.

Run:  python -m experimental.ablation.run_keywords_ablation
"""

from __future__ import annotations

import csv
from pathlib import Path

from backend.utils.config import KB_DIR, CONFIDENCE_THRESHOLD, TOP_K
from backend.text.embeddings import TextEmbedder
from backend.text.retrieval import TextRetriever

QUERIES_PATH = KB_DIR.parent / "text" / "airport_queries.csv"


def load_queries(path: Path = QUERIES_PATH) -> list[dict]:
    """Read the query set; split acceptable_ids into a set ('none' -> empty)."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ids = row["acceptable_ids"].strip()
            row["acceptable_set"] = set() if ids == "none" else set(ids.split("|"))
            row["in_scope"] = bool(row["acceptable_set"])
            rows.append(row)
    return rows


def evaluate(retriever: TextRetriever, queries: list[dict], top_k: int = TOP_K) -> dict:
    """Score one retriever variant over all queries."""
    in_scope = [q for q in queries if q["in_scope"]]
    oos = [q for q in queries if not q["in_scope"]]

    hit1 = recall_k = rr_sum = 0
    hit1_by_diff = {"easy": [0, 0], "hard": [0, 0]}  # [hits, total]

    for q in in_scope:
        cands = retriever.search(q["query"], top_k=top_k)
        ranked_ids = [c.record_id for c in cands]
        acc = q["acceptable_set"]

        is_hit1 = ranked_ids[0] in acc
        hit1 += int(is_hit1)
        recall_k += int(any(rid in acc for rid in ranked_ids))
        rank = next((i + 1 for i, rid in enumerate(ranked_ids) if rid in acc), None)
        rr_sum += (1.0 / rank) if rank else 0.0

        bucket = hit1_by_diff[q["difficulty"]]
        bucket[0] += int(is_hit1)
        bucket[1] += 1

    abstain_ok = sum(1 for q in oos if retriever.answer(q["query"]).top_score < CONFIDENCE_THRESHOLD)

    n_in = len(in_scope)
    return {
        "n_in_scope": n_in,
        "n_oos": len(oos),
        "hit@1": hit1 / n_in,
        f"recall@{top_k}": recall_k / n_in,
        "MRR": rr_sum / n_in,
        "abstain_acc": abstain_ok / len(oos) if oos else float("nan"),
        "hit@1_easy": hit1_by_diff["easy"][0] / max(1, hit1_by_diff["easy"][1]),
        "hit@1_hard": hit1_by_diff["hard"][0] / max(1, hit1_by_diff["hard"][1]),
    }


def run(embedder: TextEmbedder, queries: list[dict] | None = None) -> dict:
    """Build both variants from the given embedder and evaluate each."""
    queries = queries or load_queries()
    results = {}
    for include_kw in (False, True):
        retriever = TextRetriever(embedder, include_keywords=include_kw)
        label = "with_keywords" if include_kw else "no_keywords"
        results[label] = evaluate(retriever, queries)
    return results


def _print_table(results: dict) -> None:
    metrics = ["hit@1", f"recall@{TOP_K}", "MRR", "abstain_acc", "hit@1_easy", "hit@1_hard"]
    a, b = results["no_keywords"], results["with_keywords"]
    print(f"\nKeywords ablation  (threshold={CONFIDENCE_THRESHOLD}, top_k={TOP_K})")
    print(f"in-scope queries: {a['n_in_scope']}   out-of-scope: {a['n_oos']}\n")
    print(f"{'metric':<14}{'no_keywords':>14}{'with_keywords':>16}{'delta':>10}")
    print("-" * 54)
    for m in metrics:
        d = b[m] - a[m]
        print(f"{m:<14}{a[m]:>14.3f}{b[m]:>16.3f}{d:>+10.3f}")
    winner = "with_keywords" if b["hit@1"] > a["hit@1"] else \
             "no_keywords" if a["hit@1"] > b["hit@1"] else "tie (use no_keywords, simpler)"
    print(f"\n-> hit@1 winner: {winner}")


def main() -> None:
    embedder = TextEmbedder()  # downloads all-MiniLM-L6-v2 on first run
    _print_table(run(embedder))


if __name__ == "__main__":
    main()
