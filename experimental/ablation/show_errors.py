"""
Error analysis for the text retrieval pipeline.

The ablation gives aggregate numbers (hit@1 = 0.875 etc.); this script shows
the individual failures behind them, which is what the report's error-analysis
section needs. It prints three groups:

  1. MISSED@1  : in-scope queries where the top result was wrong
                 (shows the full top-K so we can see if the right record was
                 close, and whether the confidence threshold would have caught it).
  2. LEAKED    : out-of-scope queries that scored ABOVE the threshold, i.e. the
                 system confidently returned a record for a question it cannot
                 actually answer (false positive for the handover mechanism).
  3. NEAR-MISS : correct top-1 answers whose score is within MARGIN of the
                 threshold - queries that are answered correctly today but
                 would start abstaining if the threshold were raised. This is
                 the cost side of any "raise the threshold" fix for group 2.

Run from project root:  python -m experimental.ablation.show_errors
"""

from __future__ import annotations

from backend.utils.config import CONFIDENCE_THRESHOLD, TOP_K
from backend.text.embeddings import TextEmbedder
from backend.text.retrieval import TextRetriever
from experimental.ablation.run_keywords_ablation import load_queries

MARGIN = 0.05  # how close to the threshold counts as a near-miss


def main() -> None:
    embedder = TextEmbedder()
    # Analyse the variant we actually chose after the ablation: no keywords.
    retriever = TextRetriever(embedder, include_keywords=False)
    queries = load_queries()

    missed, leaked, near_miss = [], [], []

    for q in queries:
        cands = retriever.search(q["query"], top_k=TOP_K)
        top = cands[0]
        if q["in_scope"]:
            if top.record_id not in q["acceptable_set"]:
                missed.append((q, cands))
            elif top.score < CONFIDENCE_THRESHOLD + MARGIN:
                near_miss.append((q, top))
        else:
            if top.score >= CONFIDENCE_THRESHOLD:
                leaked.append((q, cands))

    print(f"threshold={CONFIDENCE_THRESHOLD}  top_k={TOP_K}  variant=no_keywords\n")

    print(f"=== 1. MISSED@1 - wrong top result ({len(missed)}) ===")
    for q, cands in missed:
        print(f"\n  [{q['query_id']}|{q['difficulty']}] \"{q['query']}\"")
        print(f"    expected any of : {sorted(q['acceptable_set'])}")
        for rank, c in enumerate(cands, 1):
            mark = "<- correct" if c.record_id in q["acceptable_set"] else ""
            print(f"    top{rank}: {c.score:.3f}  {c.record_id} {mark}")

    print(f"\n=== 2. LEAKED - out-of-scope but answered ({len(leaked)}) ===")
    for q, cands in leaked:
        print(f"\n  [{q['query_id']}] \"{q['query']}\"  (intent: {q['intent']})")
        for rank, c in enumerate(cands, 1):
            print(f"    top{rank}: {c.score:.3f}  {c.record_id}")

    print(f"\n=== 3. NEAR-MISS - correct but within {MARGIN} of threshold ({len(near_miss)}) ===")
    for q, top in near_miss:
        print(f"  [{q['query_id']}|{q['difficulty']}] \"{q['query']}\" -> {top.record_id}  score={top.score:.3f}")

    print("\nHow to read this:")
    print("  - Group 1 drives hit@1; look for a pattern (elliptical wording, wrong category, T1/T2 ambiguity).")
    print("  - Group 2 is the handover false-positive; raising the threshold fixes it")
    print("    ONLY if group 3 stays empty - otherwise you trade leaks for lost answers.")


if __name__ == "__main__":
    main()
