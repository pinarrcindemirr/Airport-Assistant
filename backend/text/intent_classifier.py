"""
Intent classification for passenger text queries.

Fills the brief's "intent classification" requirement (Model Design > Speech
and Text Models) and its associated metrics requirement (Training &
Evaluation > "precision, recall, F1 score, or a confusion matrix").

Architecture: a LINEAR PROBE - frozen MiniLM sentence embeddings (the exact
same TextEmbedder used by retrieval, unchanged) with a scikit-learn logistic
regression classifier on top. This respects the project's frozen-model
principle: the transformer itself is never fine-tuned; only a lightweight,
classical classifier is fitted on its fixed outputs. Linear probing is the
standard way to evaluate/exploit frozen representations, and is far lighter
than the brief's optional "fine-tuned DistilBERT" while still producing
the full set of classification metrics the brief asks for.

Evaluation: Leave-One-Out cross-validation (LOO-CV), not a single
train/test split. Reason: the 50-query dataset spans 17 intent classes,
several of which have only ONE example (find_security, find_pharmacy,
find_smoking) - a stratified 80/20 split is mathematically impossible for
those classes (they cannot appear on both sides), and an unstratified split
would silently drop them from either training or testing. LOO-CV handles
this correctly: each query is predicted exactly once, by a model trained on
the other 49, so every class contributes to the confusion matrix and no
query is ever predicted by a model that saw it during training.

Run:  python -m backend.text.intent_classifier
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from backend.text.embeddings import TextEmbedder

# NearestCentroid computes a within-class standard deviation even though it's
# unused unless shrink_threshold is set (which we don't use). When a LOO fold
# leaves only 1 training example for a class (several classes here have just
# 1-3 total examples - see module docstring), that std-dev is exactly zero
# and sklearn emits a UserWarning + a harmless divide-by-zero RuntimeWarning.
# This is an expected, understood consequence of the dataset's class
# scarcity (the same scarcity this comparison is designed to investigate),
# not a bug - suppressed here so the actual results aren't buried in noise.
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.neighbors._nearest_centroid")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.neighbors._nearest_centroid")

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_QUERIES = ROOT_DIR / "data" / "text" / "airport_queries.csv"

# Two classifiers on the SAME frozen embeddings, compared side by side:
#   - logistic_regression: fits a weighted hyperplane per class - flexible,
#     but needs enough examples per class to estimate those weights reliably.
#   - nearest_centroid: assigns each query to whichever class's MEAN
#     embedding is closest - only ONE number per dimension per class to
#     estimate (the centroid), so it degrades far more gracefully when a
#     class has only 1-3 examples, which is exactly this dataset's situation
#     (see module docstring). Included specifically to test whether the
#     logistic regression's low score (see report) is a data-scarcity
#     problem rather than a modelling problem - if NearestCentroid also
#     performs poorly, scarcity is the dominant explanation; if it does much
#     better, the classifier choice itself matters more than expected.
CLASSIFIERS = {
    "logistic_regression": lambda: LogisticRegression(max_iter=2000),
    "nearest_centroid": lambda: NearestCentroid(),
}


def load_dataset(path: str | Path = DEFAULT_QUERIES) -> tuple[list[str], list[str]]:
    """Load (queries, intent labels) from the query CSV."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return df["query"].tolist(), df["intent"].tolist()


def embed_queries(embedder: TextEmbedder, queries: list[str]) -> np.ndarray:
    """Encode all queries into frozen MiniLM embeddings - the classifier's input features."""
    return embedder.encode(queries)


def loo_cross_validate(features: np.ndarray, labels: list[str], make_classifier) -> pd.DataFrame:
    """
    Leave-One-Out CV: for each query, fit a fresh classifier (from
    make_classifier()) on the other n-1 queries' embeddings and predict the
    held-out one. Returns a DataFrame with one row per query: (true_label,
    predicted_label). Works with any scikit-learn classifier implementing
    fit()/predict(), so the same function drives both models in CLASSIFIERS.
    """
    labels_arr = np.asarray(labels)
    predictions = []

    for train_idx, test_idx in LeaveOneOut().split(features):
        clf = make_classifier()
        clf.fit(features[train_idx], labels_arr[train_idx])
        predictions.append(clf.predict(features[test_idx])[0])

    return pd.DataFrame({"true": labels_arr, "predicted": predictions})


def evaluate(results: pd.DataFrame) -> dict:
    """Compute the metrics the brief asks for: accuracy, macro precision/recall/F1."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        results["true"], results["predicted"], average="macro", zero_division=0
    )
    return {
        "accuracy": accuracy_score(results["true"], results["predicted"]),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "n": len(results),
    }


def _print_report(results: pd.DataFrame, metrics: dict, name: str = "classifier") -> None:
    print("Intent classification (frozen MiniLM embeddings, LOO-CV)\n")
    print(f"  accuracy         : {metrics['accuracy']:.3f}")
    print(f"  macro precision  : {metrics['macro_precision']:.3f}")
    print(f"  macro recall     : {metrics['macro_recall']:.3f}")
    print(f"  macro F1         : {metrics['macro_f1']:.3f}")
    print(f"  n queries        : {metrics['n']}\n")

    print("Per-class report:")
    print(classification_report(results["true"], results["predicted"], zero_division=0))

    labels_sorted = sorted(results["true"].unique())
    cm = confusion_matrix(results["true"], results["predicted"], labels=labels_sorted)
    cm_df = pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted)
    print("Confusion matrix (rows = true, columns = predicted):")
    print(cm_df.to_string())

    # Save the confusion matrix as CSV so the report (and a future
    # matplotlib heatmap) can use it without re-running the evaluation.
    out_path = ROOT_DIR / "data" / "text" / f"intent_confusion_matrix_{name}.csv"
    cm_df.to_csv(out_path)
    print(f"\nConfusion matrix saved to {out_path}")

    misclassified = results[results["true"] != results["predicted"]]
    print(f"\nMisclassified queries ({len(misclassified)}):")
    for _, row in misclassified.iterrows():
        print(f"  true={row['true']:<18} predicted={row['predicted']}")


def _print_comparison(all_metrics: dict[str, dict]) -> None:
    print("Classifier comparison (same frozen embeddings, same LOO-CV splits)\n")
    print(f"{'classifier':<22}{'accuracy':>10}{'macro_P':>10}{'macro_R':>10}{'macro_F1':>10}")
    print("-" * 62)
    for name, m in all_metrics.items():
        print(f"{name:<22}{m['accuracy']:>10.3f}{m['macro_precision']:>10.3f}"
              f"{m['macro_recall']:>10.3f}{m['macro_f1']:>10.3f}")
    print()


def main() -> None:
    queries, labels = load_dataset()
    embedder = TextEmbedder()
    features = embed_queries(embedder, queries)

    all_metrics = {}
    all_results = {}
    for name, make_classifier in CLASSIFIERS.items():
        results = loo_cross_validate(features, labels, make_classifier)
        all_results[name] = results
        all_metrics[name] = evaluate(results)

    _print_comparison(all_metrics)

    for name, results in all_results.items():
        print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
        _print_report(results, all_metrics[name], name=name)


if __name__ == "__main__":
    main()