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
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.neighbors._nearest_centroid")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.neighbors._nearest_centroid")

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_QUERIES = ROOT_DIR / "data" / "text" / "airport_queries.csv"

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
    """Leave-One-Out cross-validation on the given features and labels."""
    labels_arr = np.asarray(labels)
    predictions = []

    for train_idx, test_idx in LeaveOneOut().split(features):
        clf = make_classifier()
        clf.fit(features[train_idx], labels_arr[train_idx])
        predictions.append(clf.predict(features[test_idx])[0])

    return pd.DataFrame({"true": labels_arr, "predicted": predictions})


def evaluate(results: pd.DataFrame) -> dict:
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