"""
evaluation.py
-------------
Evaluation utilities for phrase chunking models.
Provides token-level accuracy and chunk-level precision, recall, F1 via seqeval.
"""

from sklearn.metrics import accuracy_score
from seqeval.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


def evaluate_model(true_labels, pred_labels, model_name="Model"):
    """
    Evaluate predictions against ground truth.

    Parameters
    ----------
    true_labels : list[list[str]]   – sentence-level true BIO tags
    pred_labels : list[list[str]]   – sentence-level predicted BIO tags
    model_name  : str               – label for printing

    Returns
    -------
    results : dict
        Keys: 'accuracy', 'precision', 'recall', 'f1'
    """
    # ── token-level accuracy ──────────────────────────────────────────
    flat_true = [tag for sent in true_labels for tag in sent]
    flat_pred = [tag for sent in pred_labels for tag in sent]
    acc = accuracy_score(flat_true, flat_pred)

    # ── chunk-level metrics (seqeval) ─────────────────────────────────
    prec = precision_score(true_labels, pred_labels)
    rec  = recall_score(true_labels, pred_labels)
    f1   = f1_score(true_labels, pred_labels)

    results = {
        "accuracy":  acc,
        "precision": prec,
        "recall":    rec,
        "f1":        f1,
    }

    # ── pretty print ──────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"  {model_name} — Evaluation Results")
    print(f"{'─' * 60}")
    print(f"  Token-level Accuracy : {acc:.4f}")
    print(f"  Chunk-level Precision: {prec:.4f}")
    print(f"  Chunk-level Recall   : {rec:.4f}")
    print(f"  Chunk-level F1       : {f1:.4f}")
    print(f"{'─' * 60}")

    return results


def print_detailed_report(true_labels, pred_labels, model_name="Model"):
    """Print a per-chunk-type classification report (seqeval)."""
    print(f"\n{'═' * 60}")
    print(f"  {model_name} — Detailed Classification Report")
    print(f"{'═' * 60}")
    print(classification_report(true_labels, pred_labels))


def print_comparison_table(results_dict):
    """
    Print a side-by-side comparison table.

    Parameters
    ----------
    results_dict : dict[str, dict]
        e.g. {"Rule-Based": {...}, "Hybrid (ML)": {...}}
    """
    print()
    print("=" * 64)
    print("           PHRASE CHUNKING RESULTS COMPARISON")
    print("=" * 64)
    print(f"| {'Model':<16s} | {'Precision':>9s} | {'Recall':>7s} | {'F1':>7s} | {'Accuracy':>8s} |")
    print(f"|{'-'*18}|{'-'*11}|{'-'*9}|{'-'*9}|{'-'*10}|")
    for name, r in results_dict.items():
        print(f"| {name:<16s} | {r['precision']:>9.4f} | {r['recall']:>7.4f} | {r['f1']:>7.4f} | {r['accuracy']:>8.4f} |")
    print("=" * 64)
    print()
