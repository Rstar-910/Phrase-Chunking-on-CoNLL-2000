"""
main.py
-------
Main pipeline for the Hybrid Rule-Based + ML Phrase Chunking system.

Steps:
  1. Load CoNLL-2000 data
  2. Run rule-based chunker on train and test
  3. Extract features for ML model
  4. Train Logistic Regression correction model
  5. Predict corrected chunks on test set
  6. Evaluate both systems
  7. Print comparison table
"""

import time

from data_loader import load_conll2000
from rule_chunker import predict as rule_predict
from feature_engineering import extract_features
from ml_model import HybridChunkCorrector
from evaluation import (
    evaluate_model,
    print_detailed_report,
    print_comparison_table,
)
from joblib import dump
from datetime import datetime


def main():
    start = time.time()

    # ── 1. Load data ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 1: Loading CoNLL-2000 dataset")
    print("=" * 60)
    train_data, test_data = load_conll2000(
        hf_name="eriktks/conll2000",
        local_path="conll2000_local",
    )

    # ── 2. Rule-based predictions ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 2: Running rule-based chunker")
    print("=" * 60)
    print("[main] Predicting on training set …")
    train_rule_preds = rule_predict(train_data)
    print("[main] Predicting on test set …")
    test_rule_preds = rule_predict(test_data)
    print(f"[main] Rule-based predictions generated for "
          f"{len(train_rule_preds)} train / {len(test_rule_preds)} test sentences.")

    # ── 3. Feature extraction ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 3: Extracting features for ML model")
    print("=" * 60)
    train_features, train_labels, train_lengths = extract_features(
        train_data, train_rule_preds
    )
    test_features, test_labels, test_lengths = extract_features(
        test_data, test_rule_preds
    )
    print(f"[main] Training tokens: {len(train_features):,}")
    print(f"[main] Test tokens:     {len(test_features):,}")

    # ── 4. Train ML correction model ──────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 4: Training hybrid correction model")
    print("=" * 60)
    # If using LogisticRegression backend:
    # model = HybridChunkCorrector(max_iter=1000, C=1.0)
    # model.fit(train_features, train_labels)

    # Using CRF sequence model
    model = HybridChunkCorrector()  # CRF hyperparams are in __init__
    model.fit(train_features, train_labels, train_lengths)

    # ── 5. Predict with hybrid model ──────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 5: Generating hybrid predictions on test set")
    print("=" * 60)
    hybrid_preds = model.predict_sentences(test_features, test_lengths)
    print(f"[main] Hybrid predictions generated for {len(hybrid_preds)} sentences.")

    # ── 6. Evaluate ───────────────────────────────────────────────────
    # Reconstruct sentence-level true labels for seqeval
    true_labels = [sent["chunk_tags"] for sent in test_data]

    print("\n" + "=" * 60)
    print("  STEP 6: Evaluation")
    print("=" * 60)

    # Rule-based evaluation
    rule_results = evaluate_model(true_labels, test_rule_preds, "Rule-Based Chunker")
    print_detailed_report(true_labels, test_rule_preds, "Rule-Based Chunker")

    # Hybrid evaluation
    hybrid_results = evaluate_model(true_labels, hybrid_preds, "Hybrid (ML) Chunker")
    print_detailed_report(true_labels, hybrid_preds, "Hybrid (ML) Chunker")

    # ── 7. Comparison table ───────────────────────────────────────────
    print_comparison_table({
        "Rule-Based": rule_results,
        "Hybrid (ML)": hybrid_results,
    })

    # ── 8. Save results and model ─────────────────────────────────────
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Identify hybrid backend (LogisticRegression vs CRF, etc.)
    if hasattr(model, "crf"):
        backend = "CRF"
    else:
        backend = "LogisticRegression"

    # Save metrics to a text file with backend in the name and content
    results_path = f"results_{backend}_{timestamp}.txt"
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("PHRASE CHUNKING RESULTS\n")
        f.write(f"Timestamp      : {timestamp}\n")
        f.write(f"Hybrid backend : {backend}\n")
        f.write("Dataset        : CoNLL-2000 (local JSON)\n\n")

        f.write("Rule-Based Chunker:\n")
        f.write(f"  Accuracy : {rule_results['accuracy']:.4f}\n")
        f.write(f"  Precision: {rule_results['precision']:.4f}\n")
        f.write(f"  Recall   : {rule_results['recall']:.4f}\n")
        f.write(f"  F1       : {rule_results['f1']:.4f}\n\n")

        f.write("Hybrid (ML) Chunker:\n")
        f.write(f"  Accuracy : {hybrid_results['accuracy']:.4f}\n")
        f.write(f"  Precision: {hybrid_results['precision']:.4f}\n")
        f.write(f"  Recall   : {hybrid_results['recall']:.4f}\n")
        f.write(f"  F1       : {hybrid_results['f1']:.4f}\n")

    print(f"[main] Saved metrics to {results_path}")

    # Save trained hybrid model with backend in the filename
    model_path = f"hybrid_chunker_{backend}_{timestamp}.joblib"
    dump(model, model_path)
    print(f"[main] Saved trained model to {model_path}")

    elapsed = time.time() - start
    print(f"[main] Total time: {elapsed:.1f}s")
    print("[main] Done.")


if __name__ == "__main__":
    main()
