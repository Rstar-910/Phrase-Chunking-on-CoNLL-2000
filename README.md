# 🧠 Advanced Phrase Chunking: A Survey of Neural & Hybrid Architectures

[![Research Report](https://img.shields.io/badge/Research-NLP-blueviolet)](https://github.com/Rstar-910/Phrase-Chunking-on-CoNLL-2000)
[![Dataset](https://img.shields.io/badge/Dataset-CoNLL--2000-green)](https://huggingface.co/datasets/eriktks/conll2000)
[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://www.python.org/)

**Author**: Rahul Soni

This repository contains a comprehensive survey and implementation of eight distinct systems for **Phrase Chunking (Shallow Parsing)**. We trace the evolutionary path from purely rule-based baselines to state-of-the-art **BiLSTM-CRF**, **Universal Transformer (UT)**, **Mamba SSMs**, and fine-tuned **RoBERTa** architectures.

## 📄 Abstract

This project reproduces and surpasses seminal benchmarks in neural sequence tagging. By integrating enriched hand-crafted features (56-dim feature vectors), structured decoding (CRF), and large-scale pretraining, we demonstrate that a RoBERTa-based Transformer-CRF achieves a new best F1 score of **96.44%**, significantly outperforming the historical CoNLL-2000 benchmark of 94.46%.

---

## 📊 Experimental Results

Evaluated on the official CoNLL-2000 test split (2,012 sentences).

### 1. Performance Comparison
| Chunker Architecture | Accuracy | Precision | Recall | F1 Score | Params (M) | Training Time |
|----------------------|----------|-----------|--------|----------|------------|---------------|
| **Rule-Based (Baseline)** | 82.55% | 72.69% | 78.37% | 75.42% | 0.00 | Instant |
| **Hybrid (LogReg)**  | 94.32% | 89.53% | 91.64% | 90.58% | ~0.50 | < 1 min |
| **Hybrid (CRF)**     | 95.03% | 92.51% | 92.29% | 92.40% | ~0.30 | < 5 min |
| **BiLSTM (No CRF)**  | 95.46% | 91.77% | 92.71% | 92.24% | ~2.20 | ~30 min |
| **BiLSTM-CRF (Paper-Style)** | 95.98% | 94.35% | 93.32% | **93.83%** | ~2.40 | ~60 min |
| **Mamba-130M**       | 94.51% | 88.41% | 90.56% | 89.47% | ~130.0 | ~2 hrs |
| **UT-CRF (Universal Transformer)** | 95.39% | 92.35% | 93.18% | **92.76%** | ~3.20 | ~1.7 hrs |
| **Transformer-CRF (RoBERTa)** | **97.64%** | **96.41%** | **96.48%** | **96.44%** | 124.66 | ~22 min |

### 2. Historical Benchmarks comparison
| System | F1 (%) |
|--------|--------|
| SVM Classifier [Kudo & Matsumoto, 2001] | 93.91 |
| Second-Order CRF [Sha & Pereira, 2003] | 94.30 |
| BI-LSTM-CRF (SENNA) [Huang et al., 2015] | 94.46 |
| **Our Transformer-CRF (RoBERTa)** | **96.44** |

---

## 🏗️ Architecture Deep-Dives

### 1. Hybrid Rule + ML Correction
A two-stage pipeline where a purely linguistic `RegexpParser` generates initial BIO tags, which are then corrected by a CRF sequence model.
- **Features**: Orthographic patterns, POS context window $(\pm 2)$, shape features, and lexical flags.
- **Result**: Proves that even non-neural systems can achieve >92% F1 with structured correction.

### 2. "Paper-Style" BiLSTM-CRF
Designed to faithfully reproduce Huang et al. (2015) with several specific enhancements:
- **56-dim Feature Vector**: Includes case/shape, length-based bins, position flags, and coarse POS groupings.
- **POS N-Grams**: Hashed bigram and trigram embeddings feeding directly into the output scores (MaxEnt connections).
- **Optimization**: SGD with momentum and ReduceLROnPlateau scheduling.

### 3. Universal Transformer + CRF (UT-CRF)
A shared-weight Transformer block applied recurrently over depth steps.
- **ACT (Adaptive Computation Time)**: Allows the model to halt processing of "easy" tokens earlier than "hard" ones.
- **Relative Position Bias**: Captures local phrase structure by adding distance-aware offsets to attention logits.
- **Masked BIO Constraints**: Hard constraints during CRF decoding to prevent illegal transitions (e.g., $O \to I$).

### 4. Mamba SSM (Selective State Space)
Exploration of linear-time sequence modeling using Mamba-130M.
- **Limitations**: Underperformed legacy models due to subword-word alignment challenges and the relatively small size of CoNLL-2000 for fine-tuning such large parameters without extensive hyperparameter search.

---

## 📁 Project Structure

| Module | Description |
|--------|-------------|
| `data_loader.py` | Robust loader for CoNLL-2000 local JSON/Arrow exports. |
| `rule_chunker.py` | NLTK-based phrase-structure grammar rules. |
| `ml_model.py` | Logic for Hybrid Logistic Regression and CRF backends. |
| `bilstm_crf_paper_style.py` | Implementation of the best-performing traditional neural model. |
| `ut_crf_chunker.py` | Universal Transformer with recurrent depth and ACT. |
| `transformer_crf_chunker.py` | SOTA RoBERTa-based architecture. |
| `mamba_chunker.py` | Fine-tuning scripts for State-Space models. |

---

## 🛠️ Setup & Usage

This project is managed with `uv` for reproducible environments.

### Installation
```bash
uv sync
```

### Execution
Run the desired model architecture using one of the following commands:
- **Hybrid**: `uv run python main.py`
- **BiLSTM-CRF**: `uv run python bilstm_crf_paper_style.py`
- **UT-CRF**: `uv run python ut_crf_chunker.py`
- **RoBERTa**: `uv run python transformer_crf_chunker.py`

---

## 🚀 Key Findings

1.  **Transformers Dominate**: RoBERTa-base achieves the best absolute performance, eliminating the need for complex feature engineering.
2.  **CRF is Essential**: Structuring the output layer with a CRF consistently improves F1 by 1-8% in classic architectures by enforcing BIO transition logic.
3.  **Linguistic Features Matter**: The hybrid and paper-style BiLSTM models thrive on hand-crafted rules, allowing them to remain competitive even with 100x fewer parameters than large Transformers.

---
🚀 *Developed as part of an Advanced NLP Survey on Shallow Parsing.*
