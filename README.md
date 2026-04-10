# Advanced Phrase Chunking Survey

A comprehensive evaluation and comparison of multiple architectures for **Phrase Chunking (Shallow Parsing)**, evaluated on the **CoNLL-2000 shared task** dataset. This project explores the evolution of chunking techniques from classic rule-based systems to modern selecitve state-space models and transformers.

## 🚀 Key Features

- **Multiple Architectures**: Supports Rule-Based, Hybrid, BiLSTM-CRF, Mamba, and Transformer-CRF models.
- **Dependency Management**: Fully integrated with `uv` for lightning-fast, reproducible environments.
- **Unified Evaluation**: Uses `seqeval` for standard BIO-tag evaluation metrics (Precision, Recall, F1).
- **Automated Logging**: Saves detailed results and training logs for performance tracking.

## 📊 Experimental Results

Evaluated on the CoNLL-2000 test set. The models represent a progression from baseline heuristics to state-of-the-art neural architectures.

| Chunker Model       | Accuracy | Precision | Recall | F1 Score |
|---------------------|----------|-----------|--------|----------|
| **Rule-Based**      | 82.55%   | 72.69%    | 78.37% | 75.42%   |
| **BiLSTM-CRF**      | 92.10%   | 85.32%    | 87.49% | 86.39%   |
| **Mamba**           | 94.51%   | 88.41%    | 90.56% | 89.47%   |
| **Hybrid (Rule+CRF)** | 95.03% | 92.51%    | 92.29% | 92.40%   |
| **Transformer-CRF** | **97.64%** | **96.41%** | **96.48%** | **96.44%** |

*Note: The Hybrid model benefits from hand-crafted grammar rules as features, often outperforming basic neural models on smaller datasets.*

## 🛠️ Project Structure

```text
├── data_loader.py          # Data utilities for CoNLL-2000
├── rule_chunker.py         # NLTK-based baseline heuristics
├── ml_model.py             # Hybrid model logic (ML/CRF backends)
├── bilstm_crf_chunker.py   # Neural BiLSTM-CRF implementation
├── mamba_chunker.py        # Modern Selective State Space model
├── transformer_crf_chunker.py # SOTA Transformer + CRF architecture
├── evaluation.py           # Standardized evaluation pipeline
├── main.py                 # Entry point for Hybrid model
└── pyproject.toml          # uv project configuration
```

## ⚙️ Setup & Installation

This project uses `uv` for package management.

1. **Install uv** (if not already):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Sync Dependencies**:
   ```bash
   uv sync
   ```

3. **External Resources**:
   - The dataset is automatically downloaded from HuggingFace via `data_loader.py`.
   - For GloVe/Senna embeddings, place the relevant files in the `glove/` or `senna/` directories.

## 🏃 Usage

You can run each model independently using `uv run`.

### 1. Hybrid Chunker (Rule-Based + CRF)
```bash
uv run python main.py
```

### 2. BiLSTM-CRF
```bash
uv run python bilstm_crf_chunker.py
```

### 3. Mamba Chunker
```bash
uv run python mamba_chunker.py
```

### 4. Transformer-CRF (RoBERTa)
```bash
uv run python transformer_crf_chunker.py
```

## 📝 Approach Details

### Hybrid (Rule + ML)
Uses NLTK's `RegexpParser` to generate initial predictions, which are then passed as features to a CRF (Conditional Random Field). The model learns to correct the systematic errors of the rule-based approach.

### Neural Architectures
- **BiLSTM-CRF**: Uses character-level and word-level embeddings passed through a bidirectional LSTM with a CRF output layer.
- **Mamba**: Leverages a Selective State Space model for token classification, offering efficient long-range dependency modeling.
- **Transformer-CRF**: Fine-tunes a `RoBERTa-base` encoder with a custom CRF head, achieving best-in-class performance.

---
🚀 *Developed as part of an Advanced NLP Survey on Shallow Parsing.*
