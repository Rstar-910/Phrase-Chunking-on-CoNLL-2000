"""transformer_crf_chunker.py
---------------------------------
Pretrained Transformer + CRF chunker for CoNLL-2000.

This script uses a HuggingFace Transformer encoder (e.g. BERT/RoBERTa)
followed by a linear layer and a CRF, trained on the CoNLL-2000
chunking task using the same data loader and evaluation utilities as
bilstm_crf_paper_style.py.

Key points:
- Uses a subword tokenizer with `is_split_into_words=True` and pools
  hidden states back to one representation per *word*.
- Applies a CRF on top of word-level emissions.
- Reuses `build_vocabs`, `init_crf_bias`, and train/dev split behavior
  from the paper-style BiLSTM-CRF implementation so results are
  directly comparable.

Usage (from project root):

    uv run python transformer_crf_chunker.py

"""

import time
from datetime import datetime
from typing import List, Dict, Tuple
import sys

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

try:
    from torchcrf import CRF  # type: ignore
    HAS_TORCHCRF = True
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    CRF = None  # type: ignore
    HAS_TORCHCRF = False

from transformers import AutoTokenizer, AutoModel

from data_loader import load_conll2000
from evaluation import evaluate_model, print_detailed_report, print_comparison_table

# Reuse vocabulary utilities and data-split configuration from the
# paper-style BiLSTM-CRF implementation.
from bilstm_crf_paper_style import (  # type: ignore
    build_vocabs,
    init_crf_bias,
    DEV_RATIO,
    SEED,
    set_seed,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Configuration / hyperparameters
# ---------------------------------------------------------------------------

# Pretrained encoder name from HuggingFace hub. You can switch this to
# a stronger model (e.g. "roberta-large") if you have the compute.
TRANSFORMER_MODEL_NAME = "roberta-base"

# Training hyperparameters
BATCH_SIZE_TRAIN = 16
BATCH_SIZE_DEV = 64
MAX_EPOCHS = 10
PATIENCE = 3
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01

# LR scheduler (ReduceLROnPlateau on dev F1)
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 2


class TeeStdout:
    """Mirror stdout to both the terminal and a log file.

    This matches the logging behaviour used in ut_crf_chunker.py so
    that every print during Transformer-CRF training/evaluation is
    saved to a timestamped log file for later inspection.
    """

    def __init__(self, file_path: str) -> None:
        self._file = open(file_path, "w", encoding="utf-8")
        self._stdout = sys.stdout

    def write(self, data: str) -> None:  # type: ignore[override]
        self._stdout.write(data)
        self._file.write(data)

    def flush(self) -> None:  # type: ignore[override]
        self._stdout.flush()
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def isatty(self) -> bool:  # type: ignore[override]
        """Return whether the underlying stdout is a TTY.

        Some libraries (e.g. HuggingFace transformers) query
        sys.stdout.isatty() when printing colored loading reports.
        Delegating to the real stdout avoids AttributeError.
        """

        if hasattr(self._stdout, "isatty"):
            try:
                return bool(self._stdout.isatty())  # type: ignore[call-arg]
            except Exception:
                return False
        return False


# ---------------------------------------------------------------------------
# Dataset and collate function for Transformer + CRF
# ---------------------------------------------------------------------------


class TransformerChunkingDataset(Dataset):
    """Dataset that exposes raw tokens/POS and chunk labels.

    The actual subword tokenization is performed in the collate_fn
    so that we can use the HuggingFace tokenizer's batching utilities
    (including is_split_into_words=True for alignment).
    """

    def __init__(self, data, tag2id: Dict[str, int]):
        self.data = data
        self.tag2id = tag2id

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.data)

    def __getitem__(self, idx):  # type: ignore[override]
        sent = self.data[idx]
        tokens = sent["tokens"]
        pos_tags = sent["pos_tags"]  # kept for potential future use
        chunk_tags = sent["chunk_tags"]

        tag_ids = [self.tag2id[tag] for tag in chunk_tags]
        return tokens, pos_tags, torch.tensor(tag_ids, dtype=torch.long)


def make_transformer_collate_fn(tokenizer, tag_pad_id: int):
    """Create a collate_fn that tokenizes with HF and builds word-level labels.

    Returns a function that takes a batch of examples `(tokens, pos, tag_ids)`
    and produces:
      - input_ids:     [B, S] subword token IDs
      - attention_mask:[B, S]
      - word_starts:   List[List[int]] first-subword indices for each word
      - word_tags:     [B, W] padded word-level tag IDs
      - word_mask:     [B, W] bool mask over real words
    """

    def collate_fn(batch):
        tokens_list, pos_list, tag_tensors = zip(*batch)

        # Use the tokenizer's batch encoding with word alignment.
        encoding = tokenizer(
            list(tokens_list),
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        input_ids = encoding["input_ids"].to(DEVICE)
        attention_mask = encoding["attention_mask"].to(DEVICE)

        batch_size = input_ids.size(0)
        word_starts: List[List[int]] = []
        word_tag_lists: List[List[int]] = []
        max_words = 0

        for i in range(batch_size):
            tokens = tokens_list[i]
            tag_ids = tag_tensors[i].tolist()
            word_ids = encoding.word_ids(batch_index=i)

            # Find the first subword index for each original word.
            first_indices: List[int] = [-1] * len(tokens)
            seen = [False] * len(tokens)
            for sub_idx, w_id in enumerate(word_ids):
                if w_id is None:
                    continue
                if 0 <= w_id < len(tokens) and not seen[w_id]:
                    first_indices[w_id] = sub_idx
                    seen[w_id] = True

            # Fallback: in rare edge cases, drop words with no subword mapping.
            indices: List[int] = []
            filtered_tags: List[int] = []
            for w_idx, sub_idx in enumerate(first_indices):
                if sub_idx != -1:
                    indices.append(sub_idx)
                    filtered_tags.append(tag_ids[w_idx])

            word_starts.append(indices)
            word_tag_lists.append(filtered_tags)
            max_words = max(max_words, len(filtered_tags))

        # Build padded word-level tag tensor and mask
        word_tags = torch.full(
            (batch_size, max_words),
            tag_pad_id,
            dtype=torch.long,
            device=DEVICE,
        )
        word_mask = torch.zeros(
            batch_size,
            max_words,
            dtype=torch.bool,
            device=DEVICE,
        )

        for i, tags in enumerate(word_tag_lists):
            L = len(tags)
            if L == 0:
                continue
            word_tags[i, :L] = torch.tensor(tags, dtype=torch.long, device=DEVICE)
            word_mask[i, :L] = True

        return input_ids, attention_mask, word_starts, word_tags, word_mask

    return collate_fn


# ---------------------------------------------------------------------------
# Transformer + CRF model
# ---------------------------------------------------------------------------


class TransformerCRFChunker(nn.Module):
    """Pretrained Transformer encoder with a CRF layer on top.

    - Uses a HuggingFace encoder (e.g. RoBERTa/BERT) to obtain
      contextual subword representations.
    - Pools back to one representation per original word by taking the
      hidden state at the first subword of each word.
    - Applies a linear layer to obtain emission scores and a CRF for
      structured decoding over word-level BIO tags.
    """

    def __init__(self, model_name: str, tagset_size: int) -> None:
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        hidden_size = self.transformer.config.hidden_size

        self.dropout = nn.Dropout(0.1)
        self.hidden2tag = nn.Linear(hidden_size, tagset_size)

        self.use_crf = HAS_TORCHCRF
        if self.use_crf:
            self.crf = CRF(tagset_size, batch_first=True)
        else:
            self.crf = None  # type: ignore[assignment]
            self.loss_fn = nn.CrossEntropyLoss()

    def _encode_words(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        word_starts: List[List[int]],
        max_words: int,
    ) -> torch.Tensor:
        """Run the Transformer and pool to word-level representations.

        Args:
            input_ids:    [B, S]
            attention_mask:[B, S]
            word_starts:  list of length B, each a list of subword indices
            max_words:    maximum number of words in the batch

        Returns:
            word_reps:    [B, W, H] word-level encodings
        """
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # [B, S, H]

        batch_size, _, hidden_size = hidden.shape
        device = hidden.device

        word_reps = hidden.new_zeros(batch_size, max_words, hidden_size)

        for b in range(batch_size):
            indices = word_starts[b]
            if not indices:
                continue
            L = min(len(indices), max_words)
            idx_tensor = torch.tensor(indices[:L], dtype=torch.long, device=device)
            # hidden[b]: [S, H]; select the first L subword states
            word_reps[b, :L] = hidden[b].index_select(0, idx_tensor)

        word_reps = self.dropout(word_reps)
        return word_reps

    def neg_log_likelihood(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        word_starts: List[List[int]],
        word_tags: torch.Tensor,
        word_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Negative log-likelihood loss for a batch.

        Uses CRF log-likelihood when available, otherwise falls back
        to masked token-level cross-entropy over word-level emissions.
        """
        max_words = word_tags.size(1)
        word_reps = self._encode_words(input_ids, attention_mask, word_starts, max_words)
        emissions = self.hidden2tag(word_reps)  # [B, W, C]

        if self.use_crf and self.crf is not None:
            loss = -self.crf(emissions, word_tags, mask=word_mask, reduction="token_mean")
        else:
            B, W, C = emissions.shape
            emissions_flat = emissions.view(B * W, C)
            tags_flat = word_tags.view(B * W)
            mask_flat = word_mask.view(B * W)
            emissions_flat = emissions_flat[mask_flat]
            tags_flat = tags_flat[mask_flat]
            loss = self.loss_fn(emissions_flat, tags_flat)
        return loss

    def decode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        word_starts: List[List[int]],
        word_mask: torch.Tensor,
    ) -> List[List[int]]:
        """Decode best tag sequence for each sentence in the batch."""
        max_words = word_mask.size(1)
        word_reps = self._encode_words(input_ids, attention_mask, word_starts, max_words)
        emissions = self.hidden2tag(word_reps)

        if self.use_crf and self.crf is not None:
            paths = self.crf.decode(emissions, mask=word_mask)
            return paths
        # Greedy decode fallback
        B, W, _ = emissions.shape
        paths: List[List[int]] = []
        for b in range(B):
            L = int(word_mask[b].sum().item())
            scores = emissions[b, :L]
            preds = scores.argmax(dim=-1).tolist()
            paths.append(preds)
        return paths


# ---------------------------------------------------------------------------
# Training / evaluation pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    start = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = f"train_log_Transformer-CRF_{timestamp}.txt"
    tee = TeeStdout(log_path)
    old_stdout = sys.stdout
    sys.stdout = tee  # type: ignore[assignment]

    try:
        # Ensure reproducibility
        set_seed(SEED)

        print("\n" + "=" * 60)
        print("  STEP 1: Loading CoNLL-2000 dataset")
        print("=" * 60)
        train_data, test_data = load_conll2000(
            hf_name="eriktks/conll2000",
            local_path="conll2000_local",
        )

        print("\n" + "=" * 60)
        print("  STEP 2: Building vocabularies")
        print("=" * 60)
        token2id, pos2id, tag2id, id2tag = build_vocabs(train_data, test_data)

        # Train/dev split for early stopping (reuse DEV_RATIO, SEED behavior)
        n_total = len(train_data)
        n_dev = max(1, int(n_total * DEV_RATIO))
        indices = list(range(n_total))

        import random as _random

        _random.seed(SEED)
        _random.shuffle(indices)
        dev_indices = set(indices[:n_dev])
        train_indices = indices[n_dev:]

        train_sents = [train_data[i] for i in train_indices]
        dev_sents = [train_data[i] for i in dev_indices]
        print(f"[split] Train sentences: {len(train_sents)}, Dev sentences: {len(dev_sents)}")

        # Datasets and DataLoaders
        train_dataset = TransformerChunkingDataset(train_sents, tag2id)
        dev_dataset = TransformerChunkingDataset(dev_sents, tag2id)

        tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME, use_fast=True)

        collate_fn = make_transformer_collate_fn(tokenizer, tag_pad_id=0)

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE_TRAIN,
            shuffle=True,
            collate_fn=collate_fn,
        )
        dev_loader = DataLoader(
            dev_dataset,
            batch_size=BATCH_SIZE_DEV,
            shuffle=False,
            collate_fn=collate_fn,
        )

        # Model and optimizer
        model = TransformerCRFChunker(
            model_name=TRANSFORMER_MODEL_NAME,
            tagset_size=len(tag2id),
        ).to(DEVICE)

        # Initialize CRF biases for rare classes (same heuristic as BiLSTM-CRF)
        init_crf_bias(model, tag2id, train_sents)

        if not HAS_TORCHCRF:
            print("[warning] torchcrf not available; using Transformer + softmax (no CRF layer)")

        optimizer = AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=LR_SCHEDULER_FACTOR,
            patience=LR_SCHEDULER_PATIENCE,
            # verbose=True,
        )

        print("\n" + "=" * 60)
        print("  STEP 3: Training Transformer-CRF (dev early stopping)")
        print("=" * 60)
        print(
            f"  model={TRANSFORMER_MODEL_NAME}, lr={LEARNING_RATE}, "
            f"batch_size={BATCH_SIZE_TRAIN}"
        )

        best_dev_f1 = -1.0
        best_state = None
        best_epoch = 0
        epochs_no_improve = 0

        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            total_loss = 0.0

            for (
                input_ids,
                attention_mask,
                word_starts,
                word_tags,
                word_mask,
            ) in train_loader:
                optimizer.zero_grad()
                loss = model.neg_log_likelihood(
                    input_ids,
                    attention_mask,
                    word_starts,
                    word_tags,
                    word_mask,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"[epoch {epoch}] avg training loss: {avg_loss:.4f}  (LR={current_lr:.6f})")

            # Dev evaluation
            model.eval()
            dev_true: List[List[str]] = []
            dev_pred: List[List[str]] = []
            with torch.no_grad():
                for (
                    input_ids,
                    attention_mask,
                    word_starts,
                    word_tags,
                    word_mask,
                ) in dev_loader:
                    paths = model.decode(
                        input_ids,
                        attention_mask,
                        word_starts,
                        word_mask,
                    )
                    for b, path in enumerate(paths):
                        L = int(word_mask[b].sum().item())
                        pred_ids = path[:L]
                        gold_ids = word_tags[b, :L].tolist()
                        pred_tags = [id2tag[j] for j in pred_ids]
                        gold_tags = [id2tag[j] for j in gold_ids]
                        dev_pred.append(pred_tags)
                        dev_true.append(gold_tags)

            dev_results = evaluate_model(dev_true, dev_pred, "Transformer-CRF Dev")
            dev_f1 = dev_results["f1"]
            print(f"[epoch {epoch}] dev F1: {dev_f1:.4f}")

            scheduler.step(dev_f1)

            if dev_f1 > best_dev_f1 + 1e-4:
                best_dev_f1 = dev_f1
                best_state = model.state_dict()
                best_epoch = epoch
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= PATIENCE:
                    print(f"[early stopping] No dev F1 improvement for {PATIENCE} epochs.")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        if best_epoch == 0:
            best_epoch = epoch

        print(f"[early stopping] Best dev F1: {best_dev_f1:.4f} at epoch {best_epoch}")

        # ------------------------------------------------------------------
        # Optional retraining on full training data for best_epoch
        # ------------------------------------------------------------------
        print("\n" + "=" * 60)
        print("  STEP 3b: Retraining Transformer-CRF on full training data")
        print("=" * 60)
        print(f"[full-train] Using best_epoch = {best_epoch}")

        full_train_dataset = TransformerChunkingDataset(train_data, tag2id)
        full_train_loader = DataLoader(
            full_train_dataset,
            batch_size=BATCH_SIZE_TRAIN,
            shuffle=True,
            collate_fn=collate_fn,
        )

        model_full = TransformerCRFChunker(
            model_name=TRANSFORMER_MODEL_NAME,
            tagset_size=len(tag2id),
        ).to(DEVICE)

        # Initialize CRF biases for rare classes using full train_data
        init_crf_bias(model_full, tag2id, train_data)

        optimizer_full = AdamW(
            model_full.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        scheduler_full = ReduceLROnPlateau(
            optimizer_full,
            mode="min",
            factor=LR_SCHEDULER_FACTOR,
            patience=LR_SCHEDULER_PATIENCE,
            # verbose=True,
        )

        for e in range(1, best_epoch + 1):
            model_full.train()
            total_loss = 0.0
            for (
                input_ids,
                attention_mask,
                word_starts,
                word_tags,
                word_mask,
            ) in full_train_loader:
                optimizer_full.zero_grad()
                loss = model_full.neg_log_likelihood(
                    input_ids,
                    attention_mask,
                    word_starts,
                    word_tags,
                    word_mask,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model_full.parameters(), max_norm=5.0)
                optimizer_full.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(full_train_loader)
            current_lr = optimizer_full.param_groups[0]["lr"]
            print(
                f"[full-train epoch {e}/{best_epoch}] avg training loss: "
                f"{avg_loss:.4f}  (LR={current_lr:.6f})"
            )
            scheduler_full.step(avg_loss)

        # Use the full-train model for final evaluation
        model = model_full

        print("\n" + "=" * 60)
        print("  STEP 4: Evaluation on test set (Transformer-CRF)")
        print("=" * 60)

        model.eval()
        pred_labels: List[List[str]] = []
        true_labels: List[List[str]] = []

        # Build a DataLoader over the test split for efficient batching
        test_dataset = TransformerChunkingDataset(test_data, tag2id)
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE_DEV,
            shuffle=False,
            collate_fn=collate_fn,
        )

        with torch.no_grad():
            for (
                input_ids,
                attention_mask,
                word_starts,
                word_tags,
                word_mask,
            ) in test_loader:
                paths = model.decode(
                    input_ids,
                    attention_mask,
                    word_starts,
                    word_mask,
                )
                batch_size = len(paths)
                for b in range(batch_size):
                    L = int(word_mask[b].sum().item())
                    pred_ids = paths[b][:L]
                    gold_ids = word_tags[b, :L].tolist()
                    pred_tags = [id2tag[j] for j in pred_ids]
                    gold_tags = [id2tag[j] for j in gold_ids]
                    pred_labels.append(pred_tags)
                    true_labels.append(gold_tags)

        tf_results = evaluate_model(true_labels, pred_labels, "Transformer-CRF Chunker")
        print_detailed_report(true_labels, pred_labels, "Transformer-CRF Chunker")

        print_comparison_table({
            "Transformer-CRF": tf_results,
        })

        backend = "Transformer-CRF" if HAS_TORCHCRF else "Transformer"

        results_path = f"results_{backend}_{timestamp}.txt"
        with open(results_path, "w", encoding="utf-8") as f:
            f.write("PHRASE CHUNKING RESULTS (Transformer + CRF)\n")
            f.write(f"Timestamp      : {timestamp}\n")
            f.write(f"Model backend  : {backend}\n")
            f.write("Dataset        : CoNLL-2000 (local JSON)\n\n")

            f.write("Transformer-CRF Chunker:\n")
            f.write(f"  Accuracy : {tf_results['accuracy']:.4f}\n")
            f.write(f"  Precision: {tf_results['precision']:.4f}\n")
            f.write(f"  Recall   : {tf_results['recall']:.4f}\n")
            f.write(f"  F1       : {tf_results['f1']:.4f}\n")

        print(f"[transformer_crf_chunker] Saved metrics to {results_path}")

        model_path = f"transformer_crf_chunker_{backend}_{timestamp}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "tag2id": tag2id,
                "backend": backend,
                "transformer_model": TRANSFORMER_MODEL_NAME,
            },
            model_path,
        )
        print(f"[transformer_crf_chunker] Saved model to {model_path}")

        elapsed = time.time() - start
        print(f"[transformer_crf_chunker] Total time: {elapsed:.1f}s")
        print(f"[transformer_crf_chunker] Training log captured in {log_path}")
    finally:
        sys.stdout = old_stdout  # type: ignore[assignment]
        tee.close()


if __name__ == "__main__":
    main()
