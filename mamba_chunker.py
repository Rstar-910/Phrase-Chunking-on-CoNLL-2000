"""mamba_chunker.py
-------------------
Fine-tune a pretrained Mamba-based encoder (via Hugging Face
transformers + mamba-ssm) for CoNLL-2000 phrase chunking.

This script:
  * loads the local CoNLL-2000 data via data_loader.load_conll2000
  * builds a BIO label vocabulary from train+test
  * tokenizes sentences with a pretrained Mamba tokenizer
    (word-level tokens -> subwords, using is_split_into_words=True)
  * aligns BIO labels to subwords (first subword gets the label,
    others are ignored with -100 in the loss)
  * fine-tunes AutoModelForTokenClassification on train with a
    held-out dev set and early stopping on dev F1
  * evaluates on the test split using the existing evaluation.py
    utilities
  * saves metrics and the fine-tuned model checkpoint.

Usage (from project root):

    uv run python mamba_chunker.py

NOTE: You must choose a valid pretrained Mamba model name from
Hugging Face and set MODEL_NAME below accordingly.
"""

import time
import random
from datetime import datetime
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import AutoTokenizer, AutoModel

from data_loader import load_conll2000
from evaluation import evaluate_model, print_detailed_report, print_comparison_table


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: set this to an actual pretrained Mamba checkpoint on Hugging Face,
# e.g. "state-spaces/mamba-130m-hf" or similar.
MODEL_NAME = "state-spaces/mamba-130m-hf"

class MambaTokenClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state            # [B, T, H]
        logits = self.classifier(self.dropout(sequence_output))  # [B, T, C]

        loss = None
        if labels is not None:
            # labels shape [B, T], ignore -100 positions
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return {"loss": loss, "logits": logits}


def build_label_vocab(train_data, test_data=None) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Build mapping between chunk tags and integer IDs.

    Uses tags from train (and optionally test) to ensure all labels
    such as rare I-LST are covered.
    """
    tag_set = set()
    for sent in train_data:
        tag_set.update(sent["chunk_tags"])
    if test_data is not None:
        for sent in test_data:
            tag_set.update(sent["chunk_tags"])

    tag2id: Dict[str, int] = {}
    for tag in sorted(tag_set):
        tag2id.setdefault(tag, len(tag2id))
    id2tag = {i: t for t, i in tag2id.items()}
    return tag2id, id2tag


def encode_sentence_with_labels(tokens: List[str], labels: List[str], tokenizer, tag2id: Dict[str, int]):
    """Tokenize a sentence and align word-level BIO labels to subwords.

    Strategy (standard for BERT-style token classification):
      * use tokenizer(..., is_split_into_words=True)
      * for each subword token, find its word_id
      * if word_id is None (special tokens): label = -100
      * if this is the first subword of a word: label = tag2id[word_label]
      * otherwise: label = -100 (ignored in loss)
    """
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        return_attention_mask=True,
        truncation=True,
        add_special_tokens=True,
    )

    word_ids = encoding.word_ids()
    labels_ids = []
    previous_word_id = None

    for idx, word_id in enumerate(word_ids):
        if word_id is None:
            labels_ids.append(-100)
        elif word_id != previous_word_id:
            # first subword of this word
            labels_ids.append(tag2id[labels[word_id]])
        else:
            # subsequent subword of the same word
            labels_ids.append(-100)
        previous_word_id = word_id

    return encoding, labels_ids


class MambaChunkingDataset(Dataset):
    """Dataset of pre-encoded examples for Mamba token classification."""

    def __init__(self, data, tokenizer, tag2id):
        self.examples = []
        for sent in data:
            tokens = sent["tokens"]
            chunk_tags = sent["chunk_tags"]
            encoding, label_ids = encode_sentence_with_labels(tokens, chunk_tags, tokenizer, tag2id)
            self.examples.append({
                "input_ids": encoding["input_ids"],
                "attention_mask": encoding["attention_mask"],
                "labels": label_ids,
                "tokens": tokens,
                "chunk_tags": chunk_tags,
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_batch(batch, pad_token_id: int):
    """Pad a batch of variable-length sequences to the same length."""
    max_len = max(len(item["input_ids"]) for item in batch)

    input_ids_batch = []
    attention_mask_batch = []
    labels_batch = []

    tokens_batch = []  # keep original tokens/labels for optional use
    chunk_tags_batch = []

    for item in batch:
        ids = item["input_ids"]
        mask = item["attention_mask"]
        labels = item["labels"]

        pad_length = max_len - len(ids)
        input_ids_batch.append(ids + [pad_token_id] * pad_length)
        attention_mask_batch.append(mask + [0] * pad_length)
        labels_batch.append(labels + [-100] * pad_length)

        tokens_batch.append(item["tokens"])
        chunk_tags_batch.append(item["chunk_tags"])

    input_ids_tensor = torch.tensor(input_ids_batch, dtype=torch.long, device=DEVICE)
    attention_mask_tensor = torch.tensor(attention_mask_batch, dtype=torch.long, device=DEVICE)
    labels_tensor = torch.tensor(labels_batch, dtype=torch.long, device=DEVICE)

    return {
        "input_ids": input_ids_tensor,
        "attention_mask": attention_mask_tensor,
        "labels": labels_tensor,
        "tokens": tokens_batch,
        "chunk_tags": chunk_tags_batch,
    }


def predict_sentence(tokens: List[str], tokenizer, model, id2tag: Dict[int, str]) -> List[str]:
    """Run the fine-tuned model on a single sentence and return word-level BIO tags."""
    model.eval()
    with torch.no_grad():
        encoding = tokenizer(
            tokens,
            is_split_into_words=True,
            return_attention_mask=True,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        ).to(DEVICE)

        outputs = model(**encoding)
        logits = outputs["logits"]  # [1, seq_len, num_labels]
        predictions = logits.argmax(dim=-1)[0].cpu().tolist()
        word_ids = encoding.word_ids()

        word_preds: List[str] = []
        previous_word_id = None
        for idx, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            if word_id != previous_word_id:
                label_id = predictions[idx]
                word_preds.append(id2tag[label_id])
            previous_word_id = word_id

    return word_preds


def main():
    start = time.time()

    print("\n" + "=" * 60)
    print("  STEP 1: Loading CoNLL-2000 dataset")
    print("=" * 60)
    train_data, test_data = load_conll2000(
        hf_name="eriktks/conll2000",
        local_path="conll2000_local",
    )

    print("\n" + "=" * 60)
    print("  STEP 2: Building label vocabulary")
    print("=" * 60)
    tag2id, id2tag = build_label_vocab(train_data, test_data)
    num_labels = len(tag2id)
    print(f"[labels] num_labels = {num_labels}")

    print("\n" + "=" * 60)
    print("  STEP 3: Loading tokenizer and model")
    print("=" * 60)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = MambaTokenClassifier(MODEL_NAME, num_labels=num_labels).to(DEVICE)

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    # ── 4. Train/dev split ──────────────────────────────────────────
    dev_ratio = 0.1
    n_total = len(train_data)
    n_dev = max(1, int(n_total * dev_ratio))
    indices = list(range(n_total))
    random.seed(42)
    random.shuffle(indices)
    dev_indices = set(indices[:n_dev])
    train_indices = indices[n_dev:]

    train_sents = [train_data[i] for i in train_indices]
    dev_sents = [train_data[i] for i in dev_indices]
    print(f"[split] Train sentences: {len(train_sents)}, Dev sentences: {len(dev_sents)}")

    # ── 5. Build datasets/loaders ──────────────────────────────────
    train_dataset = MambaChunkingDataset(train_sents, tokenizer, tag2id)
    dev_dataset = MambaChunkingDataset(dev_sents, tokenizer, tag2id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, pad_token_id),
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=lambda b: collate_batch(b, pad_token_id),
    )

    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)

    print("\n" + "=" * 60)
    print("  STEP 6: Training Mamba token classifier (with early stopping)")
    print("=" * 60)

    num_epochs = 15
    patience = 3
    best_dev_f1 = -1.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"[epoch {epoch}] avg training loss: {avg_loss:.4f}")

        # ── Dev evaluation ────────────────────────────────────────
        model.eval()
        dev_true: List[List[str]] = []
        dev_pred: List[List[str]] = []

        with torch.no_grad():
            for batch in dev_loader:
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]
                tokens_batch = batch["tokens"]

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs["logits"]  # [B, T, num_labels]
                preds = logits.argmax(dim=-1).cpu().tolist()

                # Reconstruct word-level predictions using tokenizer.word_ids
                for i in range(len(tokens_batch)):
                    tokens = tokens_batch[i]
                    # Re-tokenize this sentence to get word_ids mapping
                    encoding = tokenizer(
                        tokens,
                        is_split_into_words=True,
                        return_attention_mask=True,
                        truncation=True,
                        add_special_tokens=True,
                    )
                    word_ids = encoding.word_ids()
                    pred_ids = preds[i][: len(word_ids)]

                    word_level_preds: List[str] = []
                    previous_word_id = None
                    for idx, word_id in enumerate(word_ids):
                        if word_id is None:
                            continue
                        if word_id != previous_word_id:
                            label_id = pred_ids[idx]
                            word_level_preds.append(id2tag[label_id])
                        previous_word_id = word_id

                    dev_pred.append(word_level_preds)
                    dev_true.append(batch["chunk_tags"][i])

        dev_results = evaluate_model(dev_true, dev_pred, "Mamba Dev")
        dev_f1 = dev_results["f1"]
        print(f"[epoch {epoch}] dev F1: {dev_f1:.4f}")

        if dev_f1 > best_dev_f1 + 1e-4:
            best_dev_f1 = dev_f1
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[early stopping] No dev F1 improvement for {patience} epochs.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ── 7. Test evaluation ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 7: Evaluation on test set")
    print("=" * 60)

    true_labels: List[List[str]] = []
    pred_labels: List[List[str]] = []

    for sent in test_data:
        tokens = sent["tokens"]
        chunk_tags = sent["chunk_tags"]
        preds = predict_sentence(tokens, tokenizer, model, id2tag)
        pred_labels.append(preds)
        true_labels.append(chunk_tags)

    mamba_results = evaluate_model(true_labels, pred_labels, "Mamba Chunker")
    print_detailed_report(true_labels, pred_labels, "Mamba Chunker")

    print_comparison_table({
        "Mamba": mamba_results,
    })

    # ── 8. Save results and model ─────────────────────────────────
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    backend = "Mamba"

    results_path = f"results_{backend}_{timestamp}.txt"
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("PHRASE CHUNKING RESULTS (Mamba Model)\n")
        f.write(f"Timestamp      : {timestamp}\n")
        f.write(f"Model backend  : {backend}\n")
        f.write("Dataset        : CoNLL-2000 (local JSON)\n\n")

        f.write("Mamba Chunker:\n")
        f.write(f"  Accuracy : {mamba_results['accuracy']:.4f}\n")
        f.write(f"  Precision: {mamba_results['precision']:.4f}\n")
        f.write(f"  Recall   : {mamba_results['recall']:.4f}\n")
        f.write(f"  F1       : {mamba_results['f1']:.4f}\n")

    print(f"[mamba_chunker] Saved metrics to {results_path}")

    model_path = f"mamba_chunker_{backend}_{timestamp}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "tag2id": tag2id,
        "id2tag": id2tag,
        "model_name": MODEL_NAME,
    }, model_path)
    print(f"[mamba_chunker] Saved model to {model_path}")

    elapsed = time.time() - start
    print(f"[mamba_chunker] Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
