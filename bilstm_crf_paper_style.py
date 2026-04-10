"""bilstm_crf_paper_style.py
---------------------------------
Implementation of a Bidirectional LSTM-CRF sequence tagger that
follows the training setup described in:

    Zhiheng Huang, Wei Xu, Kai Yu
    "Bidirectional LSTM-CRF Models for Sequence Tagging"

Key characteristics of this script relative to the paper:

- Uses a bidirectional LSTM over word (and POS) embeddings.
- Adds a CRF layer on top of the BiLSTM emissions to model
  sentence-level tag dependencies.
- Trains with SGD using the negative log-likelihood of the
  CRF as the objective, following the paper's Algorithm 1
  (BiLSTM-CRF forward pass -> CRF forward/backward ->
   BiLSTM backward pass -> parameter update).
- Uses the existing CoNLL-2000 data loader and evaluation utilities
  from this project so it integrates cleanly with your current
  pipeline and metrics.

Improvements over the initial version (aligned with the paper):
- LR = 0.1 (paper: "We use a learning rate of 0.1")
- Smaller batch size (~10 sentences) to approximate paper's ~100 token batches
- Enriched spelling features (prefixes, suffixes, word patterns) per §4.2.1
- POS n-gram context features (unigram, bigram, trigram) per §4.2.2
- Embedding dropout for regularization
- Learning rate scheduling (ReduceLROnPlateau)
- CRF bias initialization for rare classes

Usage (from project root):

    uv run python bilstm_crf_paper_style.py

This will train the model on the CoNLL-2000 training split,
monitor development F1 on a held-out portion of the training
set (as CoNLL-2000 has no official dev set), and finally
evaluate on the official test split, printing metrics and
saving results + model to disk.
"""

import os
import time
import random
from collections import Counter
from datetime import datetime
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

try:
    from torchcrf import CRF  # type: ignore
    HAS_TORCHCRF = True
except ModuleNotFoundError:
    CRF = None  # type: ignore
    HAS_TORCHCRF = False

from data_loader import load_conll2000
from evaluation import evaluate_model, print_detailed_report, print_comparison_table


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Configuration / hyperparameters  (aligned with the paper)
# ---------------------------------------------------------------------------

# Embedding dimensions
WORD_EMB_DIM = 100
POS_EMB_DIM = 25

# BiLSTM settings
HIDDEN_DIM = 300
NUM_LAYERS = 1
DROPOUT = 0.5

# Training hyperparameters — paper: "We use a learning rate of 0.1"
BATCH_SIZE_TRAIN = 10      # ~10 sentences ≈ 240 tokens (paper uses ~100 token batches)
BATCH_SIZE_DEV = 64
MAX_EPOCHS = 30
PATIENCE = 7               # more patience with higher LR
LEARNING_RATE = 0.1        # paper: 0.1
MOMENTUM = 0.9

# LR scheduler
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 3

# Data split
DEV_RATIO = 0.1
SEED = 42

# Pretrained embeddings (GloVe) configuration
GLOVE_PATH = "glove.6B.100d.txt"
GLOVE_EMB_DIM = 100
USE_PRETRAINED = True

# Dimensionality of enriched spelling/context feature vectors per token.
# Updated to include richer features per paper §4.2.1 and §4.2.2
FEATURE_DIM = 50  # 9 (shape) + 4 (length) + 5 (position) + 14 (POS context) + 18 (spelling)

# POS n-gram embedding dimensions (context features per §4.2.2)
POS_NGRAM_EMB_DIM = 16
POS_NGRAM_HASH_SIZE = 1024  # hash table size for bigram/trigram IDs


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Vocabulary utilities
# ---------------------------------------------------------------------------


def build_vocabs(train_data, test_data=None) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int], Dict[int, str]]:
    """Build token, POS, and chunk-label vocabularies from data.

    Tokens / POS tags come from the train split; labels come from
    train and test so that rare tags only present in the test split
    (e.g. I-LST) are still recognized, matching the behavior used in
    your other chunker scripts.
    """
    token_set = set()
    pos_set = set()
    tag_set = set()

    for sent in train_data:
        token_set.update(sent["tokens"])
        pos_set.update(sent["pos_tags"])
        tag_set.update(sent["chunk_tags"])

    if test_data is not None:
        for sent in test_data:
            tag_set.update(sent["chunk_tags"])

    # Add special tokens
    token2id: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
    for tok in sorted(token_set):
        token2id.setdefault(tok, len(token2id))

    pos2id: Dict[str, int] = {"<PAD>": 0}
    for pos in sorted(pos_set):
        pos2id.setdefault(pos, len(pos2id))

    tag2id: Dict[str, int] = {}
    for tag in sorted(tag_set):
        tag2id.setdefault(tag, len(tag2id))

    id2tag: Dict[int, str] = {i: t for t, i in tag2id.items()}
    return token2id, pos2id, tag2id, id2tag


# ---------------------------------------------------------------------------
# Spelling feature helpers (paper §4.2.1)
# ---------------------------------------------------------------------------

# Common suffixes and prefixes — mapped to integer IDs for feature vectors
COMMON_SUFFIXES = [
    "ing", "tion", "ed", "ly", "er", "est", "ness", "ment",
    "ous", "ive", "al", "ful", "less", "able", "ible", "en",
    "ity", "ance", "ence", "ion", "es", "s", "ty", "ry",
]
SUFFIX2ID = {s: i + 1 for i, s in enumerate(COMMON_SUFFIXES)}  # 0 = none

COMMON_PREFIXES = [
    "un", "re", "in", "dis", "en", "non", "pre", "over",
    "mis", "out", "sub", "anti", "de", "inter", "trans",
]
PREFIX2ID = {p: i + 1 for i, p in enumerate(COMMON_PREFIXES)}  # 0 = none


def _get_suffix_id(word: str) -> int:
    """Return the ID of the longest matching common suffix, or 0."""
    lower = word.lower()
    best_id = 0
    best_len = 0
    for sfx, sid in SUFFIX2ID.items():
        if lower.endswith(sfx) and len(sfx) > best_len and len(sfx) < len(lower):
            best_id = sid
            best_len = len(sfx)
    return best_id


def _get_prefix_id(word: str) -> int:
    """Return the ID of the longest matching common prefix, or 0."""
    lower = word.lower()
    best_id = 0
    best_len = 0
    for pfx, pid in PREFIX2ID.items():
        if lower.startswith(pfx) and len(pfx) > best_len and len(pfx) < len(lower):
            best_id = pid
            best_len = len(pfx)
    return best_id


def _word_pattern(word: str) -> str:
    """Map characters: uppercase -> 'A', lowercase -> 'a', digit -> '0', other kept."""
    out = []
    for ch in word:
        if ch.isupper():
            out.append("A")
        elif ch.islower():
            out.append("a")
        elif ch.isdigit():
            out.append("0")
        else:
            out.append(ch)
    return "".join(out)


def _word_pattern_summarized(word: str) -> str:
    """Like _word_pattern but with consecutive identical chars collapsed."""
    pattern = _word_pattern(word)
    if not pattern:
        return pattern
    result = [pattern[0]]
    for ch in pattern[1:]:
        if ch != result[-1]:
            result.append(ch)
    return "".join(result)


# Pattern categories: map summarized patterns to coarse buckets
def _pattern_category(word: str) -> int:
    """Classify word pattern into coarse categories (0-7)."""
    pat = _word_pattern_summarized(word)
    if pat == "Aa":           # Capitalized word (e.g., "London")
        return 0
    elif pat == "A":          # All caps single char or acronym
        return 1
    elif pat == "a":          # All lower single char or all lower word
        return 2
    elif "0" in pat and "a" not in pat and "A" not in pat:  # All digits
        return 3
    elif "0" in pat:          # Mixed digits
        return 4
    elif pat == "A.":         # Abbreviation-like
        return 5
    elif "-" in pat:          # Hyphenated
        return 6
    else:
        return 7


def _pos_group_index(pos: str) -> int:
    """Map a POS tag to a coarse group index.

    Groups (0-based):
      0: noun-like, 1: verb-like, 2: modifier (adj/adv),
      3: preposition, 4: determiner/pronoun, 5: punctuation, 6: other.
    """
    if pos in {"NN", "NNS", "NNP", "NNPS"}:
        return 0
    if pos in {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "MD"}:
        return 1
    if pos in {"JJ", "JJR", "JJS", "RB", "RBR", "RBS", "WRB"}:
        return 2
    if pos in {"IN", "TO"}:
        return 3
    if pos in {"DT", "PDT", "PRP", "PRP$", "WDT", "WP", "WP$", "EX"}:
        return 4
    if pos in {".", ",", ":", "``", "''", "-LRB-", "-RRB-", "#", "$", "(", ")"}:
        return 5
    return 6


def _pos_group_one_hot(pos: str | None) -> List[float]:
    """Return a 7-dim one-hot vector for a POS group (or 'other' if None)."""
    idx = _pos_group_index(pos) if pos is not None else 6
    vec = [0.0] * 7
    vec[idx] = 1.0
    return vec


def _pos_bigram_hash(pos1: str | None, pos2: str | None) -> int:
    """Hash a POS bigram to an integer in [0, POS_NGRAM_HASH_SIZE)."""
    key = f"{pos1 or '<BOS>'}_{pos2 or '<EOS>'}"
    return hash(key) % POS_NGRAM_HASH_SIZE


def _pos_trigram_hash(pos1: str | None, pos2: str, pos3: str | None) -> int:
    """Hash a POS trigram to an integer in [0, POS_NGRAM_HASH_SIZE)."""
    key = f"{pos1 or '<BOS>'}_{pos2}_{pos3 or '<EOS>'}"
    return hash(key) % POS_NGRAM_HASH_SIZE


def _build_features(tokens: List[str], pos_tags: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build enriched features for a sentence.

    Returns:
        feat_mat:    [T, FEATURE_DIM] float tensor of hand-crafted features
        bigram_ids:  [T] long tensor of POS bigram hash IDs
        trigram_ids: [T] long tensor of POS trigram hash IDs
    """
    n = len(tokens)
    feat_rows: List[torch.Tensor] = []
    bigram_list: List[int] = []
    trigram_list: List[int] = []

    for i, tok in enumerate(tokens):
        # ── Basic shape / case (9 features) ──
        is_capitalized = 1.0 if tok and tok[0].isupper() else 0.0
        is_all_caps = 1.0 if tok.isupper() else 0.0
        is_all_lower = 1.0 if tok.islower() else 0.0

        has_digit = 1.0 if any(ch.isdigit() for ch in tok) else 0.0
        has_alpha = 1.0 if any(ch.isalpha() for ch in tok) else 0.0
        has_hyphen = 1.0 if "-" in tok else 0.0
        has_apostrophe = 1.0 if "'" in tok else 0.0

        is_alnum = 1.0 if tok.isalnum() else 0.0
        is_mixed_case = 1.0 if any(ch.islower() for ch in tok) and any(ch.isupper() for ch in tok) else 0.0

        # ── Length-based features (4 features) ──
        length = len(tok)
        word_length_norm = float(min(length, 20)) / 20.0 if tok else 0.0
        length_le_3 = 1.0 if length <= 3 else 0.0
        length_le_8 = 1.0 if 4 <= length <= 8 else 0.0
        length_gt_8 = 1.0 if length > 8 else 0.0

        # ── Position features (5 features) ──
        is_first = 1.0 if i == 0 else 0.0
        is_last = 1.0 if i == n - 1 else 0.0
        is_near_start = 1.0 if i <= 2 else 0.0
        is_near_end = 1.0 if i >= n - 3 else 0.0
        rel_pos = float(i) / float(max(n - 1, 1))

        # ── Coarse POS context features (14 features = 2 * 7-dim one-hot) ──
        prev_pos = pos_tags[i - 1] if i > 0 else None
        next_pos = pos_tags[i + 1] if i < n - 1 else None
        prev_group = _pos_group_one_hot(prev_pos)
        next_group = _pos_group_one_hot(next_pos)

        # ── Paper §4.2.1: enriched spelling features (24 new features) ──

        # Suffix/prefix ID (normalized to [0, 1] range, 2 features)
        suffix_id = _get_suffix_id(tok)
        prefix_id = _get_prefix_id(tok)
        suffix_norm = float(suffix_id) / float(max(len(COMMON_SUFFIXES), 1))
        prefix_norm = float(prefix_id) / float(max(len(COMMON_PREFIXES), 1))

        # Word pattern category (8-dim one-hot, 8 features)
        pat_cat = _pattern_category(tok)
        pat_one_hot = [0.0] * 8
        pat_one_hot[pat_cat] = 1.0

        # Letter-only & non-letter-only flags (paper §4.2.1)
        letters_only = 1.0 if tok.isalpha() else 0.0
        non_letters_only = 1.0 if tok and not any(ch.isalpha() for ch in tok) else 0.0

        # Apostrophe-s ending (paper §4.2.1)
        has_apostrophe_s = 1.0 if tok.endswith("'s") or tok.endswith("'S") else 0.0

        # Has punctuation (paper §4.2.1)
        has_punct = 1.0 if any(not ch.isalnum() and ch not in ("-", "'") for ch in tok) else 0.0

        # Non-initial capital (paper §4.2.1)
        non_initial_cap = 1.0 if len(tok) > 1 and any(ch.isupper() for ch in tok[1:]) else 0.0

        # Suffix length indicators (3-dim: ends with common 2-char, 3-char, 4-char suffix)
        lower_tok = tok.lower()
        has_2char_suffix = 1.0 if any(lower_tok.endswith(s) for s in COMMON_SUFFIXES if len(s) == 2) else 0.0
        has_3char_suffix = 1.0 if any(lower_tok.endswith(s) for s in COMMON_SUFFIXES if len(s) == 3) else 0.0
        has_4plus_suffix = 1.0 if any(lower_tok.endswith(s) for s in COMMON_SUFFIXES if len(s) >= 4) else 0.0

        feats = [
            # Original features (18)
            is_capitalized,
            is_all_caps,
            is_all_lower,
            has_digit,
            has_alpha,
            has_hyphen,
            has_apostrophe,
            is_alnum,
            is_mixed_case,
            word_length_norm,
            length_le_3,
            length_le_8,
            length_gt_8,
            is_first,
            is_last,
            is_near_start,
            is_near_end,
            rel_pos,
            # Coarse POS context (14)
            *prev_group,
            *next_group,
            # NEW: enriched spelling features (18)
            suffix_norm,
            prefix_norm,
            *pat_one_hot,
            letters_only,
            non_letters_only,
            has_apostrophe_s,
            has_punct,
            non_initial_cap,
            has_2char_suffix,
            has_3char_suffix,
            has_4plus_suffix,
        ]
        # Total: 9 + 4 + 5 + 14 + 18 = 50 = FEATURE_DIM
        feat_rows.append(torch.tensor(feats, dtype=torch.float))

        # ── POS n-gram IDs (paper §4.2.2) ──
        # Bigram: (prev_pos, current_pos)
        bigram_list.append(_pos_bigram_hash(prev_pos, pos_tags[i]))
        # Trigram: (prev_pos, current_pos, next_pos)
        trigram_list.append(_pos_trigram_hash(prev_pos, pos_tags[i], next_pos))

    feat_mat = torch.stack(feat_rows, dim=0)
    bigram_ids = torch.tensor(bigram_list, dtype=torch.long)
    trigram_ids = torch.tensor(trigram_list, dtype=torch.long)

    return feat_mat, bigram_ids, trigram_ids


class ChunkingDataset(Dataset):
    """PyTorch Dataset for sentence-level chunking data.

    Each item is a single sentence represented as integer IDs for
    tokens, POS tags, and chunk labels, plus enriched feature vectors
    and POS n-gram IDs.
    """

    def __init__(self, data, token2id, pos2id, tag2id):
        self.data = data
        self.token2id = token2id
        self.pos2id = pos2id
        self.tag2id = tag2id

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.data)

    def __getitem__(self, idx):  # type: ignore[override]
        sent = self.data[idx]
        tokens = sent["tokens"]
        pos_tags = sent["pos_tags"]
        chunk_tags = sent["chunk_tags"]

        word_ids = [self.token2id.get(tok, self.token2id["<UNK>"]) for tok in tokens]
        pos_ids = [self.pos2id.get(pos, self.pos2id["<PAD>"]) for pos in pos_tags]
        tag_ids = [self.tag2id[tag] for tag in chunk_tags]

        # Build enriched features
        feat_mat, bigram_ids, trigram_ids = _build_features(tokens, pos_tags)

        return (
            torch.tensor(word_ids, dtype=torch.long),
            torch.tensor(pos_ids, dtype=torch.long),
            torch.tensor(tag_ids, dtype=torch.long),
            feat_mat,
            bigram_ids,
            trigram_ids,
        )


def make_collate_fn(pad_word_id: int, pad_pos_id: int):
    """Create a collate_fn that pads variable-length sentences in a batch.

    Returns a function suitable for use as DataLoader(collate_fn=...).
    The collate function:
      - pads word / POS / tag sequences to the same length
      - returns a boolean mask indicating real (non-pad) positions
      - moves tensors to the global DEVICE.
    """

    def collate_fn(batch):
        word_seqs, pos_seqs, tag_seqs, feat_seqs, bigram_seqs, trigram_seqs = zip(*batch)
        batch_size = len(batch)
        lengths = [len(s) for s in word_seqs]
        max_len = max(lengths)

        word_ids = torch.full(
            (batch_size, max_len), pad_word_id, dtype=torch.long, device=DEVICE
        )
        pos_ids = torch.full(
            (batch_size, max_len), pad_pos_id, dtype=torch.long, device=DEVICE
        )
        # Tag padding value does not matter; masked out in the loss.
        tags = torch.zeros(batch_size, max_len, dtype=torch.long, device=DEVICE)
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=DEVICE)

        # Dense feature tensor: [B, T, FEATURE_DIM]
        feats = torch.zeros(
            batch_size,
            max_len,
            FEATURE_DIM,
            dtype=torch.float,
            device=DEVICE,
        )

        # POS n-gram ID tensors: [B, T]
        bigrams = torch.zeros(batch_size, max_len, dtype=torch.long, device=DEVICE)
        trigrams = torch.zeros(batch_size, max_len, dtype=torch.long, device=DEVICE)

        for i, (w, p, t, f, bg, tg) in enumerate(
            zip(word_seqs, pos_seqs, tag_seqs, feat_seqs, bigram_seqs, trigram_seqs)
        ):
            L = len(w)
            word_ids[i, :L] = w.to(DEVICE)
            pos_ids[i, :L] = p.to(DEVICE)
            tags[i, :L] = t.to(DEVICE)
            mask[i, :L] = True
            feats[i, :L, :] = f.to(DEVICE)
            bigrams[i, :L] = bg.to(DEVICE)
            trigrams[i, :L] = tg.to(DEVICE)

        return word_ids, pos_ids, tags, mask, feats, bigrams, trigrams

    return collate_fn


# ---------------------------------------------------------------------------
# GloVe utilities (reused from bilstm_crf_chunker.py)
# ---------------------------------------------------------------------------


def load_glove_embeddings(path: str, embedding_dim: int) -> Dict[str, torch.Tensor]:
    """Load GloVe embeddings from a text file if available."""
    embeddings: Dict[str, torch.Tensor] = {}
    if not os.path.exists(path):
        print(f"[glove] File not found at {path}, using random init for word embeddings.")
        return embeddings

    print(f"[glove] Loading GloVe vectors from {path} ...")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != embedding_dim + 1:
                continue
            word = parts[0]
            try:
                vec = torch.tensor([float(x) for x in parts[1:]], dtype=torch.float)
            except ValueError:
                continue
            if vec.shape[0] == embedding_dim:
                embeddings[word] = vec

    print(f"[glove] Loaded {len(embeddings)} word vectors from GloVe.")
    return embeddings


def init_word_embeddings_from_glove(
    embedding_layer: nn.Embedding,
    token2id: Dict[str, int],
    glove_path: str,
    embedding_dim: int,
) -> None:
    """Initialize the word embedding layer from GloVe if available."""
    glove = load_glove_embeddings(glove_path, embedding_dim)
    if not glove:
        return

    with torch.no_grad():
        weight = embedding_layer.weight.data
        found = 0
        for token, idx in token2id.items():
            if token in ("<PAD>", "<UNK>"):
                continue
            vec = glove.get(token)
            if vec is not None and vec.shape[0] == embedding_dim:
                weight[idx] = vec
                found += 1

    print(f"[glove] Initialized {found} / {len(token2id)} word embeddings from GloVe.")


# ---------------------------------------------------------------------------
# BiLSTM-CRF model (paper-style with enriched features)
# ---------------------------------------------------------------------------


class PaperBiLSTMCRF(nn.Module):
    """Bidirectional LSTM-CRF model for sequence tagging.

    This is an implementation of the architecture described in the
    "Bidirectional LSTM-CRF" paper with enriched features:
      - A bidirectional LSTM over word (and optional POS) embeddings.
      - Embedding dropout for regularization.
      - A linear layer mapping BiLSTM outputs to tag emission scores.
      - A CRF layer on top that models tag transitions across the sentence.
      - Direct feature-to-output connections (MaxEnt-style, paper §4.2.4).
      - POS n-gram embeddings (bigram, trigram) per paper §4.2.2.

    If torchcrf is not installed, the model falls back to a plain
    BiLSTM + softmax tagger with token-level cross-entropy loss.
    """

    def __init__(
        self,
        vocab_size: int,
        pos_vocab_size: int,
        tagset_size: int,
        word_emb_dim: int = 100,
        pos_emb_dim: int = 25,
        hidden_dim: int = 300,
        num_layers: int = 1,
        dropout: float = 0.5,
        use_pos: bool = True,
    ):
        super().__init__()

        self.use_pos = use_pos

        self.word_embeds = nn.Embedding(vocab_size, word_emb_dim, padding_idx=0)
        if self.use_pos:
            self.pos_embeds = nn.Embedding(pos_vocab_size, pos_emb_dim, padding_idx=0)
            lstm_input_dim = word_emb_dim + pos_emb_dim
        else:
            self.pos_embeds = None  # type: ignore[assignment]
            lstm_input_dim = word_emb_dim

        # Embedding dropout (applied before LSTM)
        self.emb_dropout = nn.Dropout(0.3)

        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

        # Direct connection from hand-crafted features to tag scores,
        # in parallel with the BiLSTM path, as suggested in the paper (§4.2.4).
        self.feat2tag = nn.Linear(FEATURE_DIM, tagset_size)

        # POS n-gram embeddings (paper §4.2.2: bigram and trigram features)
        self.bigram_emb = nn.Embedding(POS_NGRAM_HASH_SIZE, POS_NGRAM_EMB_DIM, padding_idx=0)
        self.trigram_emb = nn.Embedding(POS_NGRAM_HASH_SIZE, POS_NGRAM_EMB_DIM, padding_idx=0)
        self.ngram2tag = nn.Linear(POS_NGRAM_EMB_DIM * 2, tagset_size)

        self.use_crf = HAS_TORCHCRF
        if self.use_crf:
            self.crf = CRF(tagset_size, batch_first=True)
        else:
            self.crf = None  # type: ignore[assignment]
            self.loss_fn = nn.CrossEntropyLoss()

    def _emissions(self, word_ids, pos_ids, feats, bigram_ids, trigram_ids):
        """Compute emission scores for a batch of sentences.

        word_ids:     [B, T]
        pos_ids:      [B, T]
        feats:        [B, T, FEATURE_DIM]
        bigram_ids:   [B, T]
        trigram_ids:  [B, T]

        Returns
        -------
        emissions : torch.Tensor
            Shape [B, T, C] where C = tagset_size.
        """
        word_emb = self.word_embeds(word_ids)
        if self.use_pos and self.pos_embeds is not None:
            pos_emb = self.pos_embeds(pos_ids)
            embeds = torch.cat([word_emb, pos_emb], dim=-1)
        else:
            embeds = word_emb

        # Apply embedding dropout
        embeds = self.emb_dropout(embeds)

        lstm_out, _ = self.lstm(embeds)
        lstm_out = self.dropout(lstm_out)
        base_emissions = self.hidden2tag(lstm_out)

        # Map token-level features directly to tag scores and add (§4.2.4).
        feat_scores = self.feat2tag(feats)

        # POS n-gram scores (§4.2.2)
        bg_emb = self.bigram_emb(bigram_ids)    # [B, T, POS_NGRAM_EMB_DIM]
        tg_emb = self.trigram_emb(trigram_ids)   # [B, T, POS_NGRAM_EMB_DIM]
        ngram_cat = torch.cat([bg_emb, tg_emb], dim=-1)  # [B, T, 2*POS_NGRAM_EMB_DIM]
        ngram_scores = self.ngram2tag(ngram_cat)

        emissions = base_emissions + feat_scores + ngram_scores
        return emissions

    def neg_log_likelihood(self, word_ids, pos_ids, feats, bigram_ids, trigram_ids, tags, mask=None):
        """Compute the negative log-likelihood loss for a batch.

        This corresponds to the CRF log-likelihood objective in the
        paper. When the CRF layer is present, the loss is the negative
        log-probability of the correct tag sequence. When CRF is not
        available, we fall back to masked token-level cross-entropy.
        """
        emissions = self._emissions(word_ids, pos_ids, feats, bigram_ids, trigram_ids)
        if self.use_crf and self.crf is not None:
            if mask is not None:
                loss = -self.crf(emissions, tags, mask=mask, reduction="token_mean")
            else:
                loss = -self.crf(emissions, tags, reduction="token_mean")
        else:
            # Fall back to token-level cross-entropy
            B, T, C = emissions.shape
            if mask is not None:
                emissions = emissions[mask]
                tags = tags[mask]
            loss = self.loss_fn(emissions.view(-1, C), tags.view(-1))
        return loss

    def decode(self, word_ids, pos_ids, feats, bigram_ids, trigram_ids, mask=None):
        """Decode the best tag sequence for each sentence in the batch.

        Uses the Viterbi algorithm from the CRF layer when available;
        otherwise returns greedy argmax predictions over emission scores.
        """
        emissions = self._emissions(word_ids, pos_ids, feats, bigram_ids, trigram_ids)
        if self.use_crf and self.crf is not None:
            if mask is not None:
                return self.crf.decode(emissions, mask=mask)
            return self.crf.decode(emissions)
        # Greedy decode if CRF is unavailable
        return emissions.argmax(dim=-1).tolist()


# ---------------------------------------------------------------------------
# CRF initialization for rare classes
# ---------------------------------------------------------------------------

def init_crf_bias(model: PaperBiLSTMCRF, tag2id: Dict[str, int], train_data) -> None:
    """Initialize CRF transition biases to help with rare classes.

    We compute tag frequency from the training data and slightly bias
    the CRF start transitions toward rare tags to give them a chance
    to be predicted.
    """
    if not model.use_crf or model.crf is None:
        return

    # Count tag frequencies
    tag_counts = Counter()
    for sent in train_data:
        tag_counts.update(sent["chunk_tags"])

    total = sum(tag_counts.values())
    if total == 0:
        return

    # Compute log prior for each tag and use it to initialize start_transitions
    with torch.no_grad():
        for tag, idx in tag2id.items():
            count = tag_counts.get(tag, 0)
            if count == 0:
                # Give rare tags a small positive start bias
                model.crf.start_transitions.data[idx] = 0.1
            else:
                # Frequency-based bias: rare tags get slightly higher start probability
                freq = count / total
                if freq < 0.001:  # very rare
                    model.crf.start_transitions.data[idx] += 0.05

    print("[crf_init] Initialized CRF start transitions with frequency-based biases.")


# ---------------------------------------------------------------------------
# Training / evaluation pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    start = time.time()

    # Ensure reproducibility across runs
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

    # ------------------------------------------------------------------
    # Train/dev split for early stopping
    # ------------------------------------------------------------------
    n_total = len(train_data)
    n_dev = max(1, int(n_total * DEV_RATIO))
    indices = list(range(n_total))
    random.seed(SEED)
    random.shuffle(indices)
    dev_indices = set(indices[:n_dev])
    train_indices = indices[n_dev:]

    train_sents = [train_data[i] for i in train_indices]
    dev_sents = [train_data[i] for i in dev_indices]
    print(f"[split] Train sentences: {len(train_sents)}, Dev sentences: {len(dev_sents)}")

    # ------------------------------------------------------------------
    # Datasets and DataLoaders
    # ------------------------------------------------------------------
    train_dataset = ChunkingDataset(train_sents, token2id, pos2id, tag2id)
    dev_dataset = ChunkingDataset(dev_sents, token2id, pos2id, tag2id)

    collate_fn = make_collate_fn(token2id["<PAD>"], pos2id["<PAD>"])
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

    # ------------------------------------------------------------------
    # Model and optimizer (SGD with momentum, dev-based early stopping)
    # ------------------------------------------------------------------
    model = PaperBiLSTMCRF(
        vocab_size=len(token2id),
        pos_vocab_size=len(pos2id),
        tagset_size=len(tag2id),
        word_emb_dim=WORD_EMB_DIM,
        pos_emb_dim=POS_EMB_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        use_pos=True,
    ).to(DEVICE)

    # Initialize word embeddings from GloVe if requested
    if USE_PRETRAINED and GLOVE_EMB_DIM == model.word_embeds.embedding_dim:
        init_word_embeddings_from_glove(model.word_embeds, token2id, GLOVE_PATH, GLOVE_EMB_DIM)
    elif USE_PRETRAINED:
        print("[glove] Skipping GloVe init: embedding dim mismatch.")

    # Initialize CRF biases for rare classes
    init_crf_bias(model, tag2id, train_sents)

    if not HAS_TORCHCRF:
        print("[warning] torchcrf not available; using BiLSTM + softmax (no CRF layer)")

    optimizer = SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    # LR scheduler: reduce LR when dev F1 plateaus
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE,
        # verbose=True,
    )

    print("\n" + "=" * 60)
    print("  STEP 3: Training BiLSTM-CRF (paper-style with dev early stopping)")
    print("=" * 60)
    print(f"  LR={LEARNING_RATE}, batch_size={BATCH_SIZE_TRAIN}, "
          f"hidden={HIDDEN_DIM}, dropout={DROPOUT}")
    print(f"  Feature dim={FEATURE_DIM}, POS n-gram hash size={POS_NGRAM_HASH_SIZE}")

    best_dev_f1 = -1.0
    best_state = None
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        # Algorithm 1: for each batch -> forward (BiLSTM-CRF) ->
        # CRF loss -> backward -> parameter update.
        for word_ids, pos_ids, tags, mask, feats, bigrams, trigrams in train_loader:
            optimizer.zero_grad()
            loss = model.neg_log_likelihood(
                word_ids, pos_ids, feats, bigrams, trigrams, tags, mask=mask
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"[epoch {epoch}] avg training loss: {avg_loss:.4f}  (LR={current_lr:.6f})")

        # Dev evaluation for early stopping
        model.eval()
        dev_true: List[List[str]] = []
        dev_pred: List[List[str]] = []
        with torch.no_grad():
            for word_ids, pos_ids, tags, mask, feats, bigrams, trigrams in dev_loader:
                paths = model.decode(word_ids, pos_ids, feats, bigrams, trigrams, mask=mask)
                for i, path in enumerate(paths):
                    length = int(mask[i].sum().item())
                    pred_ids = path[:length]
                    gold_ids = tags[i, :length].tolist()
                    pred_tags = [id2tag[j] for j in pred_ids]
                    gold_tags = [id2tag[j] for j in gold_ids]
                    dev_pred.append(pred_tags)
                    dev_true.append(gold_tags)

        dev_results = evaluate_model(dev_true, dev_pred, "BiLSTM-CRF Paper Dev")
        dev_f1 = dev_results["f1"]
        print(f"[epoch {epoch}] dev F1: {dev_f1:.4f}")

        # Step the LR scheduler
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
    print("  STEP 3b: Retraining on full training data")
    print("=" * 60)
    print(f"[full-train] Using best_epoch = {best_epoch}")

    full_train_dataset = ChunkingDataset(train_data, token2id, pos2id, tag2id)
    full_train_loader = DataLoader(
        full_train_dataset,
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=True,
        collate_fn=collate_fn,
    )

    model_full = PaperBiLSTMCRF(
        vocab_size=len(token2id),
        pos_vocab_size=len(pos2id),
        tagset_size=len(tag2id),
        word_emb_dim=WORD_EMB_DIM,
        pos_emb_dim=POS_EMB_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        use_pos=True,
    ).to(DEVICE)

    if USE_PRETRAINED and GLOVE_EMB_DIM == model_full.word_embeds.embedding_dim:
        init_word_embeddings_from_glove(model_full.word_embeds, token2id, GLOVE_PATH, GLOVE_EMB_DIM)
    elif USE_PRETRAINED:
        print("[glove] Skipping GloVe init for full-train model: embedding dim mismatch.")

    # Initialize CRF biases for rare classes in full-train model
    init_crf_bias(model_full, tag2id, train_data)

    optimizer_full = SGD(model_full.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
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
        for word_ids, pos_ids, tags, mask, feats, bigrams, trigrams in full_train_loader:
            optimizer_full.zero_grad()
            loss = model_full.neg_log_likelihood(
                word_ids, pos_ids, feats, bigrams, trigrams, tags, mask=mask
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_full.parameters(), max_norm=5.0)
            optimizer_full.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(full_train_loader)
        current_lr = optimizer_full.param_groups[0]["lr"]
        print(f"[full-train epoch {e}/{best_epoch}] avg training loss: {avg_loss:.4f}  (LR={current_lr:.6f})")
        scheduler_full.step(avg_loss)

    # Use the full-train model for final evaluation
    model = model_full

    print("\n" + "=" * 60)
    print("  STEP 4: Evaluation on test set (BiLSTM-CRF paper-style)")
    print("=" * 60)

    model.eval()
    pred_labels: List[List[str]] = []
    true_labels: List[List[str]] = []

    # Use the full (train+test) vocab and the trained model to tag the
    # official CoNLL-2000 test split.
    with torch.no_grad():
        for sent in test_data:
            tokens = sent["tokens"]
            pos_tags = sent["pos_tags"]
            chunk_tags = sent["chunk_tags"]

            word_ids = torch.tensor(
                [token2id.get(tok, token2id["<UNK>"]) for tok in tokens],
                dtype=torch.long,
                device=DEVICE,
            ).unsqueeze(0)
            pos_ids = torch.tensor(
                [pos2id.get(pos, pos2id["<PAD>"]) for pos in pos_tags],
                dtype=torch.long,
                device=DEVICE,
            ).unsqueeze(0)

            # Construct enriched features for this sentence
            feat_mat, bigram_ids, trigram_ids = _build_features(tokens, pos_tags)
            feat_mat = feat_mat.unsqueeze(0).to(DEVICE)
            bigram_ids = bigram_ids.unsqueeze(0).to(DEVICE)
            trigram_ids = trigram_ids.unsqueeze(0).to(DEVICE)

            # Full-length mask (no padding for a single sentence)
            mask = torch.ones_like(word_ids, dtype=torch.bool)

            pred_seq_ids = model.decode(word_ids, pos_ids, feat_mat, bigram_ids, trigram_ids, mask=mask)[0]
            pred_seq_tags = [id2tag[i] for i in pred_seq_ids]

            pred_labels.append(pred_seq_tags)
            true_labels.append(chunk_tags)

    bilstm_results = evaluate_model(true_labels, pred_labels, "BiLSTM-CRF Chunker (paper-style)")
    print_detailed_report(true_labels, pred_labels, "BiLSTM-CRF Chunker (paper-style)")

    print_comparison_table({
        "BiLSTM-CRF (paper-style)": bilstm_results,
    })

    # ------------------------------------------------------------------
    # Save results and model snapshot
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    backend = "BiLSTM-CRF" if HAS_TORCHCRF else "BiLSTM"

    results_path = f"results_{backend}_paper_style_{timestamp}.txt"
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("PHRASE CHUNKING RESULTS (BiLSTM-CRF, paper-style)\n")
        f.write(f"Timestamp      : {timestamp}\n")
        f.write(f"Model backend  : {backend}\n")
        f.write("Dataset        : CoNLL-2000 (local JSON)\n\n")

        f.write("BiLSTM-CRF Chunker (paper-style):\n")
        f.write(f"  Accuracy : {bilstm_results['accuracy']:.4f}\n")
        f.write(f"  Precision: {bilstm_results['precision']:.4f}\n")
        f.write(f"  Recall   : {bilstm_results['recall']:.4f}\n")
        f.write(f"  F1       : {bilstm_results['f1']:.4f}\n")

    print(f"[bilstm_crf_paper_style] Saved metrics to {results_path}")

    model_path = f"bilstm_crf_chunker_{backend}_paper_style_{timestamp}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "token2id": token2id,
            "pos2id": pos2id,
            "tag2id": tag2id,
            "backend": backend,
        },
        model_path,
    )
    print(f"[bilstm_crf_paper_style] Saved model to {model_path}")

    elapsed = time.time() - start
    print(f"[bilstm_crf_paper_style] Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
