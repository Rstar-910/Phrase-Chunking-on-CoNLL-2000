"""ut_crf_chunker.py
-----------------------
Universal Transformer + CRF chunker for CoNLL-2000.

This script reuses the data loading, feature engineering, and CRF
setup from the paper-style BiLSTM-CRF implementation, but replaces the
BiLSTM encoder with a Universal Transformer encoder (shared
self-attention + feed-forward block applied recurrently over depth).

Training, dev-based early stopping, full-train retraining, and test
evaluation follow the same pattern as in bilstm_crf_paper_style_2.py,
so results are directly comparable.

Usage (from project root):

    uv run python ut_crf_chunker.py

"""

import time
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from collections import Counter
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from data_loader import load_conll2000
from evaluation import evaluate_model, print_detailed_report, print_comparison_table

# Reuse vocab/feature utilities and training hyperparameters from the
# paper-style BiLSTM-CRF implementation so that the UT model is
# directly comparable and integrates cleanly.
from bilstm_crf_paper_style import (  # type: ignore
    build_vocabs,
    ChunkingDataset,
    make_collate_fn,
    _build_features,
    init_word_embeddings_from_glove,
    init_crf_bias,
    FEATURE_DIM,
    POS_NGRAM_EMB_DIM,
    POS_NGRAM_HASH_SIZE,
    GLOVE_PATH,
    GLOVE_EMB_DIM,
    USE_PRETRAINED,
    BATCH_SIZE_TRAIN,
    BATCH_SIZE_DEV,
    MAX_EPOCHS,
    PATIENCE,
    LEARNING_RATE,
    MOMENTUM,
    LR_SCHEDULER_FACTOR,
    LR_SCHEDULER_PATIENCE,
    DEV_RATIO,
    SEED,
    set_seed,
)

try:
    from torchcrf import CRF  # type: ignore
    HAS_TORCHCRF = True
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    CRF = None  # type: ignore
    HAS_TORCHCRF = False


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Universal Transformer configuration
# ---------------------------------------------------------------------------

# Input embedding dimensions (kept identical to BiLSTM-CRF setup)
WORD_EMB_DIM = 100
POS_EMB_DIM = 25

# Universal Transformer model dimensions
D_MODEL = 300          # model/hidden size
UT_NUM_STEPS = 8       # number of recurrent depth steps (increased)
UT_NUM_HEADS = 4       # number of attention heads
UT_FF_DIM = 1024       # feed-forward inner dimension (increased)
DROPOUT = 0.5

# Optimizer settings for the UT encoder (we override the BiLSTM-style
# SGD hyperparameters with AdamW, which tends to work better for
# transformer architectures).
UT_LEARNING_RATE = 5e-4
UT_WEIGHT_DECAY = 0.01

# Maximum sequence length for positional embeddings (CoNLL-2000
# sentences are short, so 256 is ample)
MAX_SEQ_LEN = 256

# Character-level embedding configuration
CHAR_EMB_DIM = 30
CHAR_CNN_OUT = 50

# Adaptive Computation Time (ACT) configuration
ACT_THRESHOLD = 0.9
ACT_EPSILON = 0.01
ACT_LAMBDA = 0.01

# SENNA embedding configuration (used instead of GloVe for UT)
USE_SENNA_FOR_UT = True
SENNA_PATH = "senna_50d.txt"  # adjust if your SENNA file is elsewhere
SENNA_EMB_DIM = 50

# Auxiliary cost-sensitive loss configuration for ultra-rare labels
AUX_CE_LAMBDA = 0.1
RARE_CLASS_WEIGHT = 50.0
RARE_LABELS_FOR_AUX = ["B-INTJ", "I-INTJ", "B-LST", "I-LST"]


class TeeStdout:
    """Simple tee for stdout: writes to terminal and a log file.

    All prints in this script will be mirrored into the given file.
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
        try:
            self._file.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Rare-class oversampling helpers
# ---------------------------------------------------------------------------


def compute_chunk_tag_counts(sentences) -> Counter:
    """Count BIO chunk tags (e.g. B-NP, I-VP, B-LST) in a dataset."""

    counts: Counter = Counter()
    for sent in sentences:
        counts.update(sent["chunk_tags"])
    return counts


def get_rare_chunk_tags(counts: Counter, threshold: int = 50) -> List[str]:
    """Return tags that appear fewer than ``threshold`` times.

    The default threshold is conservative: it picks up ultra-rare and
    low-frequency BIO tags (e.g. B-LST, I-LST, B-INTJ, etc.) without
    radically distorting the distribution of common tags like NP/VP/PP.
    """

    rare = [tag for tag, c in counts.items() if c < threshold]
    return sorted(rare)


def oversample_rare_sentences(train_sents, rare_tags, factor: int = 5):
    """Oversample sentences that contain any of the given rare BIO tags.

    Each sentence containing at least one tag in ``rare_tags`` is
    duplicated ``factor`` times (i.e., kept once plus ``factor-1``
    additional copies). Dev and test splits are left untouched.
    """

    if not rare_tags or factor <= 1:
        return train_sents

    oversampled = list(train_sents)
    rare_sent_count = 0
    rare_tags_set = set(rare_tags)

    for sent in train_sents:
        if any(tag in rare_tags_set for tag in sent["chunk_tags"]):
            rare_sent_count += 1
            for _ in range(factor - 1):
                oversampled.append(sent)

    print(f"[oversampling] Tags={rare_tags}, factor={factor}")
    print(
        f"[oversampling] Original train sentences: {len(train_sents)}, "
        f"sentences with matching tags: {rare_sent_count}, "
        f"after oversampling: {len(oversampled)}"
    )
    return oversampled


def split_rare_medium_tags(
    counts: Counter,
    ultra_threshold: int = 50,
    medium_threshold: int = 300,
) -> tuple[List[str], List[str]]:
    """Split BIO tags into ultra-rare and medium-rare bands.

    - Ultra-rare: count < ultra_threshold
    - Medium-rare: ultra_threshold <= count < medium_threshold
    """

    ultra = []
    medium = []
    for tag, c in counts.items():
        if c < ultra_threshold:
            ultra.append(tag)
        elif c < medium_threshold:
            medium.append(tag)
    return sorted(ultra), sorted(medium)


def apply_emission_bias(
    model: nn.Module,
    tag2id: Dict[str, int],
    ultra_tags: List[str],
    medium_tags: List[str],
    ultra_bias: float = 0.2,
    medium_bias: float = 0.1,
) -> None:
    """Add a small positive bias to emission logits for rare tags.

    This operates on the bias of the hidden2tag layer so that ultra-
    rare and medium-rare labels are slightly more likely a priori.
    """

    if not hasattr(model, "hidden2tag") or model.hidden2tag.bias is None:  # type: ignore[union-attr]
        return

    bias = model.hidden2tag.bias  # type: ignore[assignment]
    applied = 0

    with torch.no_grad():
        for tag in ultra_tags:
            idx = tag2id.get(tag)
            if idx is not None and 0 <= idx < bias.size(0):
                bias[idx] += ultra_bias
                applied += 1
        for tag in medium_tags:
            idx = tag2id.get(tag)
            if idx is not None and 0 <= idx < bias.size(0):
                bias[idx] += medium_bias
                applied += 1

    print(
        f"[emission-bias] Applied bias to {applied} tags "
        f"(ultra_bias={ultra_bias}, medium_bias={medium_bias})."
    )


def init_word_embeddings_from_senna(
    embedding: nn.Embedding,
    token2id: Dict[str, int],
    senna_path: str,
    senna_dim: int,
) -> None:
    """Initialize word embeddings from SENNA vectors.

    The SENNA file is expected to be a whitespace-separated text file
    with ``word dim1 dim2 ...`` per line. Vectors are projected into
    the model's embedding dimension by copying into the first
    ``senna_dim`` components and leaving the remaining dimensions
    unchanged.
    """

    if not os.path.exists(senna_path):
        print(f"[senna] File not found at {senna_path}, skipping SENNA init.")
        return

    emb_weight = embedding.weight.data
    emb_dim = emb_weight.size(1)

    print(f"[senna] Loading SENNA vectors from {senna_path} ...")
    senna_vectors: Dict[str, torch.Tensor] = {}
    with open(senna_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != senna_dim + 1:
                continue
            word = parts[0]
            try:
                vec_vals = [float(v) for v in parts[1:]]
            except ValueError:
                continue
            senna_vectors[word] = torch.tensor(vec_vals, dtype=torch.float)

    matched = 0
    for tok, idx in token2id.items():
        if tok in senna_vectors and 0 <= idx < emb_weight.size(0):
            vec = senna_vectors[tok]
            # Project SENNA vector into the embedding space
            copy_len = min(emb_dim, senna_dim)
            emb_weight[idx, :copy_len] = vec[:copy_len]
            matched += 1

    print(f"[senna] Initialized {matched} / {len(token2id)} word embeddings from SENNA.")


class UTEncoderBlock(nn.Module):
    """Single Universal Transformer block with self-attention and FFN.

    The same block is applied repeatedly over depth with shared
    parameters. A separate step embedding is added at each depth
    iteration to let the model distinguish different computation
    steps (standard UT practice without ACT).
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
        step_embed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x: [B, T, D]
        if step_embed is not None:
            x = x + step_embed

        # Multi-head self-attention with residual + layer norm
        attn_out, _ = self.self_attn(
            x,
            x,
            x,
            key_padding_mask=key_padding_mask,  # True at PAD positions
            need_weights=False,
        )
        x = self.norm1(x + self.dropout1(attn_out))

        # Position-wise feed-forward with residual + layer norm
        ff_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ff_out))
        return x


class UTCRFChunker(nn.Module):
    """Universal Transformer + CRF model for sequence tagging.

    Architecture:
      - Word + POS embeddings, projected to D_MODEL.
      - Learned absolute positional embeddings (up to MAX_SEQ_LEN).
      - A single UTEncoderBlock applied UT_NUM_STEPS times with shared
        parameters and step embeddings.
      - Linear head from encoder outputs to tag emission scores.
      - Direct feature-to-tag and POS n-gram-to-tag connections reused
        from the paper-style BiLSTM-CRF implementation.
      - CRF layer on top (if torchcrf is installed).
    """

    def __init__(
        self,
        vocab_size: int,
        pos_vocab_size: int,
        tagset_size: int,
        token2id: Dict[str, int],
        tag2id: Dict[str, int],
        word_emb_dim: int = WORD_EMB_DIM,
        pos_emb_dim: int = POS_EMB_DIM,
        d_model: int = D_MODEL,
        num_steps: int = UT_NUM_STEPS,
        n_heads: int = UT_NUM_HEADS,
        d_ff: int = UT_FF_DIM,
        dropout: float = DROPOUT,
        use_pos: bool = True,
        use_char: bool = True,
        use_act: bool = True,
    ) -> None:
        super().__init__()

        self.token2id = token2id
        self.tag2id = tag2id
        self.use_pos = use_pos
        self.use_char = use_char
        self.use_act = use_act
        self.num_steps = num_steps

        # Token / POS embeddings
        self.word_embeds = nn.Embedding(vocab_size, word_emb_dim, padding_idx=0)
        if self.use_pos:
            self.pos_embeds = nn.Embedding(pos_vocab_size, pos_emb_dim, padding_idx=0)
            base_input_dim = word_emb_dim + pos_emb_dim
        else:
            self.pos_embeds = None  # type: ignore[assignment]
            base_input_dim = word_emb_dim

        # Character-level embeddings (built from vocabulary tokens)
        if self.use_char:
            char2id: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
            for tok in token2id.keys():
                for ch in tok:
                    if ch not in char2id:
                        char2id[ch] = len(char2id)
            self.char2id = char2id
            self.char_pad_id = char2id["<PAD>"]
            self.char_unk_id = char2id["<UNK>"]
            self.char_embeds = nn.Embedding(len(char2id), CHAR_EMB_DIM, padding_idx=self.char_pad_id)
            self.char_cnn = nn.Conv1d(CHAR_EMB_DIM, CHAR_CNN_OUT, kernel_size=3, padding=1)
            # build inverse vocab for reconstructing tokens from IDs
            id2token: List[Optional[str]] = [None] * len(token2id)
            for tok, idx in token2id.items():
                if 0 <= idx < len(id2token):
                    id2token[idx] = tok
            self.id2token = id2token
            input_dim = base_input_dim + CHAR_CNN_OUT
        else:
            self.char2id = None  # type: ignore[assignment]
            self.char_pad_id = 0
            self.char_unk_id = 1
            self.char_embeds = None  # type: ignore[assignment]
            self.char_cnn = None  # type: ignore[assignment]
            self.id2token = None  # type: ignore[assignment]
            input_dim = base_input_dim

        self.emb_dropout = nn.Dropout(0.3)

        # Project concatenated embeddings to model dimension
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional and step embeddings for the Universal Transformer
        self.pos_encoder = nn.Embedding(MAX_SEQ_LEN, d_model)
        self.step_embeddings = nn.Embedding(num_steps, d_model)

        self.ut_block = UTEncoderBlock(d_model, n_heads, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

        # Map encoder outputs to tag scores
        self.hidden2tag = nn.Linear(d_model, tagset_size)

        # Direct connections from dense features and POS n-grams to tag
        # scores, mirroring the paper-style BiLSTM-CRF implementation.
        self.feat2tag = nn.Linear(FEATURE_DIM, tagset_size)

        self.bigram_emb = nn.Embedding(POS_NGRAM_HASH_SIZE, POS_NGRAM_EMB_DIM, padding_idx=0)
        self.trigram_emb = nn.Embedding(POS_NGRAM_HASH_SIZE, POS_NGRAM_EMB_DIM, padding_idx=0)
        self.ngram2tag = nn.Linear(POS_NGRAM_EMB_DIM * 2, tagset_size)

        # ACT controller
        if self.use_act:
            self.act_fc = nn.Linear(d_model, 1)
            self.act_lambda = ACT_LAMBDA
        else:
            self.act_fc = None  # type: ignore[assignment]
            self.act_lambda = 0.0

        # Auxiliary cost-sensitive CE loss weights focusing on
        # ultra-rare labels like INTJ and LST.
        aux_weights = torch.ones(tagset_size, dtype=torch.float)
        for lbl in RARE_LABELS_FOR_AUX:
            idx = tag2id.get(lbl)
            if idx is not None and 0 <= idx < tagset_size:
                aux_weights[idx] = RARE_CLASS_WEIGHT
        self.register_buffer("aux_ce_weight", aux_weights)
        self.aux_ce_lambda = AUX_CE_LAMBDA

        self.use_crf = HAS_TORCHCRF
        if self.use_crf:
            self.crf = CRF(tagset_size, batch_first=True)
        else:
            self.crf = None  # type: ignore[assignment]
            self.loss_fn = nn.CrossEntropyLoss()

    def _build_char_rep(self, word_ids: torch.Tensor) -> torch.Tensor:
        """Construct per-token character CNN representations.

        Args:
            word_ids: [B, T]
        Returns:
            char_rep: [B, T, CHAR_CNN_OUT]
        """
        assert self.use_char and self.char_embeds is not None and self.char_cnn is not None
        assert self.id2token is not None

        B, T = word_ids.shape
        total_tokens = B * T

        # Build char-id sequences for each token
        char_seqs: List[List[int]] = []
        max_len = 0
        for i in range(B):
            for j in range(T):
                wid = int(word_ids[i, j].item())
                tok = None
                if 0 <= wid < len(self.id2token) and self.id2token[wid] is not None:
                    tok = self.id2token[wid]
                if tok is None:
                    tok = "<UNK>"
                seq = [self.char2id.get(ch, self.char_unk_id) for ch in tok]  # type: ignore[union-attr]
                if not seq:
                    seq = [self.char_unk_id]
                max_len = max(max_len, len(seq))
                char_seqs.append(seq)

        char_ids = torch.full(
            (total_tokens, max_len),
            self.char_pad_id,
            dtype=torch.long,
            device=word_ids.device,
        )
        for idx, seq in enumerate(char_seqs):
            char_ids[idx, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=word_ids.device)

        # Embedding then temporal conv + max-pool
        char_emb = self.char_embeds(char_ids).transpose(1, 2)  # [tokens, C, L]
        char_conv = torch.relu(self.char_cnn(char_emb))        # [tokens, CHAR_CNN_OUT, L]
        char_pooled, _ = torch.max(char_conv, dim=-1)          # [tokens, CHAR_CNN_OUT]

        char_rep = char_pooled.view(B, T, -1)
        return char_rep

    def _encode(
        self,
        word_ids: torch.Tensor,
        pos_ids: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the Universal Transformer encoder (with optional ACT).

        Returns (encodings, act_loss).
        """

        word_emb = self.word_embeds(word_ids)
        emb_list = [word_emb]
        if self.use_pos and self.pos_embeds is not None:
            emb_list.append(self.pos_embeds(pos_ids))
        if self.use_char and self.char_embeds is not None and self.char_cnn is not None:
            emb_list.append(self._build_char_rep(word_ids))

        embeds = torch.cat(emb_list, dim=-1)

        embeds = self.emb_dropout(embeds)
        x = self.input_proj(embeds)  # [B, T, D_MODEL]

        B, T, _ = x.shape
        # Absolute positional encodings
        if T > MAX_SEQ_LEN:
            positions = torch.arange(T, device=x.device).clamp(max=MAX_SEQ_LEN - 1)
        else:
            positions = torch.arange(T, device=x.device)
        positions = positions.unsqueeze(0).expand(B, T)
        x = x + self.pos_encoder(positions)

        # key_padding_mask expects True at PAD positions
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask  # invert: True at PAD

        # If ACT is disabled, run fixed steps
        if not self.use_act or self.act_fc is None:
            for step in range(self.num_steps):
                step_ids = torch.full((B, T), step, dtype=torch.long, device=x.device)
                step_embed = self.step_embeddings(step_ids)
                x = self.ut_block(x, key_padding_mask=key_padding_mask, step_embed=step_embed)
            x = self.dropout(x)
            act_loss = x.new_zeros(())
            return x, act_loss

        # ACT-enabled recurrent processing
        halting_prob = torch.zeros((B, T), device=x.device)
        remainders = torch.zeros((B, T), device=x.device)
        n_updates = torch.zeros((B, T), device=x.device)
        # Only real tokens participate in ACT; pads are always halted
        still_running = mask.clone() if mask is not None else torch.ones((B, T), dtype=torch.bool, device=x.device)

        previous_state = torch.zeros_like(x)

        for step in range(self.num_steps):
            step_ids = torch.full((B, T), step, dtype=torch.long, device=x.device)
            step_embed = self.step_embeddings(step_ids)
            x = self.ut_block(x, key_padding_mask=key_padding_mask, step_embed=step_embed)

            p = torch.sigmoid(self.act_fc(x)).squeeze(-1)  # [B, T]
            if mask is not None:
                p = p * mask.to(p.dtype)

            # Mask out tokens that have already halted
            p = p * still_running.to(p.dtype)

            new_halted = (halting_prob + p > ACT_THRESHOLD) & still_running
            still_running = still_running & ~new_halted

            # Remainder for units that halt at this step
            remainders = torch.where(
                new_halted,
                1.0 - halting_prob,
                remainders,
            )

            # For halted units, we use remainder; for running units, we use p
            update_prob = torch.where(new_halted, remainders, p)
            halting_prob = halting_prob + update_prob
            n_updates = n_updates + still_running.to(x.dtype) + new_halted.to(x.dtype)

            # Accumulate weighted state
            weight = update_prob.unsqueeze(-1)
            previous_state = previous_state + weight * x

            if not still_running.any():
                break

        enc = self.dropout(previous_state)
        if mask is not None:
            token_counts = mask.sum(dim=1) + 1e-6
            act_loss = self.act_lambda * ((n_updates * mask.to(n_updates.dtype)).sum(dim=1) / token_counts).mean()
        else:
            act_loss = self.act_lambda * n_updates.mean()

        return enc, act_loss

    def _emissions(
        self,
        word_ids: torch.Tensor,
        pos_ids: torch.Tensor,
        feats: torch.Tensor,
        bigram_ids: torch.Tensor,
        trigram_ids: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute emission scores for a batch.

        Args:
            word_ids:   [B, T]
            pos_ids:    [B, T]
            feats:      [B, T, FEATURE_DIM]
            bigram_ids: [B, T]
            trigram_ids:[B, T]
            mask:       [B, T] bool
        Returns:
            emissions:  [B, T, C]
            act_loss:   scalar tensor
        """
        enc, act_loss = self._encode(word_ids, pos_ids, mask)
        base_emissions = self.hidden2tag(enc)

        # Direct feature-to-tag path
        feat_scores = self.feat2tag(feats)

        # POS n-gram path (bigram + trigram embeddings)
        bg_emb = self.bigram_emb(bigram_ids)
        tg_emb = self.trigram_emb(trigram_ids)
        ngram_cat = torch.cat([bg_emb, tg_emb], dim=-1)
        ngram_scores = self.ngram2tag(ngram_cat)

        emissions = base_emissions + feat_scores + ngram_scores
        return emissions, act_loss

    def neg_log_likelihood(
        self,
        word_ids: torch.Tensor,
        pos_ids: torch.Tensor,
        feats: torch.Tensor,
        bigram_ids: torch.Tensor,
        trigram_ids: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Negative log-likelihood loss for a batch.

        Uses the CRF log-likelihood objective when available, falling
        back to masked token-level cross-entropy otherwise. Also adds
        ACT regularization loss when ACT is enabled and an auxiliary
        cost-sensitive CE loss that upweights ultra-rare labels
        (INTJ/LST) to improve macro F1.
        """
        emissions, act_loss = self._emissions(word_ids, pos_ids, feats, bigram_ids, trigram_ids, mask)
        if self.use_crf and self.crf is not None:
            if mask is not None:
                loss = -self.crf(emissions, tags, mask=mask, reduction="token_mean")
            else:
                loss = -self.crf(emissions, tags, reduction="token_mean")
        else:
            B, T, C = emissions.shape
            if mask is not None:
                emissions = emissions[mask]
                tags = tags[mask]
            loss = self.loss_fn(emissions.view(-1, C), tags.view(-1))

        # Auxiliary per-token CE with label-dependent weights to
        # penalize mistakes on ultra-rare labels more heavily.
        B, T, C = emissions.shape
        emissions_flat = emissions.view(B * T, C)
        tags_flat = tags.view(B * T)
        if mask is not None:
            mask_flat = mask.view(B * T)
            emissions_flat = emissions_flat[mask_flat]
            tags_flat = tags_flat[mask_flat]
        aux_ce = F.cross_entropy(
            emissions_flat,
            tags_flat,
            weight=self.aux_ce_weight,
            reduction="mean",
        )

        return loss + act_loss + self.aux_ce_lambda * aux_ce

    def decode(
        self,
        word_ids: torch.Tensor,
        pos_ids: torch.Tensor,
        feats: torch.Tensor,
        bigram_ids: torch.Tensor,
        trigram_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> List[List[int]]:
        """Decode best tag sequence for each sentence in the batch."""
        emissions, _ = self._emissions(word_ids, pos_ids, feats, bigram_ids, trigram_ids, mask)
        if self.use_crf and self.crf is not None:
            if mask is not None:
                return self.crf.decode(emissions, mask=mask)
            return self.crf.decode(emissions)
        # Greedy decode if CRF unavailable
        return emissions.argmax(dim=-1).tolist()


# ---------------------------------------------------------------------------
# Training / evaluation pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    start = time.time()

    # Set up tee logging so that everything printed to stdout is also
    # captured in a timestamped log file.
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = f"train_log_UT-CRF_{timestamp}.txt"
    tee = TeeStdout(log_path)
    old_stdout = sys.stdout
    sys.stdout = tee  # type: ignore[assignment]

    try:
        # Reuse random seed and train/dev split behaviour
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

    # Train/dev split
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

    # ------------------------------------------------------------------
    # Oversample sentences containing rare and medium-rare BIO chunk tags
    # ------------------------------------------------------------------
    tag_counts = compute_chunk_tag_counts(train_data)
    ultra_tags, medium_tags = split_rare_medium_tags(
        tag_counts,
        ultra_threshold=50,   # ultra-rare: e.g. LST, INTJ, UCP, CONJP
        medium_threshold=300, # medium-rare: e.g. ADJP, ADVP, PRT, SBAR BIO tags
    )
    # Extra-strong oversampling specifically for INTJ/LST sentences
    intj_lst_tags = [
        tag for tag in ultra_tags
        if "INTJ" in tag or "LST" in tag
    ]
    if intj_lst_tags:
        train_sents = oversample_rare_sentences(train_sents, intj_lst_tags, factor=40)

    # Strong oversampling for remaining ultra-rare tags, milder for
    # medium-rare labels.
    remaining_ultra_tags = [t for t in ultra_tags if t not in intj_lst_tags]
    train_sents = oversample_rare_sentences(train_sents, remaining_ultra_tags, factor=8)
    train_sents = oversample_rare_sentences(train_sents, medium_tags, factor=3)

    # Datasets / loaders
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

    # Model and optimizer
    model = UTCRFChunker(
        vocab_size=len(token2id),
        pos_vocab_size=len(pos2id),
        tagset_size=len(tag2id),
        token2id=token2id,
        tag2id=tag2id,
        word_emb_dim=WORD_EMB_DIM,
        pos_emb_dim=POS_EMB_DIM,
        d_model=D_MODEL,
        num_steps=UT_NUM_STEPS,
        n_heads=UT_NUM_HEADS,
        d_ff=UT_FF_DIM,
        dropout=DROPOUT,
        use_pos=True,
        use_char=True,
        use_act=True,
    ).to(DEVICE)

    # Initialize word embeddings: prefer SENNA for UT if enabled,
    # otherwise fall back to the shared GloVe initialiser.
    if USE_SENNA_FOR_UT:
        init_word_embeddings_from_senna(model.word_embeds, token2id, SENNA_PATH, SENNA_EMB_DIM)
    elif USE_PRETRAINED and GLOVE_EMB_DIM == model.word_embeds.embedding_dim:
        init_word_embeddings_from_glove(model.word_embeds, token2id, GLOVE_PATH, GLOVE_EMB_DIM)
    elif USE_PRETRAINED:
        print("[glove] Skipping GloVe init for UT model: embedding dim mismatch.")

    # Initialize CRF biases for rare classes (same heuristic)
    init_crf_bias(model, tag2id, train_sents)

    # Apply a small emission bias so ultra-rare and medium-rare labels
    # are not completely dominated by very frequent ones like NP/PP/VP.
    apply_emission_bias(model, tag2id, ultra_tags, medium_tags, ultra_bias=1.0, medium_bias=0.2)

    if not HAS_TORCHCRF:
        print("[warning] torchcrf not available; using UT + softmax (no CRF layer)")

    # AdamW is generally better suited for Transformer-style encoders
    # than high-LR SGD used in the BiLSTM paper setup.
    optimizer = AdamW(
        model.parameters(),
        lr=UT_LEARNING_RATE,
        weight_decay=UT_WEIGHT_DECAY,
        betas=(0.9, 0.999),
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE,
    )

        print("\n" + "=" * 60)
        print("  STEP 3: Training Universal Transformer-CRF (dev early stopping)")
        print("=" * 60)
        print(
            f"  LR={UT_LEARNING_RATE}, batch_size={BATCH_SIZE_TRAIN}, d_model={D_MODEL}, "
            f"steps={UT_NUM_STEPS}, heads={UT_NUM_HEADS}, dropout={DROPOUT}"
        )

    best_dev_f1 = -1.0
    best_state = None
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0.0

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

        # Dev evaluation
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

        dev_results = evaluate_model(dev_true, dev_pred, "UT-CRF Dev")
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
    # Retrain on full training data for best_epoch
    # ------------------------------------------------------------------
        print("\n" + "=" * 60)
        print("  STEP 3b: Retraining UT-CRF on full training data")
        print("=" * 60)
        print(f"[full-train] Using best_epoch = {best_epoch}")

    # Apply the same INTJ/LST-focused oversampling for full-train as
    # for the dev-early-stopping phase, so the final model also sees
    # many more examples of these ultra-rare labels.
    full_train_sents = list(train_data)
    if intj_lst_tags:
        full_train_sents = oversample_rare_sentences(full_train_sents, intj_lst_tags, factor=40)

    full_train_dataset = ChunkingDataset(full_train_sents, token2id, pos2id, tag2id)
    full_train_loader = DataLoader(
        full_train_dataset,
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=True,
        collate_fn=collate_fn,
    )

    model_full = UTCRFChunker(
        vocab_size=len(token2id),
        pos_vocab_size=len(pos2id),
        tagset_size=len(tag2id),
        token2id=token2id,
        tag2id=tag2id,
        word_emb_dim=WORD_EMB_DIM,
        pos_emb_dim=POS_EMB_DIM,
        d_model=D_MODEL,
        num_steps=UT_NUM_STEPS,
        n_heads=UT_NUM_HEADS,
        d_ff=UT_FF_DIM,
        dropout=DROPOUT,
        use_pos=True,
        use_char=True,
        use_act=True,
    ).to(DEVICE)

    if USE_SENNA_FOR_UT:
        init_word_embeddings_from_senna(model_full.word_embeds, token2id, SENNA_PATH, SENNA_EMB_DIM)
    elif USE_PRETRAINED and GLOVE_EMB_DIM == model_full.word_embeds.embedding_dim:
        init_word_embeddings_from_glove(model_full.word_embeds, token2id, GLOVE_PATH, GLOVE_EMB_DIM)
    elif USE_PRETRAINED:
        print("[glove] Skipping GloVe init for full-train UT model: embedding dim mismatch.")

    init_crf_bias(model_full, tag2id, train_data)

    # Mirror the emission bias used in the dev-trained model so the
    # full-train model keeps the same rare-label prior.
    apply_emission_bias(model_full, tag2id, ultra_tags, medium_tags, ultra_bias=1.0, medium_bias=0.2)

    optimizer_full = AdamW(
        model_full.parameters(),
        lr=UT_LEARNING_RATE,
        weight_decay=UT_WEIGHT_DECAY,
        betas=(0.9, 0.999),
    )
    scheduler_full = ReduceLROnPlateau(
        optimizer_full,
        mode="min",
        factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE,
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

    # Use full-train model for final evaluation
    model = model_full

        print("\n" + "=" * 60)
        print("  STEP 4: Evaluation on test set (UT-CRF)")
        print("=" * 60)

    model.eval()
    pred_labels: List[List[str]] = []
    true_labels: List[List[str]] = []

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

            feat_mat, bigram_ids, trigram_ids = _build_features(tokens, pos_tags)
            feat_mat = feat_mat.unsqueeze(0).to(DEVICE)
            bigram_ids = bigram_ids.unsqueeze(0).to(DEVICE)
            trigram_ids = trigram_ids.unsqueeze(0).to(DEVICE)

            mask = torch.ones_like(word_ids, dtype=torch.bool)

            pred_seq_ids = model.decode(word_ids, pos_ids, feat_mat, bigram_ids, trigram_ids, mask=mask)[0]
            pred_seq_tags = [id2tag[i] for i in pred_seq_ids]

            pred_labels.append(pred_seq_tags)
            true_labels.append(chunk_tags)

    ut_results = evaluate_model(true_labels, pred_labels, "UT-CRF Chunker")
    print_detailed_report(true_labels, pred_labels, "UT-CRF Chunker")

    print_comparison_table({
        "UT-CRF": ut_results,
    })

    # Save results and model snapshot
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    backend = "UT-CRF" if HAS_TORCHCRF else "UT"

    results_path = f"results_{backend}_{timestamp}.txt"
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("PHRASE CHUNKING RESULTS (Universal Transformer + CRF)\n")
        f.write(f"Timestamp      : {timestamp}\n")
        f.write(f"Model backend  : {backend}\n")
        f.write("Dataset        : CoNLL-2000 (local JSON)\n\n")

        f.write("UT-CRF Chunker:\n")
        f.write(f"  Accuracy : {ut_results['accuracy']:.4f}\n")
        f.write(f"  Precision: {ut_results['precision']:.4f}\n")
        f.write(f"  Recall   : {ut_results['recall']:.4f}\n")
        f.write(f"  F1       : {ut_results['f1']:.4f}\n")

        print(f"[ut_crf_chunker] Saved metrics to {results_path}")

    model_path = f"ut_crf_chunker_{backend}_{timestamp}.pt"
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
        print(f"[ut_crf_chunker] Saved model to {model_path}")

        elapsed = time.time() - start
        print(f"[ut_crf_chunker] Total time: {elapsed:.1f}s")
        print(f"[ut_crf_chunker] Training log captured in {log_path}")

    finally:
        # Restore original stdout and close log file
        sys.stdout = old_stdout  # type: ignore[assignment]
        tee.close()


if __name__ == "__main__":
    main()
