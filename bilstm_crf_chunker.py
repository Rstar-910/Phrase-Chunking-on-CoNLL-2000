"""bilstm_crf_chunker.py
-------------------------
Neural BiLSTM-CRF phrase chunker for CoNLL-2000, using the same
data_loader and evaluation utilities as the hybrid rule-based + CRF
system. This model ignores the rule-based predictions and learns
chunking directly from tokens (and POS tags) via embeddings.

This version uses mini-batch training with padding and an explicit
train/dev split with early stopping on dev F1 to speed up training
and avoid overfitting.

Usage (from project root):

    uv run python bilstm_crf_chunker.py

This will train the model and evaluate on the held-out test split,
printing metrics and saving results + model to disk.
"""

import time
import random
import os
from datetime import datetime
from typing import List, Dict

import torch
import torch.nn as nn
from torch.optim import SGD
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

# Path to a GloVe file, e.g. "glove.6B.100d.txt" placed in the project root.
GLOVE_PATH = "glove.6B.100d.txt"
GLOVE_EMB_DIM = 100


def load_glove_embeddings(path: str, embedding_dim: int) -> Dict[str, torch.Tensor]:
	"""Load GloVe embeddings from a text file."""
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


def init_word_embeddings_from_glove(embedding_layer: nn.Embedding, token2id: Dict[str, int],
									   glove_path: str, embedding_dim: int) -> None:
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


def build_vocabs(train_data, test_data=None):
	"""Build token, POS, and chunk-label vocabularies from data.

	Tokens/POS tags come from train; labels come from train and test so
	that rare tags only present in the test split (e.g. I-LST) are
	still recognized.
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
	token2id = {"<PAD>": 0, "<UNK>": 1}
	for tok in sorted(token_set):
		token2id.setdefault(tok, len(token2id))

	pos2id = {"<PAD>": 0}
	for pos in sorted(pos_set):
		pos2id.setdefault(pos, len(pos2id))

	tag2id = {}
	for tag in sorted(tag_set):
		tag2id.setdefault(tag, len(tag2id))

	id2tag = {i: t for t, i in tag2id.items()}
	return token2id, pos2id, tag2id, id2tag


def encode_sentence(tokens: List[str], pos_tags: List[str], chunk_tags: List[str],
					token2id: Dict[str, int], pos2id: Dict[str, int], tag2id: Dict[str, int],
					char2id: Dict[str, int]):
	"""Convert a single sentence to ID tensors (including chars)."""
	token_ids = [token2id.get(tok, token2id["<UNK>"]) for tok in tokens]
	pos_ids = [pos2id.get(pos, pos2id["<PAD>"]) for pos in pos_tags]
	tag_ids = [tag2id[tag] for tag in chunk_tags]

	# Character IDs per token (no padding here; handled per-call)
	char_id_seqs = []
	for tok in tokens:
		chars = [char2id.get(ch, char2id["<UNK>"]) for ch in tok]
		char_id_seqs.append(chars)

	# Pad characters within this sentence so we can create a tensor
	max_char_len = max(len(c) for c in char_id_seqs) if char_id_seqs else 1
	char_ids_padded = [
		seq + [char2id["<PAD>"]] * (max_char_len - len(seq)) for seq in char_id_seqs
	]

	return (
		torch.tensor(token_ids, dtype=torch.long, device=DEVICE),
		torch.tensor(pos_ids, dtype=torch.long, device=DEVICE),
		torch.tensor(tag_ids, dtype=torch.long, device=DEVICE),
		torch.tensor(char_ids_padded, dtype=torch.long, device=DEVICE),
	)


class ChunkingDataset(Dataset):
	"""PyTorch Dataset for sentence-level chunking data."""

	def __init__(self, data, token2id, pos2id, tag2id, char2id):
		self.data = data
		self.token2id = token2id
		self.pos2id = pos2id
		self.tag2id = tag2id
		self.char2id = char2id

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		sent = self.data[idx]
		tokens = sent["tokens"]
		pos_tags = sent["pos_tags"]
		chunk_tags = sent["chunk_tags"]

		word_ids = [self.token2id.get(tok, self.token2id["<UNK>"]) for tok in tokens]
		pos_ids = [self.pos2id.get(pos, self.pos2id["<PAD>"]) for pos in pos_tags]
		tag_ids = [self.tag2id[tag] for tag in chunk_tags]

		# Character ids per token (list of lists; padded in collate_fn)
		char_id_seqs = []
		for tok in tokens:
			chars = [self.char2id.get(ch, self.char2id["<UNK>"]) for ch in tok]
			char_id_seqs.append(chars)

		return (
			torch.tensor(word_ids, dtype=torch.long),
			torch.tensor(pos_ids, dtype=torch.long),
			torch.tensor(tag_ids, dtype=torch.long),
			char_id_seqs,
		)


def make_collate_fn(pad_word_id: int, pad_pos_id: int, pad_char_id: int):
	"""Create a collate_fn that pads sequences in a batch."""

	def collate_fn(batch):
		word_seqs, pos_seqs, tag_seqs, char_seqs = zip(*batch)
		batch_size = len(batch)
		lengths = [len(s) for s in word_seqs]
		max_len = max(lengths)

		word_ids = torch.full(
			(batch_size, max_len), pad_word_id, dtype=torch.long, device=DEVICE
		)
		pos_ids = torch.full(
			(batch_size, max_len), pad_pos_id, dtype=torch.long, device=DEVICE
		)
		# Tag padding value doesn't matter; it will be ignored via mask
		tags = torch.zeros(batch_size, max_len, dtype=torch.long, device=DEVICE)
		mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=DEVICE)

		# Character tensor: [B, T, C] where C is max word length in batch
		max_char_len = 1
		for sent_chars in char_seqs:
			for chs in sent_chars:
				if len(chs) > max_char_len:
					max_char_len = len(chs)

		char_ids = torch.full(
			(batch_size, max_len, max_char_len),
			pad_char_id,
			dtype=torch.long,
			device=DEVICE,
		)

		for i, (w, p, t, sent_chars) in enumerate(zip(word_seqs, pos_seqs, tag_seqs, char_seqs)):
			L = len(w)
			word_ids[i, :L] = w.to(DEVICE)
			pos_ids[i, :L] = p.to(DEVICE)
			tags[i, :L] = t.to(DEVICE)
			mask[i, :L] = True

			for j, chs in enumerate(sent_chars):
				if j >= max_len:
					break
				clen = min(len(chs), max_char_len)
				char_ids[i, j, :clen] = torch.tensor(chs[:clen], dtype=torch.long, device=DEVICE)

		return word_ids, pos_ids, tags, mask, char_ids

	return collate_fn


class BiLSTMCRF(nn.Module):
	"""BiLSTM-CRF model for sequence tagging.

	If torchcrf is not available, this falls back to a plain
	BiLSTM + softmax tagger (no CRF). Training and decoding
	APIs remain the same.
	"""

	def __init__(
		self,
		vocab_size: int,
		pos_vocab_size: int,
		tagset_size: int,
		char_vocab_size: int,
		word_emb_dim: int = 100,
		pos_emb_dim: int = 25,
		char_emb_dim: int = 30,
		char_hidden_dim: int = 50,
		hidden_dim: int = 300,
		num_layers: int = 2,
		dropout: float = 0.5,
	):
		super().__init__()

		self.word_embeds = nn.Embedding(vocab_size, word_emb_dim, padding_idx=0)
		self.pos_embeds = nn.Embedding(pos_vocab_size, pos_emb_dim, padding_idx=0)
		self.char_embeds = nn.Embedding(char_vocab_size, char_emb_dim, padding_idx=0)
		self.char_lstm = nn.LSTM(
			input_size=char_emb_dim,
			hidden_size=char_hidden_dim // 2,
			num_layers=1,
			bidirectional=True,
			batch_first=True,
		)

		lstm_input_dim = word_emb_dim + pos_emb_dim + char_hidden_dim
		self.lstm = nn.LSTM(
			lstm_input_dim,
			hidden_dim // 2,
			num_layers=num_layers,
			bidirectional=True,
			batch_first=True,
		)

		self.dropout = nn.Dropout(dropout)
		self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

		self.use_crf = HAS_TORCHCRF
		if self.use_crf:
			self.crf = CRF(tagset_size, batch_first=True)
		else:
			self.crf = None
			self.loss_fn = nn.CrossEntropyLoss()

	def _emissions(self, word_ids, pos_ids, char_ids):
		"""Compute emission scores for a batch of sentences."""
		word_emb = self.word_embeds(word_ids)
		pos_emb = self.pos_embeds(pos_ids)
		# char_ids: [B, T, C]
		B, T, C = char_ids.shape
		char_flat = char_ids.view(B * T, C)
		char_emb = self.char_embeds(char_flat)  # [B*T, C, char_emb_dim]
		# Run a small BiLSTM over characters and take the last hidden state
		char_outputs, _ = self.char_lstm(char_emb)  # [B*T, C, char_hidden_dim]
		char_repr = char_outputs[:, -1, :]  # [B*T, char_hidden_dim]
		char_repr = char_repr.view(B, T, -1)

		embeds = torch.cat([word_emb, pos_emb, char_repr], dim=-1)

		lstm_out, _ = self.lstm(embeds)
		lstm_out = self.dropout(lstm_out)
		emissions = self.hidden2tag(lstm_out)
		return emissions

	def neg_log_likelihood(self, word_ids, pos_ids, char_ids, tags, mask=None):
		emissions = self._emissions(word_ids, pos_ids, char_ids)
		if self.use_crf and self.crf is not None:
			if mask is not None:
				loss = -self.crf(emissions, tags, mask=mask, reduction="token_mean")
			else:
				loss = -self.crf(emissions, tags, reduction="token_mean")
		else:
			# Fall back to token-level cross-entropy
			# shapes: emissions [B, T, C], tags [B, T]
			B, T, C = emissions.shape
			if mask is not None:
				emissions = emissions[mask]
				tags = tags[mask]
			loss = self.loss_fn(emissions.view(-1, C), tags.view(-1))
		return loss

	def decode(self, word_ids, pos_ids, char_ids, mask=None):
		emissions = self._emissions(word_ids, pos_ids, char_ids)
		if self.use_crf and self.crf is not None:
			if mask is not None:
				return self.crf.decode(emissions, mask=mask)
			return self.crf.decode(emissions)
		# Greedy decode if CRF is unavailable
		return emissions.argmax(dim=-1).tolist()


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
	print("  STEP 2: Building vocabularies")
	print("=" * 60)
	token2id, pos2id, tag2id, id2tag = build_vocabs(train_data, test_data)

	# Character vocabulary (from train tokens)
	char_set = set()
	for sent in train_data:
		for tok in sent["tokens"]:
			char_set.update(tok)
	char2id = {"<PAD>": 0, "<UNK>": 1}
	for ch in sorted(char_set):
		char2id.setdefault(ch, len(char2id))

	# ── 2b. Train/dev split for early stopping ─────────────────────
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

	# ── 3. Datasets and DataLoaders ─────────────────────────────────
	train_dataset = ChunkingDataset(train_sents, token2id, pos2id, tag2id, char2id)
	dev_dataset = ChunkingDataset(dev_sents, token2id, pos2id, tag2id, char2id)

	collate_fn = make_collate_fn(token2id["<PAD>"], pos2id["<PAD>"], char2id["<PAD>"])
	train_loader = DataLoader(
		train_dataset,
		batch_size=32,
		shuffle=True,
		collate_fn=collate_fn,
	)
	dev_loader = DataLoader(
		dev_dataset,
		batch_size=64,
		shuffle=False,
		collate_fn=collate_fn,
	)

	model = BiLSTMCRF(
		vocab_size=len(token2id),
		pos_vocab_size=len(pos2id),
		tagset_size=len(tag2id),
		char_vocab_size=len(char2id),
	).to(DEVICE)

	# Initialize word embeddings from GloVe if available
	if GLOVE_EMB_DIM == model.word_embeds.embedding_dim:
		init_word_embeddings_from_glove(model.word_embeds, token2id, GLOVE_PATH, GLOVE_EMB_DIM)
	else:
		print("[glove] Skipping GloVe init: embedding dim mismatch.")

	if not HAS_TORCHCRF:
		print("[warning] torchcrf not available; using BiLSTM + softmax (no CRF layer)")

	optimizer = SGD(model.parameters(), lr=0.015, momentum=0.9)

	print("\n" + "=" * 60)
	print("  STEP 3: Training BiLSTM-CRF (batched with early stopping)")
	print("=" * 60)
	num_epochs = 50
	patience = 5
	best_dev_f1 = -1.0
	best_state = None
	best_epoch = 0
	epochs_no_improve = 0

	for epoch in range(1, num_epochs + 1):
		model.train()
		total_loss = 0.0
		for word_ids, pos_ids, tags, mask, char_ids in train_loader:
			optimizer.zero_grad()
			loss = model.neg_log_likelihood(word_ids, pos_ids, char_ids, tags, mask=mask)
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
			optimizer.step()
			total_loss += loss.item()

		avg_loss = total_loss / len(train_loader)
		print(f"[epoch {epoch}] avg training loss: {avg_loss:.4f}")

		# Dev evaluation for early stopping
		model.eval()
		dev_true = []
		dev_pred = []
		with torch.no_grad():
			for word_ids, pos_ids, tags, mask, char_ids in dev_loader:
				paths = model.decode(word_ids, pos_ids, char_ids, mask=mask)
				for i, path in enumerate(paths):
					length = int(mask[i].sum().item())
					pred_ids = path[:length]
					gold_ids = tags[i, :length].tolist()
					pred_tags = [id2tag[j] for j in pred_ids]
					gold_tags = [id2tag[j] for j in gold_ids]
					dev_pred.append(pred_tags)
					dev_true.append(gold_tags)

		dev_results = evaluate_model(dev_true, dev_pred, "BiLSTM-CRF Dev")
		dev_f1 = dev_results["f1"]
		print(f"[epoch {epoch}] dev F1: {dev_f1:.4f}")

		if dev_f1 > best_dev_f1 + 1e-4:
			best_dev_f1 = dev_f1
			best_state = model.state_dict()
			best_epoch = epoch
			epochs_no_improve = 0
		else:
			epochs_no_improve += 1
			if epochs_no_improve >= patience:
				print(f"[early stopping] No dev F1 improvement for {patience} epochs.")
				break

	if best_epoch == 0:
		best_epoch = epoch

	print(f"[early stopping] Best dev F1: {best_dev_f1:.4f} at epoch {best_epoch}")

	# ── 3b. Retrain on full training data for best_epoch ─────────────
	print("\n" + "=" * 60)
	print("  STEP 3b: Retraining on full training data")
	print("=" * 60)
	print(f"[full-train] Using best_epoch = {best_epoch}")

	full_train_dataset = ChunkingDataset(train_data, token2id, pos2id, tag2id, char2id)
	full_train_loader = DataLoader(
		full_train_dataset,
		batch_size=32,
		shuffle=True,
		collate_fn=collate_fn,
	)

	model_full = BiLSTMCRF(
		vocab_size=len(token2id),
		pos_vocab_size=len(pos2id),
		tagset_size=len(tag2id),
		char_vocab_size=len(char2id),
	).to(DEVICE)

	if GLOVE_EMB_DIM == model_full.word_embeds.embedding_dim:
		init_word_embeddings_from_glove(model_full.word_embeds, token2id, GLOVE_PATH, GLOVE_EMB_DIM)
	else:
		print("[glove] Skipping GloVe init for full-train model: embedding dim mismatch.")

	optimizer_full = SGD(model_full.parameters(), lr=0.015, momentum=0.9)

	for e in range(1, best_epoch + 1):
		model_full.train()
		total_loss = 0.0
		for word_ids, pos_ids, tags, mask, char_ids in full_train_loader:
			optimizer_full.zero_grad()
			loss = model_full.neg_log_likelihood(word_ids, pos_ids, char_ids, tags, mask=mask)
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model_full.parameters(), max_norm=5.0)
			optimizer_full.step()
			total_loss += loss.item()

		avg_loss = total_loss / len(full_train_loader)
		print(f"[full-train epoch {e}/{best_epoch}] avg training loss: {avg_loss:.4f}")

	# Use the full-data model for final evaluation
	model = model_full

	print("\n" + "=" * 60)
	print("  STEP 4: Evaluation (BiLSTM-CRF)")
	print("=" * 60)
	model.eval()

	pred_labels = []
	true_labels = []

	with torch.no_grad():
		for sent in test_data:
			tokens = sent["tokens"]
			pos_tags = sent["pos_tags"]
			chunk_tags = sent["chunk_tags"]

			word_ids, pos_ids, _, char_ids = encode_sentence(
				tokens, pos_tags, chunk_tags, token2id, pos2id, tag2id, char2id
			)
			word_ids = word_ids.unsqueeze(0)
			pos_ids = pos_ids.unsqueeze(0)
			char_ids = char_ids.unsqueeze(0)

			pred_seq_ids = model.decode(word_ids, pos_ids, char_ids)[0]
			pred_seq_tags = [id2tag[i] for i in pred_seq_ids]

			pred_labels.append(pred_seq_tags)
			true_labels.append(chunk_tags)

	bilstm_results = evaluate_model(true_labels, pred_labels, "BiLSTM-CRF Chunker")
	print_detailed_report(true_labels, pred_labels, "BiLSTM-CRF Chunker")

	print_comparison_table({
		"BiLSTM-CRF": bilstm_results,
	})

	# ── 5. Save results and model ───────────────────────────────────
	timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
	backend = "BiLSTM-CRF" if HAS_TORCHCRF else "BiLSTM"

	results_path = f"results_{backend}_{timestamp}.txt"
	with open(results_path, "w", encoding="utf-8") as f:
		f.write("PHRASE CHUNKING RESULTS (Neural Model)\n")
		f.write(f"Timestamp      : {timestamp}\n")
		f.write(f"Model backend  : {backend}\n")
		f.write("Dataset        : CoNLL-2000 (local JSON)\n\n")

		f.write("BiLSTM-CRF Chunker:\n")
		f.write(f"  Accuracy : {bilstm_results['accuracy']:.4f}\n")
		f.write(f"  Precision: {bilstm_results['precision']:.4f}\n")
		f.write(f"  Recall   : {bilstm_results['recall']:.4f}\n")
		f.write(f"  F1       : {bilstm_results['f1']:.4f}\n")

	print(f"[bilstm_crf_chunker] Saved metrics to {results_path}")

	model_path = f"bilstm_crf_chunker_{backend}_{timestamp}.pt"
	torch.save({
		"model_state_dict": model.state_dict(),
		"token2id": token2id,
		"pos2id": pos2id,
		"tag2id": tag2id,
		"char2id": char2id,
		"backend": backend,
	}, model_path)
	print(f"[bilstm_crf_chunker] Saved model to {model_path}")

	elapsed = time.time() - start
	print(f"[bilstm_crf_chunker] Total time: {elapsed:.1f}s")


if __name__ == "__main__":
	main()

