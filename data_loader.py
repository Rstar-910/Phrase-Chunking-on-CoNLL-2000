"""data_loader.py
-----------------
Load the CoNLL-2000 dataset from local JSONL exports, convert integer
POS/chunk IDs to human-readable labels using the official metadata in
``conll2000_local/train/dataset_info.json`` when available, and return
sentence-wise structured data for train and test splits.
"""

import json
import os


# ── Fallback tag-name look-up tables ─────────────────────────────────
# These are only used if we cannot read the canonical label lists from
# the dataset_info.json produced by HuggingFace ``datasets``.

FALLBACK_POS_TAGS = [
    "''", '#', '$', '(', ')', ',', '.', ':', '``',
    'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS',
    'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'POS', 'PRP',
    'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH',
    'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP',
    'WP$', 'WRB',
]

FALLBACK_CHUNK_TAGS = [
    'O',
    'B-ADJP', 'I-ADJP',
    'B-ADVP', 'I-ADVP',
    'B-CONJP', 'I-CONJP',
    'B-INTJ', 'I-INTJ',
    'B-LST', 'I-LST',
    'B-NP', 'I-NP',
    'B-PP', 'I-PP',
    'B-PRT', 'I-PRT',
    'B-SBAR', 'I-SBAR',
    'B-UCP', 'I-UCP',
    'B-VP', 'I-VP',
]


def _id_to_label(ids, label_list):
    """Map a list of integer IDs to their string labels."""
    return [label_list[i] for i in ids]


def load_conll2000(hf_name="eriktks/conll2000", local_path=None):  # hf_name/local_path kept for API compat
    """Load CoNLL-2000 from local JSONL files and return sentence-level data.

    Returns
    -------
    train_data, test_data : list of dict
        Each element is a dict with keys:
            'tokens'     - list[str]
            'pos_tags'   - list[str]   (e.g. 'NN', 'DT', …)
            'chunk_tags' - list[str]   (e.g. 'B-NP', 'I-VP', 'O', …)
    """
    base_dir = os.path.dirname(__file__)
    train_path = os.path.join(base_dir, "conll2000_train.json")
    test_path = os.path.join(base_dir, "conll2000_test.json")

    # Try to read canonical label names from the local dataset_info.json
    pos_names = FALLBACK_POS_TAGS
    chunk_names = FALLBACK_CHUNK_TAGS
    info_path = os.path.join(base_dir, "conll2000_local", "train", "dataset_info.json")
    try:
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        pos_names = info["features"]["pos_tags"]["feature"]["names"]
        chunk_names = info["features"]["chunk_tags"]["feature"]["names"]
    except Exception:
        # Fall back silently; POS/CHUNK mapping will still be reasonable.
        pass

    def _load_split(path):
        sentences = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)

                tokens = row["tokens"]
                pos_ids = row["pos_tags"]
                chunk_ids = row["chunk_tags"]

                # Skip empty or malformed sentences to avoid issues in
                # downstream rule-based parsing and evaluation.
                if not tokens:
                    continue
                if not (len(tokens) == len(pos_ids) == len(chunk_ids)):
                    # Optionally, log a warning here.
                    continue

                sentences.append({
                    "tokens":     tokens,
                    "pos_tags":   _id_to_label(pos_ids,   pos_names),
                    "chunk_tags": _id_to_label(chunk_ids, chunk_names),
                })
        return sentences

    train_data = _load_split(train_path)
    test_data = _load_split(test_path)

    print(f"[data_loader] Loaded {len(train_data)} train sentences, "
          f"{len(test_data)} test sentences.")
    return train_data, test_data

# ── quick sanity check ────────────────────────────────────────────────
if __name__ == "__main__":
    train, test = load_conll2000()
    s = train[0]
    for tok, pos, chk in zip(s["tokens"], s["pos_tags"], s["chunk_tags"]):
        print(f"{tok:20s} {pos:6s} {chk}")
