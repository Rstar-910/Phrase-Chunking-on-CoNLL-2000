"""
feature_engineering.py
----------------------
Extracts token-level features for the ML correction model.
Each token is described by its text, POS tag, rule-based chunk prediction,
contextual POS tags (previous / next), contextual chunk prediction,
and simple orthographic features.

Features are returned as a list of dicts suitable for sklearn DictVectorizer.
"""


def _word_shape(token):
    """
    Produce a simplified word-shape string.
    E.g. 'Hello' -> 'Xxxxx', '3.14' -> 'd.dd', 'U.S.' -> 'X.X.'
    """
    shape = []
    for ch in token:
        if ch.isupper():
            shape.append("X")
        elif ch.islower():
            shape.append("x")
        elif ch.isdigit():
            shape.append("d")
        else:
            shape.append(ch)
    return "".join(shape)


def extract_features_sentence(tokens, pos_tags, rule_chunks):
    """
    Build a feature dict for every token in a single sentence.

    Parameters
    ----------
    tokens      : list[str]  – words
    pos_tags    : list[str]  – POS labels
    rule_chunks : list[str]  – rule-based BIO predictions

    Returns
    -------
    features : list[dict]
        One dict per token, ready for DictVectorizer.
    """
    features = []
    n = len(tokens)

    for i in range(n):
        token = tokens[i]
        token_lower = token.lower()
        feat = {
            # ── core features ───────────────────────────────
            "token_lower":   token_lower,
            "pos":           pos_tags[i],
            "rule_chunk":    rule_chunks[i],

            # ── wider POS context ──────────────────────────────
            "prev_pos":      pos_tags[i - 1] if i > 0     else "<START>",
            "next_pos":      pos_tags[i + 1] if i < n - 1 else "<END>",
            "prev2_pos":     pos_tags[i - 2] if i > 1 else "<START>",
            "next2_pos":     pos_tags[i + 2] if i < n - 2 else "<END>",

            # ── contextual rule chunk ───────────────────────
            "prev_rule_chunk": rule_chunks[i - 1] if i > 0 else "<START>",
            "next_rule_chunk": rule_chunks[i + 1] if i < n - 1 else "<END>",

            # ── orthographic features ───────────────────────
            "is_capitalized": token[0].isupper() if token else False,
            "is_numeric":     token.isdigit(),
            "is_punct":       not token.isalnum(),
            "word_length":    min(len(token), 20),   # capped to avoid sparsity

            # ── word shape ──────────────────────────────────
            "word_shape":     _word_shape(tokens),

            # prefixes / suffixes
            "prefix_2": token_lower[:2],
            "prefix_3": token_lower[:3],
            "suffix_2": token_lower[-2:],
            "suffix_3": token_lower[-3:],

            # simple lexical flags
            "is_det_word":    token_lower in {"the", "a", "an", "this", "that", "these", "those"},
            "is_coord_word":  token_lower in {"and", "or", "but"},
            "is_prep_word":   token_lower in {"of", "in", "on", "at", "by", "to", "for", "from"},

            # position in sentence
            "is_first":       i == 0,
            "is_last":        i == n - 1,
            "is_near_start":  i <= 2,
            "is_near_end":    i >= n - 3,
        }
        features.append(feat)

    return features


def extract_features(data, rule_predictions):
    """
    Extract features for an entire dataset (list of sentences).

    Parameters
    ----------
    data             : list[dict]        – each with 'tokens', 'pos_tags'
    rule_predictions : list[list[str]]   – BIO tags from the rule-based chunker

    Returns
    -------
    all_features : list[dict]
        Flat list of feature dicts (one per token across all sentences).
    all_labels   : list[str]
        Flat list of true chunk labels (one per token).
    sentence_lengths : list[int]
        Number of tokens per sentence (used to reconstruct sentence structure).
    """
    all_features = []
    all_labels   = []
    sentence_lengths = []

    for sent, rule_bio in zip(data, rule_predictions):
        tokens   = sent["tokens"]
        pos_tags = sent["pos_tags"]
        labels   = sent["chunk_tags"]

        feats = extract_features_sentence(tokens, pos_tags, rule_bio)
        all_features.extend(feats)
        all_labels.extend(labels)
        sentence_lengths.append(len(tokens))

    return all_features, all_labels, sentence_lengths


# ── quick sanity check ────────────────────────────────────────────────
if __name__ == "__main__":
    tokens   = ["The", "cat", "sat", "on", "the", "mat", "."]
    pos_tags = ["DT",  "NN",  "VBD", "IN", "DT",  "NN",  "."]
    rule_bio = ["B-NP","I-NP","B-VP","B-PP","B-NP","I-NP","O"]

    feats = extract_features_sentence(tokens, pos_tags, rule_bio)
    for f in feats:
        print(f)
