"""
rule_chunker.py
---------------
Rule-based phrase chunker using NLTK RegexpParser.
Defines grammar rules for NP, VP, PP, ADJP, and ADVP,
then converts the resulting chunk trees into BIO-tagged sequences.
"""

import nltk
from nltk import RegexpParser

# Download required NLTK data (only needed once)
nltk.download("punkt", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)


# ── Phrase-structure grammar rules ────────────────────────────────────
# The order matters: earlier rules have higher priority.
# We define rules that cover the most common POS-tag patterns found
# in the CoNLL-2000 Wall Street Journal data.

GRAMMAR = r"""
    NP: {<DT|PRP\$|PDT>?<JJ|JJR|JJS>*<NN|NNS|NNP|NNPS|PRP|CD|EX>+}
        {<PRP>}
        {<DT>}
    VP: {<MD>?<RB|RBR|RBS>*<VB|VBD|VBG|VBN|VBP|VBZ>+<RP>?}
    PP: {<IN|TO>}
    ADJP: {<RB|RBR|RBS>*<JJ|JJR|JJS>+}
    ADVP: {<RB|RBR|RBS>+}
"""

_parser = RegexpParser(GRAMMAR)


def _tree_to_bio(tree, tokens):
    """
    Convert an NLTK chunk Tree to a flat BIO-tag list.

    Parameters
    ----------
    tree : nltk.Tree
        Result of RegexpParser.parse().
    tokens : list[str]
        Original token list (used only to verify alignment).

    Returns
    -------
    bio_tags : list[str]
        BIO tags aligned with the input tokens.
    """
    bio_tags = []
    for subtree in tree:
        if isinstance(subtree, nltk.Tree):
            # Named chunk (NP, VP, PP, …)
            label = subtree.label()
            for i, (word, _pos) in enumerate(subtree.leaves()):
                tag = f"B-{label}" if i == 0 else f"I-{label}"
                bio_tags.append(tag)
        else:
            # Token outside any chunk
            bio_tags.append("O")
    return bio_tags


def predict_sentence(tokens, pos_tags):
    """
    Apply rule-based chunking to a single sentence.

    Parameters
    ----------
    tokens   : list[str]   – words
    pos_tags : list[str]   – POS labels (e.g. 'NN', 'DT')

    Returns
    -------
    bio_tags : list[str]   – predicted BIO labels
    """
    tagged = list(zip(tokens, pos_tags))
    tree = _parser.parse(tagged)
    return _tree_to_bio(tree, tokens)


def predict(data):
    """
    Batch prediction over a list of sentences.

    Parameters
    ----------
    data : list[dict]
        Each dict has 'tokens' and 'pos_tags'.

    Returns
    -------
    predictions : list[list[str]]
        BIO-tag sequences for every sentence.
    """
    predictions = []
    for sent in data:
        bio = predict_sentence(sent["tokens"], sent["pos_tags"])
        predictions.append(bio)
    return predictions


# ── quick test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    tokens   = ["The", "cat", "sat", "on", "the", "mat", "."]
    pos_tags = ["DT",  "NN",  "VBD", "IN", "DT",  "NN",  "."]
    bio = predict_sentence(tokens, pos_tags)
    for t, p, b in zip(tokens, pos_tags, bio):
        print(f"{t:10s} {p:5s} {b}")
