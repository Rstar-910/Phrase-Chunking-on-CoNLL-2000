"""
ml_model.py
-----------
Machine-learning correction model that refines the rule-based chunker's
predictions.  Uses Logistic Regression with DictVectorizer for token-level
multi-class classification.

The model learns systematic error patterns from the rule-based output and
produces corrected BIO chunk predictions.
"""

# from sklearn.feature_extraction import DictVectorizer
# from sklearn.linear_model import LogisticRegression


# class HybridChunkCorrector:
#     """
#     Train a token-level classifier to correct rule-based chunk predictions.

#     Workflow
#     --------
#     1. Fit DictVectorizer + LogisticRegression on training features/labels.
#     2. At prediction time, transform features and output corrected labels.
#     """

#     def __init__(self, max_iter=1000, C=1.0, solver="lbfgs", verbose=2):
#         """Initialize the hybrid correction model.

#         Parameters
#         ----------
#         max_iter : int
#             Maximum number of iterations for LogisticRegression.
#         C : float
#             Inverse of regularization strength.
#         solver : str
#             Optimization algorithm used by LogisticRegression.
#         verbose : int
#             Verbosity level forwarded to LogisticRegression so that
#             the user can see progress information during training.
#         """

#         self.vectorizer = DictVectorizer(sparse=True)
#         self.classifier = LogisticRegression(
#             max_iter=max_iter,
#             C=C,
#             solver=solver,
#             verbose=verbose,
#             random_state=42,
#         )

#     def fit(self, train_features, train_labels):
#         """
#         Fit the correction model.

#         Parameters
#         ----------
#         train_features : list[dict]  – flat list of feature dicts (one per token)
#         train_labels   : list[str]   – flat list of true BIO labels
#         """
#         print("[ml_model] Vectorizing training features …")
#         X_train = self.vectorizer.fit_transform(train_features)
#         print(f"[ml_model] Feature matrix shape: {X_train.shape}")

#         print("[ml_model] Training Logistic Regression …")
#         self.classifier.fit(X_train, train_labels)
#         print("[ml_model] Training complete.")

#     def predict_flat(self, test_features):
#         """
#         Predict corrected labels for a flat list of token features.

#         Returns
#         -------
#         predictions : list[str]
#         """
#         X_test = self.vectorizer.transform(test_features)
#         return list(self.classifier.predict(X_test))

#     def predict_sentences(self, test_features, sentence_lengths):
#         """
#         Predict and restructure output back into sentence-level lists.

#         Parameters
#         ----------
#         test_features    : list[dict]
#         sentence_lengths : list[int]

#         Returns
#         -------
#         predictions : list[list[str]]
#             One BIO-tag list per sentence.
#         """
#         flat_preds = self.predict_flat(test_features)

#         # Rebuild sentence structure
#         sentences = []
#         idx = 0
#         for length in sentence_lengths:
#             sentences.append(flat_preds[idx : idx + length])
#             idx += length
#         return sentences


from sklearn_crfsuite import CRF


class HybridChunkCorrector:
    """
    Train a token-level CRF to correct rule-based chunk predictions.
    Uses the same dict features as before, but models label sequences
    instead of independent tokens.
    """

    def __init__(
        self,
        algorithm="lbfgs",
        c1=0.1,
        c2=0.2,
        max_iterations=500,
        all_possible_transitions=True,
    ):
        self.crf = CRF(
            algorithm=algorithm,
            c1=c1,
            c2=c2,
            max_iterations=max_iterations,
            all_possible_transitions=all_possible_transitions,
        )

    @staticmethod
    def _unflatten(features, labels, lengths):
        """
        Turn flat token lists into per-sentence sequences.
        """
        X_seq, y_seq = [], []
        idx = 0
        for length in lengths:
            X_seq.append(features[idx : idx + length])
            y_seq.append(labels[idx : idx + length])
            idx += length
        return X_seq, y_seq

    def fit(self, train_features, train_labels, train_lengths):
        """
        Fit the CRF correction model.

        Parameters
        ----------
        train_features : list[dict]  – flat list (one per token)
        train_labels   : list[str]   – flat BIO labels
        train_lengths  : list[int]   – tokens per sentence
        """
        X_train, y_train = self._unflatten(train_features, train_labels, train_lengths)
        print("[ml_model] Training CRF …")
        self.crf.fit(X_train, y_train)
        print("[ml_model] Training complete.")

    def predict_sentences(self, test_features, sentence_lengths):
        """
        Predict BIO sequences for each sentence.

        Parameters
        ----------
        test_features    : list[dict] – flat list (one per token)
        sentence_lengths : list[int]

        Returns
        -------
        predictions : list[list[str]]
        """
        # Rebuild sentence feature sequences
        X_test, _ = self._unflatten(test_features, [None] * len(sentence_lengths), sentence_lengths)
        return self.crf.predict(X_test)

# ── quick test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Tiny smoke test
    feats  = [{"pos": "DT", "rule_chunk": "B-NP"}, {"pos": "NN", "rule_chunk": "I-NP"}]
    labels = ["B-NP", "I-NP"]
    model = HybridChunkCorrector(max_iter=100)
    model.fit(feats, labels)
    preds = model.predict_flat(feats)
    print("Predictions:", preds)
