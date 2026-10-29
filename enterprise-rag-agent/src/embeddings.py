"""
Embedding layer with a single unified interface (`fit` + `encode`) that
works whether we're using a real sentence-transformers model or the
TF-IDF fallback. This means the rest of the app never needs to know or
care which backend is active.

Why a fallback? sentence-transformers needs to download model weights
from the internet on first use. In locked-down or fully offline
environments that download can fail. Rather than crashing, the app
degrades gracefully to a scikit-learn TF-IDF vectorizer, which needs
no downloads at all and still gives reasonable semantic-ish retrieval
for keyword-heavy enterprise documents.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import List

import numpy as np

import config


class EmbeddingProvider:
    def __init__(self):
        self.backend = None   # "sentence-transformer" | "tfidf"
        self._model = None
        self._vectorizer = None
        self._fitted = False
        self._load_backend()

    # ------------------------------------------------------------------
    def _load_backend(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(config.SENTENCE_TRANSFORMER_MODEL)
            self.backend = "sentence-transformer"
        except Exception:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._vectorizer = TfidfVectorizer(
                max_features=20000, ngram_range=(1, 2), stop_words="english"
            )
            self.backend = "tfidf"

    # ------------------------------------------------------------------
    def fit(self, corpus: List[str]):
        """Only meaningful for the TF-IDF backend; no-op for transformers."""
        if self.backend == "tfidf":
            self._vectorizer.fit(corpus)
        self._fitted = True

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")

        if self.backend == "sentence-transformer":
            vecs = self._model.encode(
                texts, convert_to_numpy=True, show_progress_bar=False
            )
        else:
            if not self._fitted:
                # Encoding before fit (e.g. a lone query) -- fit lazily.
                self.fit(texts)
            vecs = self._vectorizer.transform(texts).toarray()

        vecs = vecs.astype("float32")
        # L2-normalize so inner product == cosine similarity.
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        return vecs / norms

    @property
    def dim(self) -> int:
        if self.backend == "sentence-transformer":
            return self._model.get_sentence_embedding_dimension()
        if self._vectorizer is not None and hasattr(self._vectorizer, "vocabulary_"):
            return len(self._vectorizer.vocabulary_)
        return 384  # placeholder before fitting

    # ------------------------------------------------------------------
    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "backend": self.backend,
                    "vectorizer": self._vectorizer if self.backend == "tfidf" else None,
                    "fitted": self._fitted,
                },
                f,
            )

    def load_state(self, path: Path):
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.backend = state["backend"]
        self._fitted = state["fitted"]
        if self.backend == "tfidf":
            self._vectorizer = state["vectorizer"]
        else:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(config.SENTENCE_TRANSFORMER_MODEL)
