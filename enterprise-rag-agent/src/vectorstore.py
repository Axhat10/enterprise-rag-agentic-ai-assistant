"""
Hybrid retrieval store: combines a FAISS dense vector index with a BM25
sparse keyword index. Enterprise documents are full of exact terms
(policy names, product codes, acronyms) that pure dense embeddings can
miss, so blending both signals gives more reliable retrieval than
either alone.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

import config
from src.ingestion import Chunk
from src.embeddings import EmbeddingProvider


def _tokenize(text: str) -> List[str]:
    return text.lower().split()


class VectorStore:
    def __init__(self, embedder: EmbeddingProvider):
        self.embedder = embedder
        self.index: faiss.Index | None = None
        self.chunks: List[Chunk] = []
        self.bm25: BM25Okapi | None = None

    # ------------------------------------------------------------------
    def build(self, chunks: List[Chunk]):
        if not chunks:
            raise ValueError("No chunks to index. Upload documents first.")

        self.chunks = chunks
        texts = [c.text for c in chunks]

        # Fit embedder (no-op for sentence-transformers, fits vocab for TF-IDF)
        self.embedder.fit(texts)
        vecs = self.embedder.encode(texts)

        dim = vecs.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # cosine sim via normalized dot product
        self.index.add(vecs)

        self.bm25 = BM25Okapi([_tokenize(t) for t in texts])

    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = config.TOP_K) -> List[Tuple[Chunk, float]]:
        if self.index is None or not self.chunks:
            return []

        # Dense scores
        q_vec = self.embedder.encode([query])
        dense_scores, dense_idx = self.index.search(q_vec, len(self.chunks))
        dense_scores, dense_idx = dense_scores[0], dense_idx[0]

        dense_map = {}
        for score, idx in zip(dense_scores, dense_idx):
            if idx == -1:
                continue
            dense_map[idx] = float(score)

        # Sparse (BM25) scores, normalized to 0..1
        bm25_scores = self.bm25.get_scores(_tokenize(query))
        max_bm25 = max(bm25_scores) if len(bm25_scores) and max(bm25_scores) > 0 else 1.0
        bm25_map = {i: s / max_bm25 for i, s in enumerate(bm25_scores)}

        # Normalize dense scores (cosine sim already ~0..1, but clip for safety)
        combined = []
        for idx in range(len(self.chunks)):
            d = dense_map.get(idx, 0.0)
            b = bm25_map.get(idx, 0.0)
            score = config.DENSE_WEIGHT * d + config.BM25_WEIGHT * b
            combined.append((idx, score))

        combined.sort(key=lambda x: x[1], reverse=True)
        top = combined[:top_k]
        return [(self.chunks[idx], score) for idx, score in top]

    # ------------------------------------------------------------------
    def save(self):
        if self.index is not None:
            faiss.write_index(self.index, str(config.INDEX_PATH))
        with open(config.METADATA_PATH, "wb") as f:
            pickle.dump(self.chunks, f)
        with open(config.BM25_PATH, "wb") as f:
            pickle.dump(self.bm25, f)
        self.embedder.save(config.STORAGE_DIR / "embedder_state.pkl")

    def load(self) -> bool:
        if not (config.INDEX_PATH.exists() and config.METADATA_PATH.exists()):
            return False
        self.index = faiss.read_index(str(config.INDEX_PATH))
        with open(config.METADATA_PATH, "rb") as f:
            self.chunks = pickle.load(f)
        with open(config.BM25_PATH, "rb") as f:
            self.bm25 = pickle.load(f)
        embedder_state = config.STORAGE_DIR / "embedder_state.pkl"
        if embedder_state.exists():
            self.embedder.load_state(embedder_state)
        return True

    @property
    def stats(self) -> dict:
        sources = sorted({c.source for c in self.chunks})
        return {
            "num_chunks": len(self.chunks),
            "num_documents": len(sources),
            "documents": sources,
            "embedding_backend": self.embedder.backend,
        }
