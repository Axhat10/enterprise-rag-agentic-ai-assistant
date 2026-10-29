"""
Tests covering everything that doesn't require a live Anthropic API call:
ingestion, chunking, embeddings (TF-IDF fallback), hybrid retrieval, and
the calculator tool. Run with: pytest tests/ -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.ingestion import chunk_text, ingest_files
from src.embeddings import EmbeddingProvider
from src.vectorstore import VectorStore
from src.tools import calculator


def test_chunk_text_basic():
    text = "Sentence one. Sentence two. " * 100
    chunks = chunk_text(text, source="test.txt", chunk_size=200, overlap=50)
    assert len(chunks) > 1
    assert all(c.source == "test.txt" for c in chunks)
    assert all(len(c.text) > 0 for c in chunks)


def test_chunk_text_empty():
    assert chunk_text("", source="empty.txt") == []


def test_ingest_sample_docs():
    paths = list(config.SAMPLE_DOCS_DIR.glob("*.txt"))
    assert len(paths) >= 3
    chunks = ingest_files(paths)
    assert len(chunks) > 0
    sources = {c.source for c in chunks}
    assert "hr_leave_policy.txt" in sources


def test_embedding_provider_encodes():
    embedder = EmbeddingProvider()
    texts = ["annual leave policy", "VPN remote access", "pricing tiers"]
    embedder.fit(texts)
    vecs = embedder.encode(texts)
    assert vecs.shape[0] == 3
    assert vecs.shape[1] > 0


def test_vectorstore_retrieves_relevant_chunk():
    paths = list(config.SAMPLE_DOCS_DIR.glob("*.txt"))
    chunks = ingest_files(paths)
    embedder = EmbeddingProvider()
    store = VectorStore(embedder)
    store.build(chunks)

    results = store.search("how many days of annual leave do employees get", top_k=3)
    assert len(results) > 0
    top_chunk, score = results[0]
    assert "hr_leave_policy" in top_chunk.source
    assert score > 0


def test_vectorstore_save_and_load(tmp_path, monkeypatch):
    # Redirect storage paths to a temp dir so this test doesn't clobber real state
    monkeypatch.setattr(config, "INDEX_PATH", tmp_path / "faiss.index")
    monkeypatch.setattr(config, "METADATA_PATH", tmp_path / "metadata.pkl")
    monkeypatch.setattr(config, "BM25_PATH", tmp_path / "bm25.pkl")
    monkeypatch.setattr(config, "STORAGE_DIR", tmp_path)

    paths = list(config.SAMPLE_DOCS_DIR.glob("*.txt"))
    chunks = ingest_files(paths)
    embedder = EmbeddingProvider()
    store = VectorStore(embedder)
    store.build(chunks)
    store.save()

    new_store = VectorStore(EmbeddingProvider())
    assert new_store.load() is True
    assert new_store.stats["num_chunks"] == store.stats["num_chunks"]

    results = new_store.search("SSO single sign on", top_k=3)
    assert len(results) > 0


def test_calculator_basic():
    assert "12" in calculator("5 + 7")
    assert "Error" in calculator("import os")  # unsafe input rejected
    assert "Error" in calculator("__import__('os')")


def test_calculator_complex_expression():
    result = calculator("(1250 * 0.18) / 12")
    assert "18.75" in result
