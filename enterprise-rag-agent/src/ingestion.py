"""
Document ingestion: load raw files (PDF / DOCX / TXT / MD) into plain text,
then split that text into overlapping chunks suitable for embedding.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from pypdf import PdfReader
import docx

import config


@dataclass
class Chunk:
    """A single retrievable unit of text plus its provenance."""
    id: str
    text: str
    source: str          # original filename
    chunk_index: int      # position within the source document
    field_meta: dict = field(default_factory=dict)


def load_document(path: Path) -> str:
    """Extract raw text from a supported file type."""
    ext = path.suffix.lower()
    if ext not in config.SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}")

    if ext == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if ext == ".docx":
        d = docx.Document(str(path))
        return "\n".join(p.text for p in d.paragraphs)

    # .txt / .md
    return path.read_text(encoding="utf-8", errors="ignore")


def clean_text(text: str) -> str:
    """Collapse excess whitespace while preserving paragraph breaks."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> List[Chunk]:
    """
    Split text into overlapping chunks. Tries to break on sentence/paragraph
    boundaries near the target size rather than mid-word, for cleaner context.
    """
    text = clean_text(text)
    if not text:
        return []

    chunks: List[Chunk] = []
    start = 0
    idx = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # Try to extend to the next sentence boundary (., !, ?, newline)
        # within a small lookahead window, so chunks don't cut mid-sentence.
        if end < text_len:
            lookahead = text[end:end + 200]
            match = re.search(r"[.!?\n]", lookahead)
            if match:
                end += match.end()

        chunk_str = text[start:end].strip()
        if chunk_str:
            chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    text=chunk_str,
                    source=source,
                    chunk_index=idx,
                )
            )
            idx += 1

        if end >= text_len:
            break
        start = end - overlap  # step forward, keeping overlap

    return chunks


def ingest_files(paths: List[Path]) -> List[Chunk]:
    """Load and chunk a batch of files into a flat list of Chunks."""
    all_chunks: List[Chunk] = []
    for path in paths:
        try:
            raw = load_document(path)
        except Exception as e:
            raise RuntimeError(f"Failed to read {path.name}: {e}") from e
        all_chunks.extend(chunk_text(raw, source=path.name))
    return all_chunks
