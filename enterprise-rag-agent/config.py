"""
Central configuration for the Enterprise RAG & Agentic AI Knowledge Assistant.
All tunables live here so the rest of the codebase never hardcodes values.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Paths -------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_DOCS_DIR = DATA_DIR / "sample_docs"
STORAGE_DIR = BASE_DIR / "storage"
INDEX_PATH = STORAGE_DIR / "faiss.index"
METADATA_PATH = STORAGE_DIR / "metadata.pkl"
BM25_PATH = STORAGE_DIR / "bm25.pkl"

STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# --- LLM -----------------------------------------------------------------
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
# Default generation model. Override with CLAUDE_MODEL env var if needed.
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1500"))
MAX_AGENT_TOOL_ITERATIONS = 6  # safety cap on the agentic tool-use loop

# --- Embeddings ------------------------------------------------------------
# Preferred: local sentence-transformers model (downloaded on first run,
# needs internet the first time). Falls back automatically to a TF-IDF
# vectorizer (pure scikit-learn, zero downloads) if the model can't load,
# so the app always works even fully offline.
SENTENCE_TRANSFORMER_MODEL = os.getenv(
    "SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2"
)

# --- Chunking ---------------------------------------------------------------
CHUNK_SIZE = 900        # characters per chunk
CHUNK_OVERLAP = 150     # characters of overlap between consecutive chunks

# --- Retrieval ---------------------------------------------------------------
TOP_K = 5                 # chunks returned per retrieval call
DENSE_WEIGHT = 0.6        # weight given to dense (embedding) similarity
BM25_WEIGHT = 0.4         # weight given to sparse (keyword/BM25) similarity

# --- Misc ---------------------------------------------------------------
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
