"""src/config.py — shared settings."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DOCUMENTS_JSONL = PROCESSED_DIR / "documents.jsonl"
CHUNKS_JSONL = PROCESSED_DIR / "chunks.jsonl"

# Qdrant (your Docker container)
QDRANT_URL = "http://localhost:6333"
COLLECTION = "student_docs"

# Embeddings
EMBED_MODEL = "intfloat/multilingual-e5-base"
EMBED_DIM = 768

# Chunking
WORDS_PER_CHUNK = 150   # was 350 — smaller so a chunk stays within one module block
CHUNK_OVERLAP = 30      # was 50

# LLM (local, via Ollama)
LLM_MODEL = "llama3.2"

# Reranker (cross-encoder)
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"  # multilingual, matches your e5 choice

# Evaluation
EVAL_DIR = PROJECT_ROOT / "data" / "eval"
GOLD_SET = EVAL_DIR / "gold_set.jsonl"