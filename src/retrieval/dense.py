"""
src/retrieval/dense.py — Session 2
Dense semantic search over Qdrant. Run directly to test retrieval.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import QDRANT_URL, COLLECTION, EMBED_MODEL

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def search(query: str, top_k: int = 5) -> list[dict]:
    model = _get_model()
    vec = model.encode(f"query: {query}", normalize_embeddings=True).tolist()
    client = QdrantClient(url=QDRANT_URL)
    hits = client.query_points(COLLECTION, query=vec, limit=top_k).points
    return [
        {"chunk_uid": h.id, "score": h.score,
         "source": h.payload["source"], "text": h.payload["text"]}
        for h in hits
    ]


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Who teaches Natural Language Processing?"
    print(f"Query: {q}\n")
    for i, r in enumerate(search(q), 1):
        print(f"[{i}] score={r['score']:.3f}  ({r['source']})")
        print(r["text"][:300], "...\n")