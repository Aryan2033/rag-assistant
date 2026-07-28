"""

Embeds every chunk with multilingual-e5-base and upserts into Qdrant.
Note: e5 models require a "passage: " prefix on documents (and "query: " on queries).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import CHUNKS_JSONL, QDRANT_URL, COLLECTION, EMBED_MODEL, EMBED_DIM

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer


def main() -> None:
    if not CHUNKS_JSONL.exists():
        print(f"Missing {CHUNKS_JSONL}. Run chunk.py first.")
        return

    chunks = [json.loads(l) for l in CHUNKS_JSONL.read_text(encoding="utf-8").splitlines()]
    print(f"Loaded {len(chunks)} chunks.")

    print(f"Loading embedding model ({EMBED_MODEL}) — first run downloads ~1.1 GB ...")
    model = SentenceTransformer(EMBED_MODEL)

    passages = [f"passage: {c['text']}" for c in chunks]
    print("Embedding ...")
    vectors = model.encode(
        passages, batch_size=32, normalize_embeddings=True, show_progress_bar=True
    )

    client = QdrantClient(url=QDRANT_URL)
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)  # rebuild cleanly each run
    client.create_collection(
        COLLECTION,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )

    points = [
        PointStruct(
            id=c["chunk_uid"],
            vector=vectors[i].tolist(),
            payload={
                "text": c["text"],
                "doc_id": c["doc_id"],
                "source": c["source"],
                "chunk_index": c["chunk_index"],
            },
        )
        for i, c in enumerate(chunks)
    ]
    client.upsert(COLLECTION, points=points)
    print(f"\nUpserted {len(points)} vectors into '{COLLECTION}'.")
    print(f"Check the dashboard: {QDRANT_URL}/dashboard")


if __name__ == "__main__":
    main()