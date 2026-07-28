"""
src/ingest/chunk.py — Session 2
Reads data/processed/documents.jsonl and writes data/processed/chunks.jsonl,
splitting each document into overlapping word-windows.
"""
from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # make 'src' importable
from src.config import DOCUMENTS_JSONL, CHUNKS_JSONL, WORDS_PER_CHUNK, CHUNK_OVERLAP

# Stable namespace so re-running produces the same chunk IDs (no duplicates)
NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  
# any fixed UUID


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks, step, start = [], size - overlap, 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
        start += step
    return chunks


def main() -> None:
    if not DOCUMENTS_JSONL.exists():
        print(f"Missing {DOCUMENTS_JSONL}. Run parse.py first.")
        return

    docs = [json.loads(line) for line in DOCUMENTS_JSONL.read_text(encoding="utf-8").splitlines()]
    out = []
    for doc in docs:
        pieces = chunk_text(doc["text"], WORDS_PER_CHUNK, CHUNK_OVERLAP)
        for i, piece in enumerate(pieces):
            out.append({
                "chunk_uid": str(uuid.uuid5(NAMESPACE, f"{doc['id']}:{i}")),
                "doc_id": doc["id"],
                "source": doc["source"],
                "chunk_index": i,
                "text": piece,
            })
        print(f"{doc['id']}: {len(pieces)} chunks")

    with CHUNKS_JSONL.open("w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(out)} chunks -> {CHUNKS_JSONL}")


if __name__ == "__main__":
    main()