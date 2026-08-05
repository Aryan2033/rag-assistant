"""
src/retrieval/bm25.py — Session 6
Keyword (lexical) retrieval with BM25 over the same chunks used for dense search.
Strong exactly where dense is weak: proper nouns, course codes, exact terms.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import CHUNKS_JSONL

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercase and split into word tokens (handles German umlauts via \\w + UNICODE)."""
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self) -> None:
        rows = [json.loads(l) for l in CHUNKS_JSONL.read_text(encoding="utf-8").splitlines()]
        self.chunks = rows
        self.bm25 = BM25Okapi([tokenize(r["text"]) for r in rows])

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for i in order:
            c = self.chunks[i]
            results.append({
                "chunk_uid": c["chunk_uid"],
                "score": float(scores[i]),
                "source": c["source"],
                "text": c["text"],
            })
        return results


_index: BM25Index | None = None


def get_index() -> BM25Index:
    """Build the index once per process (it's in-memory and fast to rebuild)."""
    global _index
    if _index is None:
        _index = BM25Index()
    return _index


def search(query: str, top_k: int = 5) -> list[dict]:
    return get_index().search(query, top_k)


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Winfried Bantel module manager"
    print(f"BM25 query: {q}\n")
    for i, r in enumerate(search(q), 1):
        print(f"[{i}] score={r['score']:.3f}  ({r['source']})")
        print(r["text"][:300], "...\n")