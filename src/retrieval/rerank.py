"""
src/retrieval/rerank.py — Session 8
Cross-encoder reranking. Takes the fused candidate pool and re-scores each chunk
by true query-passage relevance, then returns the best top_k.

Two-stage retrieval: fast retrievers build a candidate pool (recall),
the cross-encoder re-ranks it precisely (precision).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import RERANK_MODEL
from src.retrieval.hybrid import hybrid_search

from sentence_transformers import CrossEncoder

_reranker: CrossEncoder | None = None


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


def rerank_search(query: str, top_k: int = 5, pool: int = 10) -> list[dict]:
    # Stage 1: fast fusion builds a candidate pool (get more than we return)
    candidates = hybrid_search(query, top_k=pool, pool=pool * 2)
    if not candidates:
        return []

    # Stage 2: cross-encoder scores each (query, chunk) pair together
    reranker = _get_reranker()
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)

    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)

    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return candidates[:top_k]


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "How many credits is Data Analytics?"
    print(f"Reranked query: {q}\n")
    for i, r in enumerate(rerank_search(q), 1):
        print(f"[{i}] rerank={r['rerank_score']:.3f}  ({r['source']})")
        print(r["text"][:250], "...\n")