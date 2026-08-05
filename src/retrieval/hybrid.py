"""
src/retrieval/hybrid.py — Session 7
Combine dense (semantic) and BM25 (keyword) retrieval with Reciprocal Rank Fusion.
RRF uses only rank position, so the two retrievers' incompatible score scales don't matter.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.retrieval.dense import search as dense_search
from src.retrieval.bm25 import search as bm25_search

RRF_K = 60  # standard constant; larger = flatter weighting across ranks


def _rrf_scores(results: list[dict]) -> dict[str, float]:
    """Map chunk_uid -> reciprocal-rank contribution from one retriever's ranked list."""
    return {r["chunk_uid"]: 1.0 / (RRF_K + rank) for rank, r in enumerate(results, start=1)}


def hybrid_search(query: str, top_k: int = 5, pool: int = 10) -> list[dict]:
    # Pull a deeper pool from each retriever than we finally return, so fusion has room to work
    dense_hits = dense_search(query, top_k=pool)
    bm25_hits = bm25_search(query, top_k=pool)

    # Keep the full chunk data around, keyed by uid
    by_uid: dict[str, dict] = {}
    for r in dense_hits + bm25_hits:
        by_uid.setdefault(r["chunk_uid"], r)

    dense_rrf = _rrf_scores(dense_hits)
    bm25_rrf = _rrf_scores(bm25_hits)

    fused = []
    for uid, chunk in by_uid.items():
        score = dense_rrf.get(uid, 0.0) + bm25_rrf.get(uid, 0.0)
        fused.append({
            "chunk_uid": uid,
            "score": score,
            "source": chunk["source"],
            "text": chunk["text"],
            "in_dense": uid in dense_rrf,
            "in_bm25": uid in bm25_rrf,
        })

    fused.sort(key=lambda x: x["score"], reverse=True)
    return fused[:top_k]


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Who is the module manager for Natural Language Processing?"
    print(f"Hybrid query: {q}\n")
    for i, r in enumerate(hybrid_search(q), 1):
        flags = []
        if r["in_dense"]: flags.append("dense")
        if r["in_bm25"]: flags.append("bm25")
        print(f"[{i}] rrf={r['score']:.4f}  found by: {'+'.join(flags)}  ({r['source']})")
        print(r["text"][:250], "...\n")