"""
src/generation/answer.py — Session 3
Full RAG loop: retrieve chunks, build a grounded prompt, ask the LLM, return an answer.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.retrieval.dense import search
from src.generation.llm import generate

SYSTEM = (
    "You are a helpful assistant for international students at Aalen University. "
    "Answer the question using ONLY the provided context. "
    "If the answer is not in the context, say you don't have that information — "
    "do not guess or use outside knowledge. "
    "Cite the source number in square brackets, e.g. [1], after facts you use. "
    "Keep the answer concise."
)


def build_context(chunks: list[dict]) -> str:
    blocks = []
    for i, c in enumerate(chunks, 1):
        blocks.append(f"[{i}] (source: {c['source']})\n{c['text']}")
    return "\n\n".join(blocks)


def answer(query: str, top_k: int = 5) -> tuple[str, list[dict]]:
    chunks = search(query, top_k=top_k)
    context = build_context(chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    reply = generate(prompt, system=SYSTEM)
    return reply, chunks


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "How many credits is the Data Analytics module?"
    print(f"Question: {q}\n")
    reply, chunks = answer(q)
    print("Answer:")
    print(reply)
    print("\n--- Retrieved sources ---")
    for i, c in enumerate(chunks, 1):
        print(f"[{i}] {c['source']} (score {c['score']:.3f})")