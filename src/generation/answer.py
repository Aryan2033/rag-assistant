"""
src/generation/answer.py — Session 3
Full RAG loop: retrieve chunks, build a grounded prompt, ask the LLM, return an answer.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.retrieval.rerank import rerank_search as search
from src.generation.llm import generate
from src.kg.kg_lookup import kg_lookup

SYSTEM = (
    "You answer questions for Aalen University students using ONLY the provided context.\n"
    "- If the context directly answers the question, give a concise answer and cite the "
    "source number like [1].\n"
    "- If the context does not clearly answer the specific question asked, reply exactly: "
    "\"I don't have that information in the available documents.\"\n"
    "- Do not use outside knowledge or guess."
)


def build_context(chunks: list[dict]) -> str:
    blocks = []
    for i, c in enumerate(chunks, 1):
        blocks.append(f"[{i}] (source: {c['source']})\n{c['text']}")
    return "\n\n".join(blocks)


def answer(query: str, top_k: int = 5) -> tuple[str, list[dict]]:
    # --- Stage 1: try the knowledge graph (exact, structured facts) ---
    kg_hit = kg_lookup(query)
    if kg_hit:
        reply = kg_hit["answer"]
        source = {
            "chunk_uid": "kg",
            "score": 1.0,
            "source": "Knowledge graph (module handbook)",
            "text": f"{kg_hit['entity']} → {kg_hit['attribute']} → {kg_hit['answer']}",
        }
        return reply, [source]

    # --- Stage 2: fall back to full RAG (retrieval + rerank + LLM) ---
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