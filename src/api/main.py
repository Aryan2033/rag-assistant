"""
src/api/main.py — Session 21
FastAPI wrapper around the RAG pipeline. Exposes /query and /health.
The endpoint is a thin shell over answer() — all the real work lives in the pipeline.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.generation.answer import answer

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="Student RAG Assistant",
    description="Grounded question-answering over the HS Aalen MLD module handbook.",
    version="1.0.0",
)

# Allow a browser UI (Streamlit) to call this API during local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # fine for a local/demo project; tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- request / response schemas ---
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's question")
    top_k: int = Field(5, ge=1, le=20, description="How many chunks to retrieve for the RAG path")


class Source(BaseModel):
    source: str
    score: float
    snippet: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    method: str  # "knowledge_graph" or "rag"
    sources: list[Source]


# --- endpoints ---
@app.get("/health")
def health():
    """Simple liveness check."""
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    reply, chunks = answer(req.question, top_k=req.top_k)
    method = "knowledge_graph" if chunks and chunks[0].get("chunk_uid") == "kg" else "rag"
    sources = [
        Source(source=c["source"], score=round(float(c["score"]), 3), snippet=c["text"][:200])
        for c in chunks
    ]
    return QueryResponse(question=req.question, answer=reply, method=method, sources=sources)