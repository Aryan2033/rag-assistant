"""
app.py — Hugging Face Spaces entry point.
Self-contained: builds the index in-memory at startup, then serves the Streamlit UI.
Uses Gemini for generation (GEMINI_API_KEY set as a Space secret).
No Docker, no separate API server — one process.
"""
import os
os.environ["QDRANT_IN_MEMORY"] = "1"   # force in-memory Qdrant for the cloud

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import streamlit as st

from src.config import CHUNKS_JSONL, KG_JSON, EMBED_MODEL, EMBED_DIM, COLLECTION


# ---------- one-time startup: build the in-memory index ----------
@st.cache_resource(show_spinner="Loading models and building the index...")
def bootstrap():
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL)
    chunks = [json.loads(l) for l in CHUNKS_JSONL.read_text(encoding="utf-8").splitlines()]

    client = QdrantClient(":memory:")
    client.create_collection(
        COLLECTION, vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE)
    )
    vecs = model.encode(
        [f"passage: {c['text']}" for c in chunks],
        batch_size=32, normalize_embeddings=True, show_progress_bar=False,
    )
    client.upsert(COLLECTION, points=[
        PointStruct(id=c["chunk_uid"], vector=vecs[i].tolist(), payload={
            "text": c["text"], "doc_id": c["doc_id"],
            "source": c["source"], "chunk_index": c["chunk_index"],
        }) for i, c in enumerate(chunks)
    ])
    return client


# Build once, share the client with the retrieval layer
_client = bootstrap()

# Inject the pre-built in-memory client into the dense retriever
import src.retrieval.dense as dense
dense._shared_client = _client

from src.generation.answer import answer


def friendly_source(raw: str):
    s = raw.lower()
    if "examination" in s or "studies and" in s or "regulation" in s:
        return ("Exam Regulations (SPO)", "📕")
    if "handbook" in s:
        return ("Module Handbook", "📗")
    return (raw, "📄")


# ---------- UI ----------
st.set_page_config(page_title="Aalen Student Assistant", page_icon="🎓", layout="centered")
st.title("🎓 Aalen MLD Student Assistant")
st.caption(
    "Grounded answers over the HS Aalen module handbook **and** exam regulations (SPO). "
    "Answers come only from these documents — it says so when it doesn't know."
)

examples = [
    "Who teaches Natural Language Processing?",
    "How many times can I retake a failed exam?",
    "How many credits is the Projekt module?",
    "How long do I have to write the Master's thesis?",
]
cols = st.columns(2)
for i, ex in enumerate(examples):
    if cols[i % 2].button(ex, use_container_width=True):
        st.session_state["question"] = ex

question = st.text_input("Your question", value=st.session_state.get("question", ""),
                         placeholder="e.g. What happens if I miss an exam without withdrawing?")

if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Searching the documents..."):
        reply, sources = answer(question)

    st.markdown("### Answer")
    st.write(reply)

    method = "knowledge_graph" if sources and sources[0].get("chunk_uid") == "kg" else "rag"
    if sources:
        name, icon = friendly_source(sources[0]["source"])
        doc_label = f"{icon} {name}"
    else:
        doc_label = "—"

    c1, c2 = st.columns(2)
    c1.markdown(f"**Source document:** {doc_label}")
    c2.markdown("**How:** ✅ Knowledge graph" if method == "knowledge_graph"
                else "**How:** 🔎 Retrieval (RAG)")

    if sources:
        with st.expander(f"View sources ({len(sources)})"):
            for i, s in enumerate(sources, 1):
                name, icon = friendly_source(s["source"])
                st.markdown(f"**[{i}] {icon} {name}**  ·  relevance {round(float(s['score']), 3)}")
                st.caption(s["text"][:200])

st.divider()
st.caption("Module Handbook + Exam Regulations (SPO) · hybrid retrieval · reranking · knowledge-graph grounding")