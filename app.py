"""
app.py — Hugging Face Spaces entry point (Gradio).
Self-contained: builds the index in-memory at startup, serves a Gradio UI.
Uses Gemini for generation (GEMINI_API_KEY set as a Space secret). No Docker, one process.
"""
import os
os.environ["QDRANT_IN_MEMORY"] = "1"
os.environ["GRADIO_WATCH_DIRS"] = ""

try:
    import spaces   # available on HF ZeroGPU; not installed locally
except ImportError:
    class _SpacesStub:
        def GPU(self, *args, **kwargs):
            # no-op decorator so @spaces.GPU works locally too
            def wrap(fn):
                return fn
            return wrap
    spaces = _SpacesStub() # must be imported before torch/CUDA packages (ZeroGPU requirement)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import gradio as gr

from src.config import CHUNKS_JSONL, EMBED_MODEL, EMBED_DIM, COLLECTION


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


print("Building index at startup...")
_client = bootstrap()
import src.retrieval.dense as dense
dense._shared_client = _client
from src.generation.answer import answer
print("Ready.")


def friendly_source(raw: str):
    s = raw.lower()
    if "examination" in s or "studies and" in s or "regulation" in s:
        return "📕 Exam Regulations (SPO)"
    if "handbook" in s:
        return "📗 Module Handbook"
    return f"📄 {raw}"


@spaces.GPU
def _gpu_warmup():
    """Satisfies ZeroGPU's requirement for a GPU-decorated function."""
    return True

def ask(question: str):
    if not question or not question.strip():
        return "Please enter a question.", "", ""
    reply, sources = answer(question)
    method = "knowledge_graph" if sources and sources[0].get("chunk_uid") == "kg" else "rag"
    doc = friendly_source(sources[0]["source"]) if sources else "—"
    how = "✅ Knowledge graph (exact fact)" if method == "knowledge_graph" else "🔎 Retrieval (RAG)"
    meta = f"**Source document:** {doc}  ·  **How:** {how}"
    src_md = ""
    for i, s in enumerate(sources, 1):
        src_md += f"**[{i}] {friendly_source(s['source'])}**  ·  relevance {round(float(s['score']), 3)}\n\n"
        src_md += f"> {s['text'][:220]}\n\n"
    return reply, meta, src_md


EXAMPLES = [
    "Who teaches Natural Language Processing?",
    "How many times can I retake a failed exam?",
    "How many credits is the Projekt module?",
    "How long do I have to write the Master's thesis?",
    "What happens if I miss an exam without withdrawing?",
    "Who do I contact about my student visa?",
]

CUSTOM_CSS = """
.gradio-container {max-width: 880px !important; margin: auto !important;}
#hero {text-align:center; padding: 18px 0 6px 0;}
#hero h1 {font-size: 2rem; margin-bottom: 4px;}
#hero p {color: var(--body-text-color-subdued); font-size: 0.98rem; margin: 0;}
#answer-box {
    background: var(--block-background-fill);
    color: var(--body-text-color);
    border: 1px solid var(--border-color-primary);
    border-radius: 12px;
    padding: 18px 20px;
    min-height: 60px;
    font-size: 1.05rem;
}
#meta-box {font-size: 0.9rem; color: var(--body-text-color-subdued); padding-top: 6px;}
.footer-note {text-align:center; color: var(--body-text-color-subdued); font-size:0.8rem; padding-top:10px;}
"""
HERO = """
<div id="hero">
  <h1>🎓 Aalen Student Assistant</h1>
  <p>Grounded Q&amp;A over the HS Aalen module handbook &amp; exam regulations (SPO).
     Every answer is drawn only from these documents — it says so when it doesn't know.</p>
</div>
"""
with gr.Blocks(title="Aalen Student Assistant") as demo:
    gr.HTML(HERO)

    with gr.Row():
        question = gr.Textbox(
            label="", scale=5, container=False,
            placeholder="Ask about modules, credits, professors, exams, thesis rules…",
        )
        ask_btn = gr.Button("Ask", variant="primary", scale=1, min_width=110)

    gr.Examples(examples=EXAMPLES, inputs=question, label="Try one of these")

    gr.Markdown("### Answer")
    answer_out = gr.Markdown(elem_id="answer-box")
    meta_out = gr.Markdown(elem_id="meta-box")

    with gr.Accordion("📄 View sources", open=False):
        sources_out = gr.Markdown()

    gr.HTML(
        '<div class="footer-note">Module Handbook + Exam Regulations (SPO) · '
        'hybrid retrieval (BM25 + dense) · cross-encoder reranking · knowledge-graph grounding</div>'
    )

    ask_btn.click(ask, inputs=question, outputs=[answer_out, meta_out, sources_out])
    question.submit(ask, inputs=question, outputs=[answer_out, meta_out, sources_out])

if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Soft(primary_hue="blue"),
        css=CUSTOM_CSS,
        server_name="0.0.0.0",
        server_port=7860,
    )