"""
ui/app.py — Session B1
Streamlit front-end for the multi-document RAG assistant.
Shows the answer, which DOCUMENT it came from, whether the KG or RAG answered,
and expandable source citations.
"""
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/query"

st.set_page_config(page_title="Aalen Student Assistant", page_icon="🎓", layout="centered")


def friendly_source(raw: str) -> tuple[str, str]:
    """Map a raw filename to a clean display name and an icon."""
    s = raw.lower()
    if "examination" in s or "studies and" in s or "regulation" in s:
        return ("Exam Regulations (SPO)", "📕")
    if "handbook" in s:
        return ("Module Handbook", "📗")
    return (raw, "📄")


st.title("🎓 Aalen MLD Student Assistant")
st.caption(
    "Grounded answers over the HS Aalen module handbook **and** the exam regulations (SPO). "
    "Every answer is drawn only from these documents — the assistant says so when it doesn't know."
)

# Example questions — deliberately spanning BOTH documents to show multi-source
st.markdown("**Try asking:**")
examples = [
    "Who teaches Natural Language Processing?",          # handbook / KG
    "How many times can I retake a failed exam?",        # SPO
    "How many credits is the Projekt module?",           # handbook / KG
    "How long do I have to write the Master's thesis?",  # SPO
]
cols = st.columns(2)
for i, ex in enumerate(examples):
    if cols[i % 2].button(ex, use_container_width=True):
        st.session_state["question"] = ex

question = st.text_input(
    "Your question",
    value=st.session_state.get("question", ""),
    placeholder="e.g. What happens if I miss an exam without withdrawing?",
)

if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Searching the documents..."):
        try:
            resp = requests.post(API_URL, json={"question": question}, timeout=120)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach the API. Is the backend running on port 8000?\n\n{e}")
            st.stop()

    # --- Answer ---
    st.markdown("### Answer")
    st.write(data["answer"])

    # --- Provenance row: which document + which method ---
    sources = data.get("sources", [])
    method = data.get("method", "rag")

    # Determine the primary source document from the top source
    if sources:
        name, icon = friendly_source(sources[0]["source"])
        doc_label = f"{icon} {name}"
    else:
        doc_label = "—"

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**Source document:** {doc_label}")
    with col_b:
        if method == "knowledge_graph":
            st.markdown("**How:** ✅ Knowledge graph (exact fact)")
        else:
            st.markdown("**How:** 🔎 Retrieval (RAG)")

    # --- Expandable detailed sources ---
    if sources:
        with st.expander(f"View sources ({len(sources)})"):
            for i, s in enumerate(sources, 1):
                name, icon = friendly_source(s["source"])
                st.markdown(f"**[{i}] {icon} {name}**  ·  relevance {s['score']}")
                st.caption(s["snippet"])

st.divider()
st.caption(
    "Corpus: Module Handbook + Exam Regulations (SPO)  ·  "
    "Hybrid retrieval (BM25 + dense) · cross-encoder reranking · knowledge-graph grounding"
)