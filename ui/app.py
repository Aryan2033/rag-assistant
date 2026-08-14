"""
ui/app.py — Session 23
Streamlit front-end. Calls the FastAPI backend over HTTP and renders
the answer, which path (KG vs RAG) answered, and the source citations.
"""
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/query"

st.set_page_config(page_title="Aalen MLD Study Assistant", page_icon="🎓", layout="centered")

st.title("🎓 MLD Module Handbook Assistant")
st.caption(
    "Grounded question-answering over the HS Aalen Master's module handbook. "
    "Answers come only from the document — the assistant says so when it doesn't know."
)

# A few example questions users can click
st.markdown("**Try asking:**")
examples = [
    "Who teaches Natural Language Processing?",
    "How many credits is the Projekt module?",
    "What is the exam format for Big Data & Data Mining?",
    "What topics does Machine Learning and Deep Learning cover?",
]
cols = st.columns(2)
for i, ex in enumerate(examples):
    if cols[i % 2].button(ex, use_container_width=True):
        st.session_state["question"] = ex

question = st.text_input(
    "Your question",
    value=st.session_state.get("question", ""),
    placeholder="e.g. Who is the module manager for Data Analytics?",
)

if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Thinking..."):
        try:
            resp = requests.post(API_URL, json={"question": question}, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach the API. Is the backend running on port 8000?\n\n{e}")
            st.stop()

    # Answer
    st.markdown("### Answer")
    st.write(data["answer"])

    # Which path answered — makes the KG work visible
    if data["method"] == "knowledge_graph":
        st.success("✅ Answered from the **knowledge graph** (exact structured fact)")
    else:
        st.info("🔎 Answered via **retrieval** (RAG over document chunks)")

    # Sources
    with st.expander(f"Sources ({len(data['sources'])})"):
        for i, s in enumerate(data["sources"], 1):
            st.markdown(f"**[{i}]** {s['source']}  ·  score {s['score']}")
            st.caption(s["snippet"])

st.divider()
st.caption("Built with hybrid retrieval (BM25 + dense) · cross-encoder reranking · knowledge-graph grounding")