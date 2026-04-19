"""Streamlit UI for the Swiggy RAG chatbot."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from rag import get_answer, _get_vectorstore, _get_embeddings
from ingest import build_index, INDEX_DIR

st.set_page_config(page_title="Swiggy RAG Chatbot", page_icon="🍔", layout="wide")

st.title("🍔 Swiggy RAG Chatbot")
st.caption(
    "Retrieval-Augmented chatbot — grounded in Swiggy FAQs, a sample menu, and a refund policy."
)

# ---- Sidebar ----------------------------------------------------------------
with st.sidebar:
    st.header("About")
    st.markdown(
        "- **LLM:** Groq (Llama 3.3 70B)\n"
        "- **Embeddings:** MiniLM (local)\n"
        "- **Vector store:** FAISS (local)\n"
        "- **Docs:** `data/` folder"
    )

    st.divider()
    st.subheader("Try asking")
    st.markdown(
        "- How do I cancel my order?\n"
        "- How much is Swiggy One?\n"
        "- What's the price of Butter Chicken?\n"
        "- When am I not eligible for a refund?\n"
        "- What payment methods are supported?"
    )

    st.divider()
    if st.button("🔄 Rebuild index", use_container_width=True):
        with st.spinner("Rebuilding FAISS index from data/ ..."):
            n = build_index()
            # Invalidate in-memory caches so the freshly-built index is loaded
            _get_vectorstore.cache_clear()
            _get_embeddings.cache_clear()
        st.success(f"Index rebuilt with {n} chunks. Ask your question again.")

    index_ready = Path(INDEX_DIR).exists()
    st.write("Index status:", "✅ ready" if index_ready else "⏳ building on first run...")

# Auto-build the FAISS index on first startup (useful for cloud deploys
# where the vectorstore/ folder is gitignored and not shipped with the repo).
if not Path(INDEX_DIR).exists():
    with st.spinner("First-time setup: building vector index from data/ ..."):
        build_index()

# ---- Chat state -------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {role, content}
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

# Render existing chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---- Input ------------------------------------------------------------------
prompt = st.chat_input("Ask about Swiggy orders, menu, or refunds...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                history = [
                    (st.session_state.messages[i]["content"], st.session_state.messages[i + 1]["content"])
                    for i in range(0, len(st.session_state.messages) - 1, 2)
                    if st.session_state.messages[i]["role"] == "user"
                    and i + 1 < len(st.session_state.messages)
                    and st.session_state.messages[i + 1]["role"] == "assistant"
                ]
                result = get_answer(prompt, history=history)
                answer = result["answer"]
                st.session_state.last_sources = result["sources"]
            except Exception as e:
                answer = f"⚠️ Error: {e}"
                st.session_state.last_sources = []
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

# ---- Sources panel ----------------------------------------------------------
if st.session_state.last_sources:
    with st.expander("📚 Sources used for the last answer", expanded=False):
        for s in st.session_state.last_sources:
            st.markdown(f"**{s['file']}**")
            st.caption(s["preview"] + "...")
