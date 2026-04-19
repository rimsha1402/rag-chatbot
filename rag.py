"""RAG logic: load the FAISS index and answer questions with Groq/Llama."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_groq import ChatGroq

load_dotenv()

ROOT = Path(__file__).parent
INDEX_DIR = ROOT / "vectorstore"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.3-70b-versatile"
TOP_K = 4

SYSTEM_PROMPT = """You are a helpful Swiggy customer-support assistant.
Answer the user's question using ONLY the information in the CONTEXT below.
If the context does not contain the answer, say: "I don't have that information in my knowledge base."
Be concise, friendly, and cite the relevant policy/section when useful.
Do not invent prices, timings, or policies that are not in the context.

CONTEXT:
{context}
"""


@lru_cache(maxsize=1)
def _get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


@lru_cache(maxsize=1)
def _get_vectorstore() -> FAISS:
    if not INDEX_DIR.exists():
        raise RuntimeError(
            f"No index found at {INDEX_DIR}. Run `python ingest.py` first."
        )
    return FAISS.load_local(
        str(INDEX_DIR),
        _get_embeddings(),
        allow_dangerous_deserialization=True,
    )


@lru_cache(maxsize=1)
def _get_llm() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and paste your key from https://console.groq.com/keys"
        )
    return ChatGroq(
        model=LLM_MODEL,
        groq_api_key=api_key,
        temperature=0.2,
    )


def _format_context(docs: List[Document]) -> str:
    blocks = []
    for i, d in enumerate(docs, 1):
        src = Path(d.metadata.get("source", "unknown")).name
        blocks.append(f"[Source {i}: {src}]\n{d.page_content}")
    return "\n\n".join(blocks)


def get_answer(question: str, history: List[Tuple[str, str]] | None = None) -> dict:
    """Retrieve relevant chunks and ask Groq. Returns {answer, sources}."""
    vs = _get_vectorstore()
    docs = vs.similarity_search(question, k=TOP_K)
    context = _format_context(docs)

    messages = [("system", SYSTEM_PROMPT.format(context=context))]
    for user_msg, ai_msg in (history or [])[-4:]:  # last 4 turns for context
        messages.append(("human", user_msg))
        messages.append(("ai", ai_msg))
    messages.append(("human", question))

    llm = _get_llm()
    resp = llm.invoke(messages)
    answer = resp.content if hasattr(resp, "content") else str(resp)

    sources = []
    seen = set()
    for d in docs:
        name = Path(d.metadata.get("source", "unknown")).name
        if name not in seen:
            seen.add(name)
            sources.append({"file": name, "preview": d.page_content[:200]})

    return {"answer": answer, "sources": sources}
