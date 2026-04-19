"""Ingest documents from data/ into a local FAISS vector store.

Run once (and whenever you change files in data/):
    python ingest.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
INDEX_DIR = ROOT / "vectorstore"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def build_index() -> int:
    """Build FAISS index from all markdown/txt files in data/. Returns chunk count."""
    if not DATA_DIR.exists() or not any(DATA_DIR.iterdir()):
        raise RuntimeError(f"No documents found in {DATA_DIR}. Add files and retry.")

    loader = DirectoryLoader(
        str(DATA_DIR),
        glob="**/*.*",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    docs = loader.load()
    print(f"Loaded {len(docs)} source file(s) from {DATA_DIR}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunk(s)")

    print(f"Loading embedding model: {EMBED_MODEL} (first run will download ~90 MB)")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    if INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(str(INDEX_DIR))
    print(f"Saved FAISS index to {INDEX_DIR}")
    return len(chunks)


if __name__ == "__main__":
    n = build_index()
    print(f"Done. Indexed {n} chunks.")
