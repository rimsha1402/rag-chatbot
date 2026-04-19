    # Swiggy RAG Chatbot

A Retrieval-Augmented-Generation (RAG) chatbot that answers questions about Swiggy FAQs, a sample restaurant menu, and a refund policy. Built as an internship demo project.

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| UI | **Streamlit** | Single-file chat UI, one command to launch |
| LLM | **Groq (Llama 3.3 70B)** | Free API key, very fast inference |
| Embeddings | **MiniLM (sentence-transformers)** | Runs locally, free, ~90 MB |
| Vector store | **FAISS** | Local file, no server to manage |
| Orchestration | **LangChain** | Standard RAG building blocks |

## How it works (RAG in 5 steps)

1. **Load** markdown files from `data/`.
2. **Split** them into ~500-character chunks (with overlap so context isn't cut mid-sentence).
3. **Embed** each chunk into a vector using MiniLM.
4. **Store** vectors in a local FAISS index.
5. On each question → embed the question → retrieve top-4 most similar chunks → feed them as CONTEXT into a Groq/Llama prompt → return the grounded answer with source citations.

## Setup (4 commands)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add your free Groq API key

Get a free key at https://console.groq.com/keys (no credit card required), then:

```bash
cp .env.example .env
# open .env and paste your key after GROQ_API_KEY=
```

### 3. Build the vector index (one-time)

```bash
python ingest.py
```

First run downloads the embedding model (~90 MB). Subsequent runs are fast.

### 4. Launch the chatbot

```bash
streamlit run app.py
```

Open the browser URL shown in the terminal (usually http://localhost:8501).

## Demo questions

Good prompts that show retrieval is working:

- "How do I cancel my order on Swiggy?"
- "What is the refund timeline for UPI payments?"
- "How much does Butter Chicken cost at Spice Route Kitchen?"
- "Am I eligible for a refund if I just didn't like the food?"
- "What perks do Swiggy One members get?"

## Project structure

```
rag-chatbot/
├── app.py             # Streamlit chat UI
├── rag.py             # Retrieval + Groq LLM call
├── ingest.py          # Builds the FAISS index from data/
├── data/              # Source documents (markdown)
├── vectorstore/       # Auto-generated FAISS index (gitignored)
├── requirements.txt
├── .env.example
└── README.md
```

## What I can explain in an interview

- **Why RAG over fine-tuning:** cheaper, instantly updatable (just drop a new file in `data/` and re-run `ingest.py`), reduces hallucination because answers are grounded in retrieved context.
- **Key tuning knobs:** chunk size, chunk overlap, top-k retrieval, embedding model choice, LLM temperature.
- **Trade-offs:** MiniLM is fast & free but weaker than OpenAI/Cohere embeddings; Groq free tier is rate-limited per minute; FAISS is in-memory so doesn't scale past ~millions of vectors (you'd move to pgvector / Pinecone / Weaviate).
- **What I'd add next:** PDF upload UI, conversation memory across sessions, evaluation with a small Q&A test set, Docker packaging.

## Troubleshooting

- **`GROQ_API_KEY is not set`** → make sure `.env` exists and is in the project root.
- **`No index found`** → run `python ingest.py` first.
- **Slow first question** → the embedding model loads lazily on first query; it's instant after that.
