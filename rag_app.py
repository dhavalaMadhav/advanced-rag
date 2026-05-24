"""
rag_app.py
----------
FastAPI application — Observable AI Inference Platform

This file is the EXTENDED version of the original RAG app.
All original RAG retrieval logic is preserved. New additions:

  1. Query model extended with optional `session_id` field
     (fully backward-compatible — existing frontend still works).
  2. llm.invoke(prompt) replaced with generate_response() wrapper
     which captures telemetry and logs to JSONL + MongoDB.
  3. /ask/stream  — SSE streaming endpoint.
  4. Analytics endpoints:
       GET /analytics/summary
       GET /analytics/latency
       GET /analytics/tokens
       GET /analytics/sessions
  5. CORS configured for local development.
"""

import os
import logging
from typing import Optional

from dotenv import load_dotenv

# -------------------------------------------------------------------------
# Logging configuration (must be set before other imports write logs)
# -------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

print("[START] Starting Observable RAG Platform...")

# -------------------------------------------------------------------------
# Load API Keys
# -------------------------------------------------------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN     = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# -------------------------------------------------------------------------
# FastAPI setup
# -------------------------------------------------------------------------
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI(
    title="Observable RAG Platform",
    description="RAG system with telemetry ingestion and inference analytics",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------------
# LangChain / VectorStore imports
# -------------------------------------------------------------------------
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# -------------------------------------------------------------------------
# SDK — wrapper replaces direct llm.invoke() calls
# -------------------------------------------------------------------------
from sdk.llm_wrapper import generate_response, stream_response

# -------------------------------------------------------------------------
# Database — MongoDB persistence (gracefully degrades to JSONL if unavailable)
# -------------------------------------------------------------------------
import db  # importing triggers connection + index setup at startup

# -------------------------------------------------------------------------
# LOCAL EMBEDDING  — runs on-device via sentence-transformers, no API needed
# The same model (intfloat/e5-small-v2, 384-dim) used to build the index.
# -------------------------------------------------------------------------
def get_embeddings():
    print("[INFO] Loading local embedding model (e5-small-v2)...")
    return HuggingFaceEmbeddings(
        model_name="intfloat/e5-small-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# -------------------------------------------------------------------------
# Load PDFs  (unchanged from original)
# -------------------------------------------------------------------------
def load_documents(folder="data"):
    docs = []
    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            path = os.path.join(folder, file)
            loader = PyMuPDFLoader(path)
            loaded_docs = loader.load()
            for d in loaded_docs:
                d.metadata["source"] = file
            docs.extend(loaded_docs)
    return docs


# -------------------------------------------------------------------------
# Split Documents  (unchanged from original)
# -------------------------------------------------------------------------
def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    return splitter.split_documents(docs)


# -------------------------------------------------------------------------
# Create / Load ChromaDB  (unchanged from original)
# -------------------------------------------------------------------------
def get_vectorstore():
    persist_dir = os.path.join(os.getcwd(), "chroma_db")
    embeddings  = get_embeddings()

    # =====================================================================
    # INDEXING MODE (uncomment locally to re-index PDFs)
    # =====================================================================
    # docs   = load_documents("data")
    # chunks = split_documents(docs)
    # db_vec = Chroma.from_documents(
    #     documents=chunks,
    #     embedding=embeddings,
    #     persist_directory=persist_dir
    # )
    # print("[INFO] DB CREATED / UPDATED")
    # return db_vec
    # =====================================================================

    # =====================================================================
    # PRODUCTION MODE — load existing persisted index
    # =====================================================================
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )


# -------------------------------------------------------------------------
# Load system at startup
# -------------------------------------------------------------------------
print("[INFO] Loading vector DB...")
_chroma = get_vectorstore()
retriever = _chroma.as_retriever(search_kwargs={"k": 3})

print("[INFO] Connecting to Groq...")
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",
    temperature=0.1,
)

print("[OK] Backend ready!")


# =========================================================================
# REQUEST MODELS
# =========================================================================

class Query(BaseModel):
    """
    Request body for /ask and /ask/stream.

    session_id is OPTIONAL and defaults to "default" so the existing
    frontend (which doesn't send session_id) continues to work without
    any changes.
    """
    question:   str
    session_id: Optional[str] = "default"


# =========================================================================
# RAG ENDPOINT  — extended with telemetry wrapper
# =========================================================================

@app.post("/ask")
def ask(query: Query):
    """
    Answer a question using RAG retrieval + Groq LLM.

    Telemetry is captured automatically via generate_response():
      - latency, tokens, session_id logged to JSONL + MongoDB
      - PII redacted before storage
    """
    try:
        # --- Retrieval (unchanged) ----------------------------------------
        docs = retriever.invoke(query.question)

        if not docs:
            return {"answer": "No relevant documents found", "sources": []}

        context = "\n\n".join([d.page_content for d in docs])

        prompt = f"""Answer ONLY using the context below.
If not found, say: I don't know.

Context:
{context}

Question:
{query.question}
"""

        # --- LLM call via wrapper (replaces llm.invoke) -------------------
        answer = generate_response(
            llm,
            prompt,
            session_id=query.session_id or "default",
            provider="groq",
            model="llama-3.1-8b-instant",
        )

        return {
            "answer": answer,
            "sources": [
                {
                    "file": d.metadata.get("source"),
                    "page": d.metadata.get("page"),
                }
                for d in docs
            ],
        }

    except Exception as e:
        return {"error": str(e)}


# =========================================================================
# STREAMING ENDPOINT  — Server-Sent Events
# =========================================================================

@app.post("/ask/stream")
def ask_stream(query: Query):
    """
    SSE streaming endpoint — yields answer tokens as they are generated.

    Frontend can consume this with EventSource or fetch + ReadableStream.
    Each chunk is formatted as:  data: <token>\\n\\n
    A final "data: [DONE]\\n\\n" signals end-of-stream.
    """
    docs = retriever.invoke(query.question)

    if not docs:
        def empty_stream():
            yield "data: No relevant documents found\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(empty_stream(), media_type="text/event-stream")

    context = "\n\n".join([d.page_content for d in docs])
    prompt  = f"""Answer ONLY using the context below.
If not found, say: I don't know.

Context:
{context}

Question:
{query.question}
"""

    def event_generator():
        try:
            for chunk in stream_response(
                llm,
                prompt,
                session_id=query.session_id or "default",
                provider="groq",
                model="llama-3.1-8b-instant",
            ):
                yield f"data: {chunk}\n\n"
        except Exception as e:
            yield f"data: [ERROR] {e}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# =========================================================================
# ANALYTICS ENDPOINTS
# =========================================================================

@app.get("/analytics/summary")
def analytics_summary():
    """
    Returns total requests, failures, successes, and success rate.
    Falls back to JSONL scan if MongoDB is unavailable.
    """
    mongo_stats = db.get_analytics_summary()
    if mongo_stats:
        return mongo_stats

    # JSONL fallback
    from ingestion.logger import read_logs
    logs   = read_logs(limit=10_000)
    total  = len(logs)
    failed = sum(1 for r in logs if r.get("status") == "error")
    return {
        "total_requests":  total,
        "total_failures":  failed,
        "total_successes": total - failed,
        "success_rate":    round((total - failed) / total * 100, 2) if total else 0,
        "mongodb":         False,
        "source":          "jsonl",
    }


@app.get("/analytics/latency")
def analytics_latency():
    """
    Returns avg / min / max / p95 latency across all successful requests.
    Falls back to JSONL scan if MongoDB is unavailable.
    """
    mongo_stats = db.get_latency_stats()
    if mongo_stats:
        return mongo_stats

    # JSONL fallback
    from ingestion.logger import read_logs
    logs     = read_logs(limit=10_000)
    latencies = [
        r["latency_ms"]
        for r in logs
        if r.get("status") == "success" and r.get("latency_ms") is not None
    ]
    if not latencies:
        return {}
    latencies.sort()
    p95_idx = int(len(latencies) * 0.95)
    return {
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
        "min_latency_ms": round(min(latencies), 2),
        "max_latency_ms": round(max(latencies), 2),
        "p95_latency_ms": round(latencies[p95_idx], 2),
        "source": "jsonl",
    }


@app.get("/analytics/tokens")
def analytics_tokens():
    """
    Returns total and average token usage for all successful requests.
    Falls back to JSONL scan if MongoDB is unavailable.
    """
    mongo_stats = db.get_token_stats()
    if mongo_stats:
        return mongo_stats

    # JSONL fallback
    from ingestion.logger import read_logs
    logs = read_logs(limit=10_000)
    rows = [r for r in logs if r.get("total_tokens") is not None]
    if not rows:
        return {}
    total_p = sum(r.get("prompt_tokens") or 0 for r in rows)
    total_c = sum(r.get("completion_tokens") or 0 for r in rows)
    total_t = sum(r.get("total_tokens") or 0 for r in rows)
    n       = len(rows)
    return {
        "total_prompt_tokens":     total_p,
        "total_completion_tokens": total_c,
        "total_tokens":            total_t,
        "avg_prompt_tokens":       round(total_p / n, 1),
        "avg_completion_tokens":   round(total_c / n, 1),
        "source": "jsonl",
    }


@app.get("/analytics/sessions")
def analytics_sessions():
    """
    Returns per-session request counts sorted by most active first.
    Falls back to JSONL scan if MongoDB is unavailable.
    """
    mongo_stats = db.get_session_stats()
    if mongo_stats:
        return mongo_stats

    # JSONL fallback
    from ingestion.logger import read_logs
    from collections import Counter
    logs = read_logs(limit=10_000)
    counts = Counter(r.get("session_id", "unknown") for r in logs)
    return [
        {"session_id": sid, "request_count": cnt}
        for sid, cnt in counts.most_common(50)
    ]


# =========================================================================
# HEALTH CHECK  (unchanged from original)
# =========================================================================

@app.get("/")
def home():
    return {
        "message": "Observable RAG Platform is running",
        "version": "2.0.0",
        "features": [
            "RAG retrieval",
            "Telemetry logging (JSONL + MongoDB)",
            "PII redaction",
            "Session tracking",
            "Streaming (SSE)",
            "Analytics endpoints",
        ],
    }