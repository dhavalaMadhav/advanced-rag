# Observable AI Inference Platform

> A production-grade **RAG (Retrieval-Augmented Generation)** system extended with full **telemetry ingestion**, **inference analytics**, **PII redaction**, **session tracking**, and **streaming** support.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENT / BROWSER                        │
│  frontend/index.html — chat UI + Analytics panel (session_id)  │
└───────────────────────────────┬─────────────────────────────────┘
                                │ HTTP / SSE
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI  (rag_app.py)                         │
│                                                                  │
│  POST /ask           — RAG retrieval + telemetry logging         │
│  POST /ask/stream    — SSE streaming endpoint                    │
│  GET  /analytics/*   — summary / latency / tokens / sessions     │
│  GET  /             — health check                               │
└──────────┬──────────────────────────────────────────────────────┘
           │                             │
           ▼                             ▼
┌──────────────────────┐    ┌────────────────────────────────────┐
│  ChromaDB            │    │  sdk/llm_wrapper.py                │
│  (vector store)      │    │  ─ latency measurement             │
│  Retrieves top-3 docs│    │  ─ token usage extraction          │
└──────────────────────┘    │  ─ PII redaction (sdk/pii_redact)  │
                            │  ─ routes to ingestion pipeline    │
                            └─────────────┬──────────────────────┘
                                          │
                       ┌──────────────────┼──────────────────────┐
                       ▼                  ▼                       │
           ┌───────────────────┐  ┌───────────────────────┐      │
           │ ingestion/logger  │  │  db.py (MongoDB)       │      │
           │ JSONL append-only │  │  Atlas cluster         │      │
           │ logs/inference_   │  │  rag_telemetry db      │      │
           │ logs.jsonl        │  │  inference_logs coll.  │      │
           └───────────────────┘  └───────────────────────┘      │
                                                                   │
                                    ┌──────────────────────────────┘
                                    ▼
                           Groq API (llama-3.1-8b-instant)
```

---

## Project Structure

```
rag/
├── sdk/
│   ├── __init__.py
│   ├── llm_wrapper.py       ← production LLM wrapper
│   └── pii_redactor.py      ← regex PII scrubbing
│
├── ingestion/
│   ├── __init__.py
│   └── logger.py            ← append-only JSONL pipeline
│
├── models/
│   ├── __init__.py
│   └── inference_log.py     ← Pydantic telemetry schema
│
├── logs/
│   └── inference_logs.jsonl ← auto-created at runtime
│
├── db.py                    ← MongoDB connection + analytics queries
├── rag_app.py               ← FastAPI app (extended)
├── frontend/
│   └── index.html           ← Chat UI + Analytics panel
├── chroma_db/               ← Persisted vector index
├── data/                    ← Source PDFs
├── Dockerfile
└── docker-compose.yml
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

Edit `.env`:

```env
GROQ_API_KEY=<your groq key>
HUGGINGFACEHUB_API_TOKEN=<your hf token>
MONGODB_URI=mongodb+srv://<user>:<pass>@cluster.mongodb.net/?appName=Cluster0
MONGODB_DB=rag_telemetry
```

### 3. Run locally

```bash
uvicorn rag_app:app --reload
```

Open `frontend/index.html` in your browser.

### 4. Run with Docker

```bash
docker-compose up --build
```

> **Note:** Docker Compose uses a local MongoDB container instead of Atlas.

---

## Ingestion Flow

```
LLM call (llm.invoke / llm.stream)
         │
         ▼
sdk/llm_wrapper.py
  1. Start timer (time.perf_counter)
  2. Call LLM
  3. Stop timer → latency_ms
  4. Extract token usage from response_metadata
  5. Redact PII from previews
  6. Build metadata dict
  7. Call log_inference(metadata)
         │
         ├─► ingestion/logger.py
         │     ├── Redact PII (2nd pass safety net)
         │     ├── Add ISO-8601 UTC timestamp
         │     └── Append JSON line → logs/inference_logs.jsonl
         │
         └─► db.py
               └── insert_one() → MongoDB Atlas
                   rag_telemetry.inference_logs
```

---

## Telemetry Schema

| Field | Type | Description |
|-------|------|-------------|
| `request_id` | UUID string | Unique ID per inference call |
| `session_id` | string | Groups calls from one browser tab |
| `provider` | string | LLM provider (e.g. `"groq"`) |
| `model` | string | Model name |
| `latency_ms` | float | Wall-clock time in milliseconds |
| `prompt_tokens` | int \| null | Tokens in the prompt |
| `completion_tokens` | int \| null | Tokens generated |
| `total_tokens` | int \| null | Total tokens consumed |
| `status` | `"success"` \| `"error"` | Outcome of the call |
| `input_preview` | string (200 chars) | PII-redacted prompt preview |
| `output_preview` | string (200 chars) | PII-redacted response preview |
| `timestamp` | ISO 8601 string | UTC datetime |
| `error_message` | string \| null | Exception message on failure |

---

## Analytics Endpoints

| Endpoint | Response |
|----------|----------|
| `GET /analytics/summary` | total_requests, total_failures, success_rate |
| `GET /analytics/latency` | avg / min / max / p95 latency_ms |
| `GET /analytics/tokens` | total and avg prompt/completion tokens |
| `GET /analytics/sessions` | per-session request counts |

All endpoints fall back to scanning `logs/inference_logs.jsonl` when MongoDB is unavailable.

---

## PII Redaction

Applied to `input_preview` and `output_preview` before any storage:

| Pattern | Replaced with |
|---------|---------------|
| Email addresses | `[EMAIL]` |
| Phone numbers | `[PHONE]` |
| Credit card numbers | `[CARD]` |
| US Social Security numbers | `[SSN]` |
| IPv4 addresses | `[IP]` |

---

## Multi-turn Session Support

- Each browser tab generates a `session_id` via `crypto.randomUUID()` at page load.
- `session_id` is stored in `sessionStorage` (persists for the tab's lifetime).
- Every `/ask` request includes `session_id` in the request body.
- All telemetry records are tagged with `session_id` in MongoDB + JSONL.
- The `/analytics/sessions` endpoint shows per-session request counts.

---

## Streaming

`POST /ask/stream` returns a `text/event-stream` (SSE) response:

```
data: The answer
data:  is streamed
data:  token by token
data: [DONE]
```

Telemetry is logged once the full stream completes.

---

## Scaling Considerations

| Concern | Current approach | Production path |
|---------|-----------------|-----------------|
| Write throughput | Thread-locked JSONL append | Replace with Kafka / Kinesis |
| Read latency | MongoDB indexes on `session_id`, `timestamp`, `status` | Add Atlas Search for text queries |
| Horizontal scaling | Single uvicorn process | Multiple workers behind nginx / gunicorn |
| Secrets | `.env` file | HashiCorp Vault / AWS Secrets Manager |
| Embeddings | HuggingFace Inference API | Self-hosted model for latency control |

---

## Failure Handling

- **MongoDB unavailable**: `db.py` catches `ConnectionFailure` at startup and sets `collection = None`. Every analytics endpoint falls back to the JSONL scanner.
- **HF Embedding API fails**: `EmbeddingManager` returns a random 384-dim vector (fallback). Answers may be less accurate but the app keeps running.
- **Groq API error**: `llm_wrapper.py` catches the exception, logs an error telemetry record (status=`"error"`), then re-raises so FastAPI returns an HTTP error to the client.
- **JSONL write fails**: Logged as an error, does not crash the request handler.

---

## Future Improvements

- [ ] OpenTelemetry traces for distributed tracing
- [ ] Grafana dashboard connected to MongoDB Atlas Charts
- [ ] Multi-provider support (OpenAI, Anthropic, Cohere)
- [ ] Token budget enforcement per session
- [ ] Webhook alerts on error rate spikes
- [ ] Re-ranking layer (cross-encoder) for better retrieval quality
- [ ] Redis session store for multi-process session sharing
- [ ] Fine-grained RBAC on analytics endpoints
