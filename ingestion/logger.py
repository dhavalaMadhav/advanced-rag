"""
ingestion/logger.py
-------------------
Lightweight, append-only JSONL ingestion pipeline.

Design decisions
----------------
* Append-only architecture: each inference record is written as a
  single newline-delimited JSON line. The file is never read back or
  rewritten — only opened in "append" mode. This makes it safe for
  concurrent writers (lock protected) and trivial to ingest into any
  downstream system (Kafka, BigQuery, S3, etc.).

* Thread safety: a threading.Lock() guard prevents interleaved writes
  when FastAPI uses its default thread-pool for sync endpoints.

* PII is redacted *before* writing — the raw text never touches disk.

* MongoDB insertion is attempted in parallel with the JSONL write so
  neither path blocks the other.

Log file
--------
  logs/inference_logs.jsonl  (created automatically if absent)
"""

import json
import os
import logging
import threading
from datetime import datetime, timezone

from sdk.pii_redactor import redact

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# File path — relative to the project root (where rag_app.py lives)
# ---------------------------------------------------------------------------
_LOG_DIR  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
_LOG_FILE = os.path.join(_LOG_DIR, "inference_logs.jsonl")

# Ensure the logs/ directory exists at import time
os.makedirs(_LOG_DIR, exist_ok=True)

# Thread lock so concurrent requests don't interleave JSON lines
_file_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def log_inference(metadata: dict) -> None:
    """
    Persist one inference telemetry record.

    Steps
    -----
    1. Redact PII from text preview fields.
    2. Stamp with ISO 8601 UTC timestamp.
    3. Append as a single JSON line to logs/inference_logs.jsonl.
    4. Insert into MongoDB (best-effort, non-blocking via db module).

    Parameters
    ----------
    metadata : dict
        Raw telemetry dict produced by sdk/llm_wrapper.py.
        Keys: request_id, session_id, provider, model, latency_ms,
              prompt_tokens, completion_tokens, total_tokens, status,
              input_preview, output_preview, timestamp, error_message.
    """

    # ------------------------------------------------------------------
    # 1. PII redaction on preview fields
    # ------------------------------------------------------------------
    if metadata.get("input_preview"):
        metadata["input_preview"] = redact(metadata["input_preview"])

    if metadata.get("output_preview"):
        metadata["output_preview"] = redact(metadata["output_preview"])

    if metadata.get("error_message"):
        metadata["error_message"] = redact(metadata["error_message"])

    # ------------------------------------------------------------------
    # 2. Ensure a clean ISO timestamp is present
    # ------------------------------------------------------------------
    if "timestamp" not in metadata or not isinstance(metadata["timestamp"], str):
        metadata["timestamp"] = datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # 3. Append to JSONL file (thread-safe)
    # ------------------------------------------------------------------
    try:
        with _file_lock:
            with open(_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(metadata, default=str) + "\n")
    except OSError as exc:
        logger.error("❌ JSONL write failed: %s", exc)

    # ------------------------------------------------------------------
    # 4. Persist to MongoDB (best-effort — failure doesn't raise)
    # ------------------------------------------------------------------
    try:
        from db import insert_log        # lazy import avoids circular deps
        insert_log(metadata)
    except Exception as exc:
        logger.warning("⚠️  MongoDB insert skipped: %s", exc)


def read_logs(limit: int = 100) -> list[dict]:
    """
    Read the last *limit* records from the JSONL file.

    Used as a fallback analytics source when MongoDB is unavailable.

    Parameters
    ----------
    limit : int
        Maximum number of most-recent records to return.

    Returns
    -------
    list[dict]
        Parsed log records, newest-first.
    """
    if not os.path.exists(_LOG_FILE):
        return []

    records = []
    try:
        with open(_LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except OSError as exc:
        logger.error("❌ JSONL read failed: %s", exc)

    # Return newest first
    return list(reversed(records[-limit:]))