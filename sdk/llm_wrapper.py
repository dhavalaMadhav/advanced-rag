"""
sdk/llm_wrapper.py
------------------
Production-grade LLM inference wrapper / lightweight SDK.

Responsibilities
----------------
* Wrap every LLM call so the rest of the codebase never calls
  llm.invoke() or llm.stream() directly.
* Measure precise wall-clock latency with time.perf_counter().
* Extract Groq token usage from response_metadata.
* Apply PII redaction on input/output previews.
* Route telemetry to the ingestion pipeline (JSONL + MongoDB).
* Expose a streaming variant that yields SSE-compatible chunks.
* Handle errors gracefully — always log failures, always re-raise
  so FastAPI can return a proper HTTP error response.

Public API
----------
  generate_response(llm, prompt, *, session_id, provider, model) -> str
  stream_response(llm, prompt, *, session_id, provider, model)    -> Iterator[str]
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Iterator

from sdk.pii_redactor import redact
from ingestion.logger import log_inference

logger = logging.getLogger(__name__)

# Maximum characters stored in preview fields (PII is redacted first)
_PREVIEW_LEN = 200


def _extract_token_usage(response) -> dict:
    """
    Extract token counts from a LangChain ChatGroq response object.

    Groq exposes usage under response_metadata["token_usage"] with keys:
      prompt_tokens, completion_tokens, total_tokens.

    Returns an empty dict if the data is absent (e.g. streaming mode
    or a different provider that uses different metadata keys).
    """
    try:
        usage = (
            response.response_metadata.get("token_usage")
            or response.response_metadata.get("usage")
            or {}
        )
        return {
            "prompt_tokens":     usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens":      usage.get("total_tokens"),
        }
    except Exception:
        return {}


def generate_response(
    llm,
    prompt: str,
    *,
    session_id: str = "default",
    provider: str = "groq",
    model: str = "llama-3.1-8b-instant",
) -> str:
    """
    Invoke the LLM, capture full telemetry, and return the answer string.

    Parameters
    ----------
    llm        : LangChain chat model instance (e.g. ChatGroq).
    prompt     : Full prompt string to send to the LLM.
    session_id : Caller-supplied session identifier for multi-turn tracking.
    provider   : Provider label stored in telemetry ("groq", "openai", …).
    model      : Model name stored in telemetry.

    Returns
    -------
    str
        The LLM-generated answer text.

    Raises
    ------
    Exception
        Re-raises any LLM/network error after logging the failure record.
    """
    request_id = str(uuid.uuid4())

    # ------------------------------------------------------------------ #
    # Inference call — timed with perf_counter for sub-millisecond precision
    # ------------------------------------------------------------------ #
    t_start = time.perf_counter()

    try:
        response = llm.invoke(prompt)
        t_end    = time.perf_counter()
        latency  = round((t_end - t_start) * 1000, 2)

        answer = response.content

        # ---------------------------------------------------------------- #
        # Build telemetry record
        # ---------------------------------------------------------------- #
        metadata: dict = {
            "request_id":     request_id,
            "session_id":     session_id,
            "provider":       provider,
            "model":          model,
            "latency_ms":     latency,
            "status":         "success",
            "input_preview":  redact(prompt[:_PREVIEW_LEN]),
            "output_preview": redact(answer[:_PREVIEW_LEN]),
        }

        # Token usage (may be absent depending on provider / response)
        metadata.update(_extract_token_usage(response))

        log_inference(metadata)

        logger.info(
            "✅ request=%s session=%s latency=%.1fms tokens=%s",
            request_id,
            session_id,
            latency,
            metadata.get("total_tokens"),
        )

        return answer

    except Exception as exc:
        t_end   = time.perf_counter()
        latency = round((t_end - t_start) * 1000, 2)

        error_metadata: dict = {
            "request_id":    request_id,
            "session_id":    session_id,
            "provider":      provider,
            "model":         model,
            "latency_ms":    latency,
            "status":        "error",
            "error_message": str(exc),
            "input_preview": redact(prompt[:_PREVIEW_LEN]),
        }

        log_inference(error_metadata)

        logger.error(
            "❌ request=%s session=%s error=%s",
            request_id,
            session_id,
            exc,
        )

        raise


def stream_response(
    llm,
    prompt: str,
    *,
    session_id: str = "default",
    provider: str = "groq",
    model: str = "llama-3.1-8b-instant",
) -> Iterator[str]:
    """
    Stream LLM tokens as they are generated.

    Yields each text chunk as it arrives from the provider.
    After the stream completes, one telemetry record is logged
    (latency = time to full completion, token counts not available
    for streaming mode — logged as None).

    Parameters
    ----------
    Same as generate_response().

    Yields
    ------
    str
        Raw text chunks from the LLM stream.
    """
    request_id  = str(uuid.uuid4())
    t_start     = time.perf_counter()
    full_output = []

    try:
        for chunk in llm.stream(prompt):
            text = chunk.content if hasattr(chunk, "content") else str(chunk)
            full_output.append(text)
            yield text

        t_end   = time.perf_counter()
        latency = round((t_end - t_start) * 1000, 2)
        answer  = "".join(full_output)

        metadata: dict = {
            "request_id":     request_id,
            "session_id":     session_id,
            "provider":       provider,
            "model":          model,
            "latency_ms":     latency,
            "status":         "success",
            "input_preview":  redact(prompt[:_PREVIEW_LEN]),
            "output_preview": redact(answer[:_PREVIEW_LEN]),
            # Token counts are not available in streaming mode
            "prompt_tokens":     None,
            "completion_tokens": None,
            "total_tokens":      None,
        }

        log_inference(metadata)
        logger.info(
            "✅ [stream] request=%s session=%s latency=%.1fms",
            request_id,
            session_id,
            latency,
        )

    except Exception as exc:
        t_end   = time.perf_counter()
        latency = round((t_end - t_start) * 1000, 2)

        error_metadata: dict = {
            "request_id":    request_id,
            "session_id":    session_id,
            "provider":      provider,
            "model":         model,
            "latency_ms":    latency,
            "status":        "error",
            "error_message": str(exc),
            "input_preview": redact(prompt[:_PREVIEW_LEN]),
        }

        log_inference(error_metadata)
        logger.error("❌ [stream] request=%s error=%s", request_id, exc)
        raise