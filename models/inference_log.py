"""
models/inference_log.py
-----------------------
Pydantic data model for a single LLM inference telemetry record.

This model is the canonical schema shared by:
  - ingestion/logger.py  (JSONL serialisation)
  - db.py                (MongoDB document insertion)
  - Analytics endpoints  (response serialisation)

All 13 telemetry fields required by the assignment are present.
Optional fields default to None so partial records (e.g. error
paths that have no token counts) can still be stored cleanly.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field
import uuid


class InferenceLog(BaseModel):
    """
    Captures the full telemetry payload for one LLM inference call.

    Fields
    ------
    request_id         : Unique ID for this inference call (UUID4).
    session_id         : Groups calls belonging to one user session.
    provider           : LLM provider name, e.g. "groq", "openai".
    model              : Model identifier, e.g. "llama-3.1-8b-instant".
    latency_ms         : Wall-clock latency in milliseconds.
    prompt_tokens      : Number of prompt tokens consumed (if available).
    completion_tokens  : Number of completion tokens produced (if available).
    total_tokens       : Total tokens = prompt + completion (if available).
    status             : "success" | "error".
    input_preview      : First 200 chars of the prompt (PII-redacted).
    output_preview     : First 200 chars of the response (PII-redacted).
    timestamp          : UTC datetime when the call was made.
    error_message      : Exception message on failure, else None.
    """

    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = "default"
    provider: str = "groq"
    model: str = "llama-3.1-8b-instant"
    latency_ms: Optional[float] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    status: str = "success"                    # "success" | "error"
    input_preview: Optional[str] = None
    output_preview: Optional[str] = None
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    error_message: Optional[str] = None

    class Config:
        # Allow datetime serialisation in JSON responses
        json_encoders = {datetime: lambda v: v.isoformat()}

    def to_mongo_dict(self) -> dict:
        """
        Serialise to a plain dict suitable for MongoDB insertion.
        Converts datetime to ISO string so it round-trips cleanly.
        """
        d = self.model_dump()
        d["timestamp"] = self.timestamp.isoformat()
        return d

    def to_jsonl_dict(self) -> dict:
        """
        Serialise to a plain dict suitable for JSONL line append.
        """
        return self.to_mongo_dict()
