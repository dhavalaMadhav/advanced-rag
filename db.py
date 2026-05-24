"""
db.py
-----
MongoDB persistence layer for inference telemetry.

Design decisions
----------------
* Uses synchronous pymongo so it integrates cleanly with FastAPI thread-pool
  workers without requiring an async event loop.
* Connection is established once at module import time and reused
  across all requests (connection pooling handled by pymongo driver).
* Graceful degradation: if MongoDB is unreachable, collection = None and
  every write is silently skipped. The app keeps running with JSONL only.

Collections
-----------
  rag_telemetry.inference_logs  -- one document per LLM inference call
"""

import os
import logging

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Connection setup
# ---------------------------------------------------------------------------

_client     = None
_db         = None
collection  = None        # exposed to callers; None when unavailable

MONGODB_URI = os.getenv("MONGODB_URI", "")
MONGODB_DB  = os.getenv("MONGODB_DB", "rag_telemetry")

if MONGODB_URI:
    try:
        from pymongo import MongoClient
        from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

        _client = MongoClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
        )

        # Validate the connection
        _client.admin.command("ping")

        _db        = _client[MONGODB_DB]
        collection = _db["inference_logs"]

        # Performance indexes
        collection.create_index("session_id", background=True)
        collection.create_index("timestamp",  background=True)
        collection.create_index("status",     background=True)

        logger.info("[OK] MongoDB connected -> %s.inference_logs", MONGODB_DB)
        print(f"[OK] MongoDB connected -> {MONGODB_DB}.inference_logs")

    except Exception as exc:
        logger.warning("[WARN] MongoDB unavailable (%s) - falling back to JSONL only", exc)
        print(f"[WARN] MongoDB unavailable - falling back to JSONL only")
        collection = None
else:
    logger.warning("[WARN] MONGODB_URI not set - falling back to JSONL only")
    print("[WARN] MONGODB_URI not set - falling back to JSONL only")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def insert_log(record: dict) -> bool:
    """Insert a single telemetry record into MongoDB."""
    if collection is None:
        return False
    try:
        collection.insert_one(record)
        return True
    except Exception as exc:
        logger.error("[ERROR] MongoDB insert failed: %s", exc)
        return False


def get_all_logs() -> list:
    """Retrieve all inference logs sorted by timestamp descending."""
    if collection is None:
        return []
    try:
        docs = list(collection.find({}, {"_id": 0}).sort("timestamp", -1))
        return docs
    except Exception as exc:
        logger.error("[ERROR] MongoDB query failed: %s", exc)
        return []


def get_analytics_summary() -> dict:
    """Return aggregate counts: total requests, successes, failures."""
    if collection is None:
        return {}
    try:
        total    = collection.count_documents({})
        failures = collection.count_documents({"status": "error"})
        return {
            "total_requests":  total,
            "total_failures":  failures,
            "total_successes": total - failures,
            "success_rate":    round((total - failures) / total * 100, 2) if total else 0,
            "mongodb":         True,
        }
    except Exception as exc:
        logger.error("[ERROR] Analytics summary failed: %s", exc)
        return {}


def get_latency_stats() -> dict:
    """Compute avg / min / max latency_ms using MongoDB aggregation."""
    if collection is None:
        return {}
    try:
        pipeline = [
            {"$match": {"status": "success", "latency_ms": {"$ne": None}}},
            {
                "$group": {
                    "_id":           None,
                    "avg_latency_ms": {"$avg": "$latency_ms"},
                    "min_latency_ms": {"$min": "$latency_ms"},
                    "max_latency_ms": {"$max": "$latency_ms"},
                }
            },
        ]
        result = list(collection.aggregate(pipeline))
        if not result:
            return {}
        r = result[0]

        # p95 via separate sort+skip (percentile operator needs MongoDB 7+)
        total_success = collection.count_documents({"status": "success", "latency_ms": {"$ne": None}})
        p95_idx = max(0, int(total_success * 0.95) - 1)
        p95_cursor = list(
            collection.find(
                {"status": "success", "latency_ms": {"$ne": None}},
                {"latency_ms": 1, "_id": 0}
            ).sort("latency_ms", 1).skip(p95_idx).limit(1)
        )
        p95 = p95_cursor[0]["latency_ms"] if p95_cursor else r.get("max_latency_ms", 0)

        return {
            "avg_latency_ms": round(r.get("avg_latency_ms") or 0, 2),
            "min_latency_ms": round(r.get("min_latency_ms") or 0, 2),
            "max_latency_ms": round(r.get("max_latency_ms") or 0, 2),
            "p95_latency_ms": round(p95, 2),
        }
    except Exception as exc:
        logger.error("[ERROR] Latency stats failed: %s", exc)
        return {}


def get_token_stats() -> dict:
    """Compute total and average token usage across all successful calls."""
    if collection is None:
        return {}
    try:
        pipeline = [
            {"$match": {"status": "success", "total_tokens": {"$ne": None}}},
            {
                "$group": {
                    "_id":                    None,
                    "total_prompt_tokens":    {"$sum": "$prompt_tokens"},
                    "total_completion_tokens":{"$sum": "$completion_tokens"},
                    "total_tokens":           {"$sum": "$total_tokens"},
                    "avg_prompt_tokens":      {"$avg": "$prompt_tokens"},
                    "avg_completion_tokens":  {"$avg": "$completion_tokens"},
                }
            },
        ]
        result = list(collection.aggregate(pipeline))
        if not result:
            return {}
        r = result[0]
        return {
            "total_prompt_tokens":     int(r.get("total_prompt_tokens") or 0),
            "total_completion_tokens": int(r.get("total_completion_tokens") or 0),
            "total_tokens":            int(r.get("total_tokens") or 0),
            "avg_prompt_tokens":       round(r.get("avg_prompt_tokens") or 0, 1),
            "avg_completion_tokens":   round(r.get("avg_completion_tokens") or 0, 1),
        }
    except Exception as exc:
        logger.error("[ERROR] Token stats failed: %s", exc)
        return {}


def get_session_stats() -> list:
    """Return per-session request counts, sorted by most active first."""
    if collection is None:
        return []
    try:
        pipeline = [
            {
                "$group": {
                    "_id":           "$session_id",
                    "request_count": {"$sum": 1},
                    "last_seen":     {"$max": "$timestamp"},
                }
            },
            {"$sort":  {"request_count": -1}},
            {"$limit": 50},
        ]
        result = list(collection.aggregate(pipeline))
        return [
            {
                "session_id":    r["_id"],
                "request_count": r["request_count"],
                "last_seen":     r["last_seen"],
            }
            for r in result
        ]
    except Exception as exc:
        logger.error("[ERROR] Session stats failed: %s", exc)
        return []