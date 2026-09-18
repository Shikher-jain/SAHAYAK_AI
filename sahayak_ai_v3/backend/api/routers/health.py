"""v2 advanced — Zero-cost keep-alive health probe.

This router exists for ONE reason: Render free-tier web services spin down
after 15 min of zero inbound traffic, which cold-starts every subsequent
request. A single lightweight GET that answers in microseconds also keeps
Render's own ``healthCheckPath`` green (no crash loops from 502s).

Deliberately lightweight by construction (AGENTS Rule 1 — zero compute):
  * no DB, no Qdrant/Neo4j pings, no cache touches — a cold path answers in
    <1 ms and consumes ~0 RAM;
  * pickled-nothing, JSON-only — the payload is three constant keys;

Mount this router in the same **guarded** v2 block as the supervisor +
payments + audio (see ``backend/api/main.py``) so the legacy env is never
entangled:
    from sahayak_ai_v3.backend.api.routers import v2_chat, payments, audio
    app.include_router(health.router)
"""
from datetime import datetime, timezone

from fastapi import APIRouter

router = APIRouter(prefix="/api/v2/health", tags=["v2 Health"])


@router.get("/ping", summary="Zero-compute keep-alive probe")
async def ping() -> dict:
    """Answer in <1 ms so cron CRON/JOB/CDN pings never cold-start the box."""
    return {"status": "alive", "timestamp": datetime.now(timezone.utc).isoformat()}
