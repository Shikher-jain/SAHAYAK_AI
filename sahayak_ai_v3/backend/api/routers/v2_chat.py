"""v2 Agent chat + voice streaming endpoints.

  POST /api/v2/chat/orchestrate  — runs the Multi-Agent Supervisor
  POST /api/v2/chat/voice-stream — 5-10s audio blob -> Groq Whisper -> Supervisor

All dependencies are imported inside handlers so this module also imports cleanly
in the legacy env (where the v2 deps are absent) — main.py guards registration.
"""
import logging
from typing import List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, BackgroundTasks, Response, Depends
from pydantic import BaseModel
from fastapi.responses import JSONResponse

from backend.services.payments.user_store import user_store
from sahayak_ai_v3.backend.cache.semantic_cache import semantic_cache
from sahayak_ai_v3.backend.security.dependencies import RateLimitDependency

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v2/chat", tags=["v2 Chat"])


class ChatRequest(BaseModel):
    message: str
    user_id: str = "guest_user"
    tier: str = "free"


class ChatResponse(BaseModel):
    routed_agent: str
    response: str
    emergency_triggered: bool
    distress_level: float
    citations: List[dict] = []
    cached: bool = False


@router.post("/orchestrate", response_model=ChatResponse, dependencies=[Depends(RateLimitDependency)])
async def orchestrate(req: ChatRequest, background_tasks: BackgroundTasks, response: Response):
    """Runs the supervisor graph, bypassing via Semantic Cache if identical query exists."""
    from backend.services.agents.supervisor import supervisor_app

    # 1. Semantic Cache Lookup
    cached_hit = await semantic_cache.get_cached_response(req.message, req.user_id)
    if cached_hit:
        response.headers["X-Cache-Status"] = "HIT"
        return ChatResponse(
            routed_agent=cached_hit["routed_agent"],
            response=cached_hit["response"],
            emergency_triggered=False,
            distress_level=0.0,
            citations=cached_hit.get("citations", []),
            cached=True
        )

    response.headers["X-Cache-Status"] = "MISS"
    # 2. Cache Miss - Proceed with Agent Orchestration
    history = user_store.get_history(req.user_id, limit=10)
    initial = {
        "messages": [*history, {"role": "user", "content": req.message}],
        "next_agent": "",
        "user_id": req.user_id,
        "user_tier": req.tier,
        "distress_level": 0.0,
        "emergency_flag": False,
        "final_output": "",
        "citations": [],
    }
    result = await supervisor_app.ainvoke(initial)

    user_store.append_message(req.user_id, "user", req.message)
    
    response_data = {
        "routed_agent": result.get("next_agent", "unknown"),
        "response": result.get("final_output", ""),
        "emergency_triggered": bool(result.get("emergency_flag")),
        "distress_level": float(result.get("distress_level") or 0.0),
        "citations": list(result.get("citations") or []),
        "cached": False
    }

    if result.get("final_output"):
        user_store.append_message(req.user_id, "assistant", result["final_output"])
        # 3. Asynchronously store the new response in cache
        background_tasks.add_task(semantic_cache.set_cached_response, req.message, response_data)

    return ChatResponse(**response_data)


async def _transcribe(data: bytes, filename: str) -> str:
    """Blocks async-free: Groq Whisper transcription of one audio blob."""
    from backend.services.agents.supervisor import _groq

    resp = await _groq().audio.transcriptions.create(
        model="whisper-large-v3", file=(filename, data), response_format="text"
    )
    return (resp or "").strip()


@router.post("/voice-stream", dependencies=[Depends(RateLimitDependency)])
async def voice_stream(
    background_tasks: BackgroundTasks,
    response: Response,
    audio: UploadFile = File(...),
    user_id: str = Form("guest_user"),
    tier: str = Form("free"),
):
    """Near-real-time voice: transcribes a 5-10s MediaRecorder blob, then orchestrates."""
    data = await audio.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio payload.")
    try:
        transcript = await _transcribe(data, audio.filename or "voice.webm")
    except Exception as e:
        logger.warning("Whisper transcription failed: %s", e)
        raise HTTPException(status_code=502, detail="Audio transcription failed.")
    if not transcript:
        raise HTTPException(status_code=422, detail="Could not transcribe audio.")

    chat = await orchestrate(ChatRequest(message=transcript, user_id=user_id, tier=tier), background_tasks, response)
    payload = chat.model_dump()
    payload["transcript"] = transcript
    return payload