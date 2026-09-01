from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request

from backend.common.rate_limit import limiter
from backend.common.request_utils import get_request_data
from backend.services import vector_service

router = APIRouter(tags=["rag"])


@router.post("/vector")
@limiter.limit("30/minute")
async def vector_search(request: Request) -> Dict[str, Any]:
    data = await get_request_data(request)
    query = str(data.get("query") or data.get("message") or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query text is required.")
    top_k = int(data.get("top_k") or 5)
    target = str(data.get("target") or "auto")
    return {"results": vector_service.search_vectors(query, top_k=top_k, target=target)}


@router.post("/rag")
@limiter.limit("20/minute")
async def rag_search(request: Request) -> Dict[str, Any]:
    data = await get_request_data(request)
    query = str(data.get("query") or data.get("message") or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query text is required.")
    top_k = int(data.get("top_k") or 5)
    target = str(data.get("target") or "auto")
    session_id = data.get("session_id")
    learning_mode = data.get("learning_mode")
    user_mode = data.get("user_mode")

    resolved_session = str(session_id).strip() if session_id and str(session_id).strip() else None
    resolved_mode = str(learning_mode or "student").strip()
    resolved_user_mode = str(user_mode).strip() if user_mode and str(user_mode).strip() else None

    return vector_service.rag_answer(
        query,
        top_k=top_k,
        target=target,
        session_id=resolved_session,
        learning_mode=resolved_mode,
        user_mode=resolved_user_mode,
    )