from typing import Any, Dict, Optional

from fastapi import APIRouter, Form, Request

from backend.common.rate_limit import limiter
from backend.services import vector_service

router = APIRouter(tags=["rag"])


@router.post("/vector")
@limiter.limit("30/minute")
def vector_search(request: Request, query: str = Form(...), top_k: int = 5, target: str = "auto"):
    return {"results": vector_service.search_vectors(query, top_k=top_k, target=target)}


@router.post("/rag")
@limiter.limit("20/minute")
def rag_search(
    request: Request,
    query: str = Form(...),
    top_k: int = 5,
    target: str = "auto",
    session_id: Optional[str] = Form(None),
    learning_mode: Optional[str] = Form("student"),
    user_mode: Optional[str] = Form(None),
) -> Dict[str, Any]:
    resolved_session = session_id.strip() if session_id and session_id.strip() else None
    resolved_mode = (learning_mode or "student").strip()
    return vector_service.rag_answer(
        query,
        top_k=top_k,
        target=target,
        session_id=resolved_session,
        learning_mode=resolved_mode,
        user_mode=user_mode,
    )