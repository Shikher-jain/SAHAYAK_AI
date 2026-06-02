from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Form

from backend.auth import api_key_auth
from backend.services import vector_service

router = APIRouter(tags=["rag"], dependencies=[Depends(api_key_auth)])


@router.post("/vector")
def vector_search(query: str = Form(...), top_k: int = 5, target: str = "auto"):
    return {"results": vector_service.search_vectors(query, top_k=top_k, target=target)}


@router.post("/rag")
def rag_search(
    query: str = Form(...),
    top_k: int = 5,
    target: str = "auto",
    session_id: Optional[str] = Form(None),
) -> Dict[str, Any]:
    resolved_session = session_id.strip() if session_id and session_id.strip() else None
    return vector_service.rag_answer(
        query,
        top_k=top_k,
        target=target,
        session_id=resolved_session,
    )
