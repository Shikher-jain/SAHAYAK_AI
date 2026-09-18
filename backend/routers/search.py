from typing import Any, Dict, Optional, List

from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel, Field

from backend.common.rate_limit import limiter
from backend.services import vector_service
from backend.auth_system.auth_service import get_current_user
from backend.auth_system.models import User

router = APIRouter(tags=["rag"])


class VectorSearchRequest(BaseModel):
    query: Optional[str] = Field(None, description="The search query text")
    message: Optional[str] = Field(None, description="Alternative to query")
    top_k: int = Field(5, description="Number of results to return")
    target: str = Field("auto", description="Target collection or 'auto'")


class RagSearchRequest(BaseModel):
    query: Optional[str] = Field(None, description="The search query text")
    message: Optional[str] = Field(None, description="Alternative to query")
    top_k: int = Field(5, description="Number of results to return")
    target: str = Field("auto", description="Target collection or 'auto'")
    session_id: Optional[str] = Field(None, description="Optional session ID for chat history")
    learning_mode: str = Field("student", description="Learning mode context")
    user_mode: Optional[str] = Field(None, description="User-defined mode context")


@router.post("/vector", response_model=Dict[str, Any])
@limiter.limit("30/minute")
async def vector_search(request: Request, payload: VectorSearchRequest, user: Optional[User] = Depends(get_current_user)):
    query = (payload.query or payload.message or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query text is required.")
        
    filters = {"user_id": str(user.id)} if user else None
    results = await vector_service.search_vectors(query, top_k=payload.top_k, target=payload.target, filters=filters)
    return {"results": results}


@router.post("/rag", response_model=Dict[str, Any])
@limiter.limit("20/minute")
async def rag_search(request: Request, payload: RagSearchRequest, user: Optional[User] = Depends(get_current_user)):
    query = (payload.query or payload.message or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query text is required.")
        
    resolved_session = payload.session_id.strip() if payload.session_id and payload.session_id.strip() else None
    resolved_mode = payload.learning_mode.strip()
    resolved_user_mode = payload.user_mode.strip() if payload.user_mode and payload.user_mode.strip() else None

    filters = {"user_id": str(user.id)} if user else None

    return await vector_service.rag_answer(
        query,
        top_k=payload.top_k,
        target=payload.target,
        session_id=resolved_session,
        learning_mode=resolved_mode,
        user_mode=resolved_user_mode,
        filters=filters,
    )