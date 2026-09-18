from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from backend.auth import api_key_auth
from backend.services import vector_service

router = APIRouter(tags=["summaries"], dependencies=[Depends(api_key_auth)])


class SummarizeTextRequest(BaseModel):
    text: str = Field(..., description="The text content to summarize")


@router.post("/text")
async def summarize_text_endpoint(payload: SummarizeTextRequest):
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text content is required.")
    return {"summary": vector_service.summarize_text(text)}
