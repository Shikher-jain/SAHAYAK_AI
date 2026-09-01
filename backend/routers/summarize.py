from fastapi import APIRouter, Depends, HTTPException, Request

from backend.auth import api_key_auth
from backend.common.request_utils import get_request_data
from backend.services import vector_service

router = APIRouter(tags=["summaries"], dependencies=[Depends(api_key_auth)])


@router.post("/text")
async def summarize_text_endpoint(request: Request):
    data = await get_request_data(request)
    text = str(data.get("text") or data.get("content") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text content is required.")
    return {"summary": vector_service.summarize_text(text)}
