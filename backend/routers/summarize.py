from fastapi import APIRouter, Depends, Form

from backend.auth import api_key_auth
from backend.services import vector_service

router = APIRouter(tags=["summaries"], dependencies=[Depends(api_key_auth)])


@router.post("/text")
def summarize_text_endpoint(text: str = Form(...)):
    return {"summary": vector_service.summarize_text(text)}
