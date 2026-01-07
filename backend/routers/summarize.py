from fastapi import APIRouter, Form

from backend.services import vector_service

router = APIRouter(tags=["summaries"])


@router.post("/text")
def summarize_text_endpoint(text: str = Form(...)):
    return {"summary": vector_service.summarize_text(text)}
