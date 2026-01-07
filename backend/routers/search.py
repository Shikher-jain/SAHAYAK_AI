from fastapi import APIRouter, Form

from backend.services import vector_service

router = APIRouter(tags=["rag"])


@router.post("/vector")
def vector_search(query: str = Form(...), top_k: int = 5, target: str = "auto"):
    return {"results": vector_service.search_vectors(query, top_k=top_k, target=target)}


@router.post("/rag")
def rag_search(query: str = Form(...), top_k: int = 5, target: str = "auto"):
    return vector_service.rag_answer(query, top_k=top_k, target=target)
