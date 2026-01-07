from fastapi import APIRouter

from backend.vector_store import qdrant_store

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/health")
def health():
    return qdrant_store.status()


@router.get("/uploads")
def uploaded_files():
    return {"files": qdrant_store.recent_payloads()}
