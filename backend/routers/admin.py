from fastapi import APIRouter, Depends

from backend.auth import api_key_auth
from backend.vector_store import qdrant_store

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(api_key_auth)])


@router.get("/health")
def health():
    return qdrant_store.status()


@router.get("/uploads")
def uploaded_files():
    return {"files": qdrant_store.recent_payloads()}
