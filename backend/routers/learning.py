"""Learning mode router — student, teacher, self-learning paths with bookmarks and notes."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Form
from pydantic import BaseModel

from backend.auth_system.auth_service import get_current_user, require_user
from backend.auth_system.models import User
from backend.services import learning_service

router = APIRouter(prefix="/learning", tags=["learning"])


class ModeRequest(BaseModel):
    mode: str  # student | teacher | self_learning
    preferences: Optional[Dict[str, Any]] = None


class BookmarkRequest(BaseModel):
    document_id: str
    title: str = ""
    note: str = ""


class NoteRequest(BaseModel):
    title: str
    content: str = ""
    tags: str = ""


@router.get("/mode")
def get_learning_mode(user: Optional[User] = Depends(get_current_user)):
    """Get current learning mode (returns default if not authenticated)."""
    if not user:
        return {"mode": "student", "preferences": {}}
    return learning_service.get_mode(str(user.id))


@router.post("/mode")
def set_learning_mode(req: ModeRequest, user: User = Depends(require_user)):
    """Set learning mode (student, teacher, self_learning)."""
    return learning_service.set_mode(str(user.id), req.mode, req.preferences)


@router.get("/bookmarks")
def list_bookmarks(user: Optional[User] = Depends(get_current_user)) -> List[Dict[str, Any]]:
    """List bookmarks (empty list if not authenticated)."""
    if not user:
        return []
    return learning_service.get_bookmarks(str(user.id))


@router.post("/bookmarks")
def create_bookmark(req: BookmarkRequest, user: User = Depends(require_user)):
    """Add a bookmark for a document."""
    bookmark_id = learning_service.add_bookmark(str(user.id), req.document_id, req.title, req.note)
    return {"id": bookmark_id, "status": "ok"}


@router.post("/notes")
def create_note(req: NoteRequest, user: User = Depends(require_user)):
    """Add a learning note."""
    note_id = learning_service.add_note(str(user.id), req.title, req.content, req.tags)
    return {"id": note_id, "status": "ok"}


@router.get("/notes")
def list_notes(user: Optional[User] = Depends(get_current_user)) -> List[Dict[str, Any]]:
    """List learning notes (empty list if not authenticated)."""
    if not user:
        return []
    return learning_service.get_notes(str(user.id))
