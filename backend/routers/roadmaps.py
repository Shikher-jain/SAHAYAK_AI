"""Roadmaps router — learning roadmaps with progress tracking."""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from backend.auth_system.auth_service import require_user
from backend.auth_system.models import User
from backend.services import roadmap_service

router = APIRouter(prefix="/roadmaps", tags=["roadmaps"])


class ProgressUpdate(BaseModel):
    topic_id: str
    completed: bool


@router.get("")
def list_roadmaps() -> List[Dict[str, Any]]:
    """List all available learning roadmaps."""
    return roadmap_service.list_roadmaps()


@router.get("/{roadmap_id}")
def get_roadmap(roadmap_id: str):
    """Get a specific roadmap with topics and progress."""
    roadmap = roadmap_service.get_roadmap(roadmap_id)
    if not roadmap:
        return {"error": "Roadmap not found"}
    return roadmap


@router.get("/{roadmap_id}/progress")
def get_progress(roadmap_id: str, user: User = Depends(require_user)):
    """Get the user's progress on a roadmap."""
    progress = roadmap_service.get_progress(str(user.id), roadmap_id)
    percentage = roadmap_service.get_completion_percentage(str(user.id), roadmap_id)
    return {"roadmap_id": roadmap_id, "progress": progress, "completion_percentage": percentage}


@router.put("/{roadmap_id}/progress")
def update_progress(roadmap_id: str, req: ProgressUpdate, user: User = Depends(require_user)):
    """Update progress on a roadmap topic."""
    return roadmap_service.update_progress(str(user.id), roadmap_id, req.topic_id, req.completed)
