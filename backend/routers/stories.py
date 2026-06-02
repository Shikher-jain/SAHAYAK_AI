"""Stories router — user experience testimonials."""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from backend.auth_system.auth_service import require_user
from backend.auth_system.models import User
from backend.services import stories_service

router = APIRouter(prefix="/stories", tags=["stories"])


class StoryCreate(BaseModel):
    title: str
    content: str
    tags: str = ""
    rating: int = Field(default=5, ge=1, le=5)


class StoryUpdate(BaseModel):
    title: str | None = None
    content: str | None = None


@router.get("")
def get_stories(limit: int = 20) -> List[Dict[str, Any]]:
    """List approved user stories/testimonials."""
    return stories_service.list_stories(limit=limit)


@router.post("")
def create_story(req: StoryCreate, user: User = Depends(require_user)):
    """Submit a new user experience story."""
    story_id = stories_service.create_story(
        str(user.id), user.username, req.title, req.content, req.tags, req.rating
    )
    return {"id": story_id, "status": "ok"}


@router.put("/{story_id}")
def update_story(story_id: int, req: StoryUpdate, user: User = Depends(require_user)):
    """Update your own story."""
    ok = stories_service.update_story(story_id, str(user.id), req.title, req.content)
    return {"updated": ok}
