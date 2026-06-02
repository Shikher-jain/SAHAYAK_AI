"""Progress tracking router — course progress and quiz scores."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from backend.auth_system.auth_service import require_user
from backend.auth_system.models import User
from backend.services import progress_service

router = APIRouter(prefix="/progress", tags=["progress"])


class ProgressUpdateRequest(BaseModel):
    topics_completed: Optional[int] = None
    total_topics: Optional[int] = None
    quiz_score: Optional[float] = None
    time_spent_minutes: Optional[int] = None


@router.get("")
def get_all_progress(user: User = Depends(require_user)):
    """Get all course progress for the current user."""
    return progress_service.get_summary(str(user.id))


@router.get("/{course_id}")
def get_course_progress(course_id: str, user: User = Depends(require_user)):
    """Get progress for a specific course."""
    result = progress_service.get_course_progress(str(user.id), course_id)
    if not result:
        return {"error": "No progress found for this course"}
    return result


@router.put("/{course_id}")
def update_progress(course_id: str, req: ProgressUpdateRequest, user: User = Depends(require_user)):
    """Update progress for a course."""
    return progress_service.update_course_progress(
        str(user.id), course_id,
        topics_completed=req.topics_completed,
        total_topics=req.total_topics,
        quiz_score=req.quiz_score,
        time_spent_minutes=req.time_spent_minutes,
    )
