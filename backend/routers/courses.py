"""Courses router — curated course catalog with search."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/courses", tags=["courses"])


class AddCourseRequest(BaseModel):
    title: str
    provider: str
    subject: str
    level: str = "Beginner"
    url: str = ""
    price: str = "Free"
    duration: str = ""


@router.get("/list")
def list_all_courses(subject: Optional[str] = None, level: Optional[str] = None) -> List[Dict[str, Any]]:
    """List all courses with optional filters."""
    from backend.services.course_service import list_courses
    return list_courses(subject=subject, level=level)


@router.get("/{course_id}")
def get_course(course_id: str):
    """Get a single course by ID."""
    from backend.services.course_service import get_course
    course = get_course(course_id)
    if not course:
        return {"error": "Course not found"}
    return course


@router.post("/add-source")
def add_course_source(req: AddCourseRequest):
    """Add a custom course source."""
    from backend.services.course_service import add_custom_source
    return add_custom_source(req.model_dump())


@router.get("/search")
def search_courses_endpoint(query: str) -> List[Dict[str, Any]]:
    """Search courses by keyword."""
    from backend.services.course_service import search_courses
    return search_courses(query)
