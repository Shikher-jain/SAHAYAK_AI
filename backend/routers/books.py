"""Books router — NCERT and online textbook catalog."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter

router = APIRouter(prefix="/books", tags=["books"])


@router.get("/catalog")
def book_catalog(subject: Optional[str] = None, class_level: Optional[int] = None) -> List[Dict[str, Any]]:
    """List available books with optional filters."""
    from backend.services.books_service import list_books
    return list_books(subject=subject, class_level=class_level)


@router.get("/{book_id}")
def get_book(book_id: str):
    """Get a specific book by ID."""
    from backend.services.books_service import get_book
    book = get_book(book_id)
    if not book:
        return {"error": "Book not found"}
    return book


@router.get("/meta/subjects")
def available_subjects() -> List[str]:
    """List available subjects."""
    from backend.services.books_service import get_subjects
    return get_subjects()


@router.get("/meta/classes")
def available_classes() -> List[int]:
    """List available class levels."""
    from backend.services.books_service import get_class_levels
    return get_class_levels()
