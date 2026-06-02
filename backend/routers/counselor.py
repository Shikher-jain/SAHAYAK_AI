"""AI Counselor router — academic and career guidance."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from backend.services.counselor_service import CounselorService

router = APIRouter(prefix="/counselor", tags=["counselor"])
_counselor = CounselorService()


class CounselorChatRequest(BaseModel):
    message: str
    domain: str = "general"
    history: str = ""


@router.post("/chat")
def counselor_chat(req: CounselorChatRequest):
    """Chat with the AI counselor for academic/career guidance."""
    return _counselor.chat(req.message, domain=req.domain, history=req.history)


@router.get("/suggestions")
def domain_suggestions(domain: str = "stem") -> List[Dict[str, str]]:
    """Get pre-built career suggestions for a domain."""
    return _counselor.get_domain_suggestions(domain)
