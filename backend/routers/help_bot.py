"""Help center router — FAQ and navigation bot."""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from backend.services.help_bot import HelpBot

router = APIRouter(prefix="/help", tags=["help"])
_bot = HelpBot()


class HelpRequest(BaseModel):
    question: str


@router.post("/ask")
def ask_help(req: HelpRequest):
    """Ask the help bot a question about the platform."""
    return _bot.answer(req.question)


@router.get("/faq")
def get_faq():
    """Get all FAQ entries."""
    return _bot.get_all_faqs()
