"""Quiz router — generate and track interactive quizzes."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Form
from pydantic import BaseModel

from backend.auth_system.auth_service import get_current_user, require_user
from backend.auth_system.models import User
from backend.services import quiz_service, vector_service

router = APIRouter(prefix="/quiz", tags=["quiz"])


class QuizGenerateRequest(BaseModel):
    topic: str
    num_questions: int = 5


class QuizAnswerRequest(BaseModel):
    quiz_id: Optional[int] = None
    topic: str = ""
    questions: List[Dict[str, Any]]
    answers: List[int]


@router.post("/generate")
def generate_quiz(
    req: QuizGenerateRequest,
    user: Optional[User] = Depends(get_current_user),
):
    """Generate a quiz from ingested content on the given topic (public — auth optional)."""
    # Search for relevant context from the vector store
    hits = vector_service.search_vectors(req.topic, top_k=3, target="auto")
    context = "\n\n".join(h.get("content", "") for h in hits if h.get("content"))
    if not context.strip():
        context = req.topic
    questions = quiz_service.generate_quiz_from_context(context, req.num_questions)
    return {"topic": req.topic, "questions": questions}


@router.post("/answer")
def submit_quiz(
    req: QuizAnswerRequest,
    user: Optional[User] = Depends(get_current_user),
):
    """Submit quiz answers and get a score (public — auth optional)."""
    correct = 0
    results = []
    for idx, question in enumerate(req.questions):
        user_answer = req.answers[idx] if idx < len(req.answers) else -1
        is_correct = user_answer == question.get("correct_answer", -1)
        if is_correct:
            correct += 1
        results.append({
            "question": question.get("question", ""),
            "correct": is_correct,
            "explanation": question.get("explanation", ""),
        })
    total = len(req.questions) or 1
    score = correct / total
    # Save to history only if user is authenticated
    quiz_id = None
    if user:
        quiz_id = quiz_service.save_quiz(str(user.id), req.topic, req.questions, score)
    return {
        "quiz_id": quiz_id,
        "score": score,
        "correct": correct,
        "total": total,
        "results": results,
    }


@router.get("/history")
def quiz_history(user: User = Depends(require_user)) -> List[Dict[str, Any]]:
    """Get quiz history for the current user."""
    return quiz_service.get_quiz_history(str(user.id))
