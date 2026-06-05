"""Quiz service — generate and evaluate interactive Q&A from ingested content."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

_BASE_DIR = Path(__file__).resolve().parents[2]
_DB_DIR = _BASE_DIR / "data" / "quiz"
_DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = _DB_DIR / "quiz.db"


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_quiz_db() -> None:
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS quizzes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            topic TEXT DEFAULT '',
            questions TEXT DEFAULT '[]',
            score REAL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    conn.close()


init_quiz_db()


def generate_quiz_from_context(context: str, num_questions: int = 5) -> List[Dict[str, Any]]:
    """Generate quiz questions from context using the LLM generator."""
    from backend.rag.generator import Generator
    gen = Generator()
    prompt = (
        f"Generate exactly {num_questions} quiz questions from this content. "
        "Return JSON array of objects with fields: question, options (list of 4 strings), "
        "correct_answer (index 0-3), explanation.\n\n"
        f"Content:\n{context[:2000]}\n\nQuestions JSON:"
    )
    try:
        result = gen.generate_answer(context[:2000], prompt, sources=None)
        # Parse the answer as JSON; fall back to simple extraction
        answer_text = result.get("answer", "")
        # Try to extract JSON array from the response
        start = answer_text.find("[")
        end = answer_text.rfind("]") + 1
        if start >= 0 and end > start:
            questions = json.loads(answer_text[start:end])
            return questions
    except Exception:
        import logging
        logger = logging.getLogger("sahayak.quiz_service")
        raw_preview = locals().get('answer_text', '')[:500] if isinstance(locals().get('answer_text'), str) else "N/A"
        logger.warning("Quiz generation JSON parse failed. Raw LLM output (first 500 chars): %s", raw_preview)
    # Fallback: generate simple fill-in-the-blank questions from sentences
    return _fallback_quiz(context, num_questions)


def _fallback_quiz(context: str, num_questions: int) -> List[Dict[str, Any]]:
    """Generate simple questions when the LLM is unavailable."""
    sentences = [s.strip() for s in context.replace("\n", " ").split(".") if len(s.strip()) > 30]
    questions: List[Dict[str, Any]] = []
    for sentence in sentences[:num_questions]:
        words = sentence.split()
        if len(words) < 5:
            continue
        # Mask a keyword (middle word)
        mask_idx = len(words) // 2
        blanked = " ".join(words[:mask_idx] + ["___"] + words[mask_idx + 1:])
        questions.append({
            "question": f"Fill in the blank: {blanked}",
            "options": [words[mask_idx], "unknown", "none", "all"],
            "correct_answer": 0,
            "explanation": f"The answer is: {words[mask_idx]}",
        })
    return questions


def save_quiz(user_id: str, topic: str, questions: List[Dict], score: float) -> int:
    conn = _get_conn()
    cur = conn.execute(
        "INSERT INTO quizzes (user_id, topic, questions, score) VALUES (?, ?, ?, ?)",
        (user_id, topic, json.dumps(questions), score),
    )
    quiz_id = cur.lastrowid
    conn.commit()
    conn.close()
    return quiz_id


def get_quiz_history(user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, topic, score, created_at FROM quizzes WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
        (user_id, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
