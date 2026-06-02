"""Stories service — user experience testimonials CRUD (SQLite)."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

_BASE_DIR = Path(__file__).resolve().parents[2]
_DB_DIR = _BASE_DIR / "data" / "stories"
_DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = _DB_DIR / "stories.db"


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_stories_db() -> None:
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS stories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            username TEXT DEFAULT '',
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            tags TEXT DEFAULT '',
            rating INTEGER DEFAULT 5,
            approved INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    # Seed some example stories
    cur = conn.execute("SELECT COUNT(*) FROM stories")
    if cur.fetchone()[0] == 0:
        seed_stories = [
            ("seed", "Priya S.", "Sahayak transformed my exam prep", "I used Sahayak to prepare for my board exams. The RAG-based Q&A helped me understand concepts deeply. Scored 95% in Maths!", "exam,mathematics,ncert", 5),
            ("seed", "Rahul K.", "Best tool for coding interviews", "The code ingestion feature let me upload my DSA notes and practice questions. Got placed at a top tech company!", "coding,interview,dsa", 5),
            ("seed", "Dr. Meena R.", "Invaluable for teaching", "As a teacher, I use Sahayak to create quizzes and share notes with my students. The AI counselor helps students choose the right career path.", "teaching,quiz,career", 4),
            ("seed", "Ankit P.", "Great for competitive exams", "Used Sahayak for JEE preparation. The roadmap feature kept me on track and the multilingual support helped me study in Hindi.", "jee,roadmap,hindi", 5),
        ]
        conn.executemany(
            "INSERT INTO stories (user_id, username, title, content, tags, rating) VALUES (?, ?, ?, ?, ?, ?)",
            seed_stories,
        )
    conn.commit()
    conn.close()


init_stories_db()


def list_stories(limit: int = 20, approved_only: bool = True) -> List[Dict[str, Any]]:
    conn = _get_conn()
    if approved_only:
        rows = conn.execute(
            "SELECT * FROM stories WHERE approved = 1 ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM stories ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def create_story(user_id: str, username: str, title: str, content: str, tags: str = "", rating: int = 5) -> int:
    conn = _get_conn()
    cur = conn.execute(
        "INSERT INTO stories (user_id, username, title, content, tags, rating) VALUES (?, ?, ?, ?, ?, ?)",
        (user_id, username, title, content, tags, min(max(rating, 1), 5)),
    )
    story_id = cur.lastrowid
    conn.commit()
    conn.close()
    return story_id


def update_story(story_id: int, user_id: str, title: str | None = None, content: str | None = None) -> bool:
    conn = _get_conn()
    updates = []
    params: list = []
    if title is not None:
        updates.append("title = ?")
        params.append(title)
    if content is not None:
        updates.append("content = ?")
        params.append(content)
    if not updates:
        conn.close()
        return False
    params.extend([story_id, user_id])
    cur = conn.execute(f"UPDATE stories SET {', '.join(updates)} WHERE id = ? AND user_id = ?", params)
    conn.commit()
    updated = cur.rowcount > 0
    conn.close()
    return updated
