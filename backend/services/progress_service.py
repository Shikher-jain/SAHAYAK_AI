"""Progress tracking service — course progress, quiz scores, time tracking."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

_BASE_DIR = Path(__file__).resolve().parents[2]
_DB_DIR = _BASE_DIR / "data" / "progress"
_DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = _DB_DIR / "progress.db"


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_progress_db() -> None:
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS course_progress (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            course_id TEXT NOT NULL,
            topics_completed INTEGER DEFAULT 0,
            total_topics INTEGER DEFAULT 0,
            quiz_scores TEXT DEFAULT '[]',
            time_spent_minutes INTEGER DEFAULT 0,
            last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    conn.close()


init_progress_db()


def get_all_progress(user_id: str) -> List[Dict[str, Any]]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM course_progress WHERE user_id = ? ORDER BY last_active DESC", (user_id,)
    ).fetchall()
    conn.close()
    results = []
    for r in rows:
        d = dict(r)
        d["quiz_scores"] = json.loads(d.get("quiz_scores", "[]"))
        d["completion_percentage"] = round(d["topics_completed"] / max(d["total_topics"], 1) * 100, 1)
        results.append(d)
    return results


def get_course_progress(user_id: str, course_id: str) -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM course_progress WHERE user_id = ? AND course_id = ?", (user_id, course_id)
    ).fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    d["quiz_scores"] = json.loads(d.get("quiz_scores", "[]"))
    d["completion_percentage"] = round(d["topics_completed"] / max(d["total_topics"], 1) * 100, 1)
    return d


def update_course_progress(
    user_id: str,
    course_id: str,
    topics_completed: int | None = None,
    total_topics: int | None = None,
    quiz_score: float | None = None,
    time_spent_minutes: int | None = None,
) -> Dict[str, Any]:
    conn = _get_conn()
    existing = conn.execute(
        "SELECT * FROM course_progress WHERE user_id = ? AND course_id = ?", (user_id, course_id)
    ).fetchone()

    if existing:
        tc = topics_completed if topics_completed is not None else existing["topics_completed"]
        tt = total_topics if total_topics is not None else existing["total_topics"]
        scores = json.loads(existing["quiz_scores"] or "[]")
        if quiz_score is not None:
            scores.append(quiz_score)
        tsm = (time_spent_minutes or 0) + existing["time_spent_minutes"]
        conn.execute(
            "UPDATE course_progress SET topics_completed=?, total_topics=?, quiz_scores=?, "
            "time_spent_minutes=?, last_active=CURRENT_TIMESTAMP WHERE id=?",
            (tc, tt, json.dumps(scores), tsm, existing["id"]),
        )
    else:
        scores = [quiz_score] if quiz_score is not None else []
        conn.execute(
            "INSERT INTO course_progress (user_id, course_id, topics_completed, total_topics, quiz_scores, time_spent_minutes) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, course_id, topics_completed or 0, total_topics or 0, json.dumps(scores), time_spent_minutes or 0),
        )
    conn.commit()
    conn.close()
    return {"user_id": user_id, "course_id": course_id, "status": "updated"}


def get_summary(user_id: str) -> Dict[str, Any]:
    """Get an overall progress summary for the user."""
    all_progress = get_all_progress(user_id)
    total_courses = len(all_progress)
    completed_courses = sum(1 for p in all_progress if p["completion_percentage"] >= 100)
    total_time = sum(p["time_spent_minutes"] for p in all_progress)
    all_scores = []
    for p in all_progress:
        all_scores.extend(p.get("quiz_scores", []))
    avg_score = round(sum(all_scores) / max(len(all_scores), 1), 2)
    return {
        "total_courses": total_courses,
        "completed_courses": completed_courses,
        "total_time_minutes": total_time,
        "average_quiz_score": avg_score,
        "courses": all_progress,
    }
