"""Learning mode service — student, teacher, and self-learning paths."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

_BASE_DIR = Path(__file__).resolve().parents[2]
_DB_DIR = _BASE_DIR / "data" / "learning"
_DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = _DB_DIR / "learning.db"


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_learning_db() -> None:
    """Create learning tables if they do not exist."""
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS learning_mode (
            user_id TEXT PRIMARY KEY,
            mode TEXT NOT NULL DEFAULT 'student',
            preferences TEXT DEFAULT '{}',
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS bookmarks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            document_id TEXT NOT NULL,
            title TEXT DEFAULT '',
            note TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS learning_notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            title TEXT NOT NULL,
            content TEXT DEFAULT '',
            tags TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    conn.close()


init_learning_db()


# --- Learning Mode ---

def set_mode(user_id: str, mode: str, preferences: Dict[str, Any] | None = None) -> Dict[str, Any]:
    conn = _get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO learning_mode (user_id, mode, preferences, updated_at) "
        "VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
        (user_id, mode, json.dumps(preferences or {})),
    )
    conn.commit()
    conn.close()
    return {"user_id": user_id, "mode": mode, "preferences": preferences or {}}


def get_mode(user_id: str) -> Dict[str, Any]:
    conn = _get_conn()
    row = conn.execute("SELECT * FROM learning_mode WHERE user_id = ?", (user_id,)).fetchone()
    conn.close()
    if not row:
        return {"user_id": user_id, "mode": "student", "preferences": {}}
    return {"user_id": row["user_id"], "mode": row["mode"], "preferences": json.loads(row["preferences"] or "{}")}


# --- Bookmarks ---

def add_bookmark(user_id: str, document_id: str, title: str = "", note: str = "") -> int:
    conn = _get_conn()
    cur = conn.execute(
        "INSERT INTO bookmarks (user_id, document_id, title, note) VALUES (?, ?, ?, ?)",
        (user_id, document_id, title, note),
    )
    bookmark_id = cur.lastrowid
    conn.commit()
    conn.close()
    return bookmark_id


def get_bookmarks(user_id: str) -> List[Dict[str, Any]]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, document_id, title, note, created_at FROM bookmarks WHERE user_id = ? ORDER BY created_at DESC",
        (user_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_bookmark(bookmark_id: int, user_id: str) -> bool:
    conn = _get_conn()
    cur = conn.execute("DELETE FROM bookmarks WHERE id = ? AND user_id = ?", (bookmark_id, user_id))
    conn.commit()
    deleted = cur.rowcount > 0
    conn.close()
    return deleted


# --- Learning Notes ---

def add_note(user_id: str, title: str, content: str = "", tags: str = "") -> int:
    conn = _get_conn()
    cur = conn.execute(
        "INSERT INTO learning_notes (user_id, title, content, tags) VALUES (?, ?, ?, ?)",
        (user_id, title, content, tags),
    )
    note_id = cur.lastrowid
    conn.commit()
    conn.close()
    return note_id


def get_notes(user_id: str) -> List[Dict[str, Any]]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, title, content, tags, created_at FROM learning_notes WHERE user_id = ? ORDER BY created_at DESC",
        (user_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
