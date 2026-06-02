"""Roadmap service — learning roadmaps with progress tracking."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

_BASE_DIR = Path(__file__).resolve().parents[2]
_DB_DIR = _BASE_DIR / "data" / "roadmaps"
_DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = _DB_DIR / "roadmaps.db"


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_roadmap_db() -> None:
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS roadmap_progress (
            user_id TEXT NOT NULL,
            roadmap_id TEXT NOT NULL,
            topic_id TEXT NOT NULL,
            completed INTEGER DEFAULT 0,
            completed_at TIMESTAMP,
            PRIMARY KEY (user_id, roadmap_id, topic_id)
        );
    """)
    conn.commit()
    conn.close()


init_roadmap_db()

# Pre-built roadmaps
_ROADMAPS: Dict[str, Dict[str, Any]] = {
    "data-science": {
        "id": "data-science",
        "title": "Data Science Roadmap",
        "description": "Complete path from beginner to data scientist",
        "url": "https://roadmap.sh/ai-data-scientist",
        "topics": [
            {"id": "ds-1", "title": "Python Fundamentals", "prerequisites": [], "resources": ["Python for Everybody (Coursera)"]},
            {"id": "ds-2", "title": "Statistics & Probability", "prerequisites": ["ds-1"], "resources": ["Khan Academy Statistics"]},
            {"id": "ds-3", "title": "Linear Algebra", "prerequisites": ["ds-1"], "resources": ["3Blue1Brown Essence of Linear Algebra"]},
            {"id": "ds-4", "title": "Data Wrangling (Pandas)", "prerequisites": ["ds-1", "ds-2"], "resources": ["Pandas documentation, Kaggle"]},
            {"id": "ds-5", "title": "Data Visualization", "prerequisites": ["ds-4"], "resources": ["Matplotlib, Seaborn, Plotly"]},
            {"id": "ds-6", "title": "Machine Learning Basics", "prerequisites": ["ds-2", "ds-3", "ds-4"], "resources": ["Andrew Ng ML Course"]},
            {"id": "ds-7", "title": "Feature Engineering", "prerequisites": ["ds-6"], "resources": ["Kaggle competitions"]},
            {"id": "ds-8", "title": "Deep Learning", "prerequisites": ["ds-6"], "resources": ["Deep Learning Specialization"]},
            {"id": "ds-9", "title": "NLP & Computer Vision", "prerequisites": ["ds-8"], "resources": ["HuggingFace courses"]},
            {"id": "ds-10", "title": "MLOps & Deployment", "prerequisites": ["ds-6"], "resources": ["MLflow, Docker, FastAPI"]},
        ],
    },
    "web-dev": {
        "id": "web-dev",
        "title": "Web Development Roadmap",
        "description": "Full-stack web development from scratch",
        "url": "https://roadmap.sh/full-stack",
        "topics": [
            {"id": "web-1", "title": "HTML & CSS", "prerequisites": [], "resources": ["freeCodeCamp, MDN"]},
            {"id": "web-2", "title": "JavaScript Fundamentals", "prerequisites": ["web-1"], "resources": ["JavaScript.info"]},
            {"id": "web-3", "title": "Git & GitHub", "prerequisites": [], "resources": ["Pro Git book"]},
            {"id": "web-4", "title": "React / Next.js", "prerequisites": ["web-2"], "resources": ["React docs, Next.js docs"]},
            {"id": "web-5", "title": "Node.js & Express", "prerequisites": ["web-2"], "resources": ["Node.js docs"]},
            {"id": "web-6", "title": "Databases (SQL + NoSQL)", "prerequisites": ["web-5"], "resources": ["PostgreSQL, MongoDB docs"]},
            {"id": "web-7", "title": "REST APIs & GraphQL", "prerequisites": ["web-5", "web-6"], "resources": ["FastAPI, Express"]},
            {"id": "web-8", "title": "Authentication & Security", "prerequisites": ["web-7"], "resources": ["OWASP, JWT.io"]},
            {"id": "web-9", "title": "DevOps & Deployment", "prerequisites": ["web-7"], "resources": ["Docker, AWS, Vercel"]},
        ],
    },
    "ml-ai": {
        "id": "ml-ai",
        "title": "Machine Learning & AI Roadmap",
        "description": "From math foundations to advanced AI systems",
        "url": "https://roadmap.sh/ai-engineer",
        "topics": [
            {"id": "ml-1", "title": "Math Foundations", "prerequisites": [], "resources": ["Khan Academy, 3Blue1Brown"]},
            {"id": "ml-2", "title": "Python for ML", "prerequisites": ["ml-1"], "resources": ["NumPy, Pandas, Scikit-learn"]},
            {"id": "ml-3", "title": "Classical ML Algorithms", "prerequisites": ["ml-2"], "resources": ["Andrew Ng course"]},
            {"id": "ml-4", "title": "Neural Networks", "prerequisites": ["ml-3"], "resources": ["Deep Learning book"]},
            {"id": "ml-5", "title": "CNNs & Computer Vision", "prerequisites": ["ml-4"], "resources": ["CS231n"]},
            {"id": "ml-6", "title": "RNNs & NLP", "prerequisites": ["ml-4"], "resources": ["CS224n, HuggingFace"]},
            {"id": "ml-7", "title": "Transformers & LLMs", "prerequisites": ["ml-6"], "resources": ["Attention Is All You Need paper"]},
            {"id": "ml-8", "title": "RLHF & Fine-tuning", "prerequisites": ["ml-7"], "resources": ["InstructGPT paper, PEFT"]},
            {"id": "ml-9", "title": "RAG & AI Agents", "prerequisites": ["ml-7"], "resources": ["LangChain, LlamaIndex"]},
        ],
    },
    "dsa": {
        "id": "dsa",
        "title": "DSA (Data Structures & Algorithms) Roadmap",
        "description": "Master DSA for coding interviews",
        "url": "https://roadmap.sh/datastructures-and-algorithms",
        "topics": [
            {"id": "dsa-1", "title": "Arrays & Strings", "prerequisites": [], "resources": ["LeetCode Easy"]},
            {"id": "dsa-2", "title": "Linked Lists", "prerequisites": ["dsa-1"], "resources": ["GeeksForGeeks"]},
            {"id": "dsa-3", "title": "Stacks & Queues", "prerequisites": ["dsa-2"], "resources": ["LeetCode"]},
            {"id": "dsa-4", "title": "Trees & BST", "prerequisites": ["dsa-2"], "resources": ["LeetCode Medium"]},
            {"id": "dsa-5", "title": "Graphs & BFS/DFS", "prerequisites": ["dsa-4"], "resources": ["LeetCode Medium"]},
            {"id": "dsa-6", "title": "Dynamic Programming", "prerequisites": ["dsa-1"], "resources": ["NeetCode DP playlist"]},
            {"id": "dsa-7", "title": "Sorting & Searching", "prerequisites": ["dsa-1"], "resources": ["LeetCode"]},
            {"id": "dsa-8", "title": "Greedy & Backtracking", "prerequisites": ["dsa-6"], "resources": ["LeetCode Hard"]},
        ],
    },
}


def list_roadmaps() -> List[Dict[str, Any]]:
    return [{"id": r["id"], "title": r["title"], "description": r["description"], "url": r["url"]} for r in _ROADMAPS.values()]


def get_roadmap(roadmap_id: str) -> Optional[Dict[str, Any]]:
    return _ROADMAPS.get(roadmap_id)


def get_progress(user_id: str, roadmap_id: str) -> Dict[str, bool]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT topic_id, completed FROM roadmap_progress WHERE user_id = ? AND roadmap_id = ?",
        (user_id, roadmap_id),
    ).fetchall()
    conn.close()
    return {row["topic_id"]: bool(row["completed"]) for row in rows}


def update_progress(user_id: str, roadmap_id: str, topic_id: str, completed: bool) -> Dict[str, Any]:
    conn = _get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO roadmap_progress (user_id, roadmap_id, topic_id, completed, completed_at) "
        "VALUES (?, ?, ?, ?, CASE WHEN ? THEN CURRENT_TIMESTAMP ELSE NULL END)",
        (user_id, roadmap_id, topic_id, int(completed), int(completed)),
    )
    conn.commit()
    conn.close()
    return {"user_id": user_id, "roadmap_id": roadmap_id, "topic_id": topic_id, "completed": completed}


def get_completion_percentage(user_id: str, roadmap_id: str) -> float:
    roadmap = get_roadmap(roadmap_id)
    if not roadmap:
        return 0.0
    total = len(roadmap["topics"])
    if total == 0:
        return 0.0
    progress = get_progress(user_id, roadmap_id)
    completed = sum(1 for v in progress.values() if v)
    return round(completed / total * 100, 1)
