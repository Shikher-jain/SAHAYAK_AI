"""Knowledge graph service — entity extraction and relationship mapping."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

_BASE_DIR = Path(__file__).resolve().parents[2]
_DB_DIR = _BASE_DIR / "data" / "knowledge"
_DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = _DB_DIR / "knowledge_graph.db"


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_knowledge_db() -> None:
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            entity_type TEXT DEFAULT 'concept',
            description TEXT DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS relationships (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            target TEXT NOT NULL,
            relation TEXT DEFAULT 'related_to',
            weight REAL DEFAULT 1.0,
            FOREIGN KEY (source) REFERENCES entities(name),
            FOREIGN KEY (target) REFERENCES entities(name)
        );
    """)
    conn.commit()
    conn.close()


init_knowledge_db()


def add_entity(name: str, entity_type: str = "concept", description: str = "") -> int:
    conn = _get_conn()
    try:
        cur = conn.execute(
            "INSERT OR IGNORE INTO entities (name, entity_type, description) VALUES (?, ?, ?)",
            (name, entity_type, description),
        )
        conn.commit()
        entity_id = cur.lastrowid
    except Exception:
        entity_id = 0
    conn.close()
    return entity_id


def add_relationship(source: str, target: str, relation: str = "related_to", weight: float = 1.0) -> int:
    add_entity(source)
    add_entity(target)
    conn = _get_conn()
    cur = conn.execute(
        "INSERT INTO relationships (source, target, relation, weight) VALUES (?, ?, ?, ?)",
        (source, target, relation, weight),
    )
    conn.commit()
    rel_id = cur.lastrowid
    conn.close()
    return rel_id


def get_graph(limit: int = 200) -> Dict[str, Any]:
    """Return nodes and edges for visualization."""
    conn = _get_conn()
    entities = conn.execute("SELECT name, entity_type, description FROM entities LIMIT ?", (limit,)).fetchall()
    relationships = conn.execute("SELECT source, target, relation, weight FROM relationships LIMIT ?", (limit * 3,)).fetchall()
    conn.close()
    return {
        "nodes": [{"name": e["name"], "type": e["entity_type"], "description": e["description"]} for e in entities],
        "edges": [{"source": r["source"], "target": r["target"], "relation": r["relation"], "weight": r["weight"]} for r in relationships],
    }


def get_entity(name: str) -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    entity = conn.execute("SELECT * FROM entities WHERE name = ?", (name,)).fetchone()
    if not entity:
        conn.close()
        return None
    rels = conn.execute(
        "SELECT * FROM relationships WHERE source = ? OR target = ?", (name, name)
    ).fetchall()
    conn.close()
    return {
        "name": entity["name"],
        "type": entity["entity_type"],
        "description": entity["description"],
        "relationships": [dict(r) for r in rels],
    }


def query_path(source: str, target: str) -> List[str]:
    """Simple BFS path finding between two entities."""
    conn = _get_conn()
    edges = conn.execute("SELECT source, target FROM relationships").fetchall()
    conn.close()
    adj: Dict[str, List[str]] = {}
    for e in edges:
        adj.setdefault(e["source"], []).append(e["target"])
        adj.setdefault(e["target"], []).append(e["source"])
    visited = set()
    queue = [(source, [source])]
    while queue:
        current, path = queue.pop(0)
        if current == target:
            return path
        if current in visited:
            continue
        visited.add(current)
        for neighbor in adj.get(current, []):
            if neighbor not in visited:
                queue.append((neighbor, path + [neighbor]))
    return []


def extract_from_text(text: str) -> Dict[str, Any]:
    """Extract entities and relationships from text.

    Tries, in order: local NER model (only if ENABLE_LOCAL_ML_MODELS=true) ->
    HF Inference API (task-specific model, no local RAM cost) -> Groq
    (general LLM, no local RAM cost) -> crude capitalized-word heuristic
    (always works, lowest quality, zero dependencies).
    """
    try:
        from backend.common.hf_models import HFModels
        ner = HFModels.get_ner()
        if ner:
            entities_raw = ner(text[:1000])
            entities_found = list(set(e["word"] for e in entities_raw if e["entity"].startswith("B-")))
            for entity in entities_found:
                add_entity(entity, entity_type="named_entity")
            for i in range(len(entities_found)):
                for j in range(i + 1, min(i + 3, len(entities_found))):
                    add_relationship(entities_found[i], entities_found[j], "co_occurs")
            return {"entities": entities_found, "relationships": len(entities_found) * (len(entities_found) - 1) // 2}
    except Exception:
        pass

    # Local NER disabled or failed — try HF Inference API (task-specific model, no local RAM cost).
    try:
        from backend.common.hf_inference_api import hf_api_ner
        entities_found = hf_api_ner(text)
        if entities_found:
            for entity in entities_found:
                add_entity(entity, entity_type="named_entity")
            for i in range(len(entities_found)):
                for j in range(i + 1, min(i + 3, len(entities_found))):
                    add_relationship(entities_found[i], entities_found[j], "co_occurs")
            return {"entities": entities_found, "relationships": len(entities_found) * (len(entities_found) - 1) // 2}
    except Exception:
        pass

    # HF API unavailable/failed — try Groq (general LLM, no local RAM cost).
    try:
        import json
        from backend.common.groq_client import groq_complete

        raw = groq_complete(
            system_prompt=(
                'Extract named entities (people, organizations, places, key concepts) from the text. '
                'Respond ONLY with valid JSON: {"entities": ["entity1", "entity2", ...]}. Max 20 entities.'
            ),
            user_prompt=text[:2000],
            temperature=0.1,
            max_tokens=300,
        )
        if raw:
            parsed = json.loads(raw)
            entities_found = [e.strip() for e in parsed.get("entities", []) if e.strip()][:20]
            if entities_found:
                for entity in entities_found:
                    add_entity(entity, entity_type="named_entity")
                for i in range(len(entities_found)):
                    for j in range(i + 1, min(i + 3, len(entities_found))):
                        add_relationship(entities_found[i], entities_found[j], "co_occurs")
                return {"entities": entities_found, "relationships": len(entities_found) * (len(entities_found) - 1) // 2}
    except Exception:
        pass

    # Final fallback: simple noun extraction (always available, no dependencies).
    words = [w.strip(".,;:!?") for w in text.split() if len(w) > 4 and w[0].isupper()]
    unique_words = list(set(words))[:20]
    for word in unique_words:
        add_entity(word, entity_type="concept")
    return {"entities": unique_words, "relationships": 0}