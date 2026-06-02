"""Cloud sync service — export/import data, Qdrant backup."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.local_stack import db as local_db
from backend.vector_store import qdrant_store

_BASE_DIR = Path(__file__).resolve().parents[2]
_EXPORT_DIR = _BASE_DIR / "data" / "exports"
_EXPORT_DIR.mkdir(parents=True, exist_ok=True)


def export_data(user_id: str = "default") -> Dict[str, Any]:
    """Export all local data to a JSON file."""
    texts, embeddings, metadatas = local_db.get_all_records()
    export_payload = {
        "export_date": datetime.utcnow().isoformat(),
        "user_id": user_id,
        "version": "1.0",
        "records": [
            {"text": text, "metadata": meta}
            for text, meta in zip(texts, metadatas)
        ],
        "total_records": len(texts),
    }
    export_path = _EXPORT_DIR / f"sahayak_export_{user_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with open(export_path, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, ensure_ascii=False, indent=2)
    return {
        "export_path": str(export_path),
        "total_records": len(texts),
        "status": "success",
    }


def import_data(file_path: str, target: str = "auto") -> Dict[str, Any]:
    """Import data from a JSON export file."""
    path = Path(file_path)
    if not path.exists():
        return {"error": f"File not found: {file_path}", "status": "failed"}
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    records = payload.get("records", [])
    if not records:
        return {"imported": 0, "status": "empty"}

    from backend.services import vector_service
    imported = 0
    for record in records:
        text = record.get("text", "")
        metadata = record.get("metadata", {})
        if text.strip():
            try:
                vector_service.ingest_text(text, metadata=metadata, target=target)
                imported += 1
            except Exception:
                pass
    return {"imported": imported, "total": len(records), "status": "success"}


def get_status() -> Dict[str, Any]:
    """Return sync status information."""
    exports = list(_EXPORT_DIR.glob("sahayak_export_*.json"))
    return {
        "total_exports": len(exports),
        "latest_export": str(max(exports)) if exports else None,
        "qdrant_available": qdrant_store.is_available,
        "local_records": local_db.build_faiss_index_with_metadata()[1].__len__() if True else 0,
    }


def list_exports() -> List[Dict[str, str]]:
    """List all available export files."""
    exports = sorted(_EXPORT_DIR.glob("sahayak_export_*.json"), reverse=True)
    return [
        {"filename": e.name, "path": str(e), "size_kb": round(e.stat().st_size / 1024, 1)}
        for e in exports[:20]
    ]
