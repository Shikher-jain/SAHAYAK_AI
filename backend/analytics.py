from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List


class Analytics:
    def __init__(self, log_file: str | Path = "analytics_log.json") -> None:
        self.log_file = Path(log_file)
        self._lock = Lock()
        if not self.log_file.exists():
            self.log_file.write_text("[]", encoding="utf-8")

    def log_event(self, event_type: str, metadata: Dict[str, Any] | None = None) -> None:
        event = dict(metadata or {})
        event["timestamp"] = datetime.utcnow().isoformat()
        event["event_type"] = event_type

        with self._lock:
            try:
                data = json.loads(self.log_file.read_text(encoding="utf-8"))
            except Exception:
                data = []
            data.append(event)
            self.log_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def get_events(self, event_type: str | None = None) -> List[Dict[str, Any]]:
        try:
            data = json.loads(self.log_file.read_text(encoding="utf-8"))
        except Exception:
            return []
        if event_type is None:
            return data
        return [item for item in data if item.get("event_type") == event_type]


def get_ingestion_stats() -> Dict[str, int]:
    return {"total_files": 0, "total_audio": 0, "total_video": 0}
