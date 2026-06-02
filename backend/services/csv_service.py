from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from backend.ingestion.csv_excel import SUPPORTED_EXTENSIONS, extract_table_chunks
from backend.services import vector_service


def process_csv(
    file_path: str | Path,
    metadata: Dict[str, Any] | None = None,
    target: str = "auto",
) -> List[Dict[str, str]]:
    path = Path(file_path)
    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported tabular extension: {extension}")

    chunks = extract_table_chunks(path)
    records: List[Dict[str, str]] = []
    base_metadata = metadata or {}
    for chunk in chunks:
        chunk_metadata = dict(base_metadata)
        chunk_metadata.update(chunk.get("metadata", {}))
        chunk_metadata.setdefault("source", path.name)
        chunk_metadata.setdefault("modality", "csv")
        records.extend(
            vector_service.ingest_text(
                chunk.get("text", ""),
                metadata=chunk_metadata,
                target=target,
                chunking_strategy="fixed",
            )
        )
    return records
