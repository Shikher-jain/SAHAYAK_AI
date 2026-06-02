from __future__ import annotations

from pathlib import Path

from backend.ingestion.image import ocr_image


def process_image(file_path: str | Path, metadata=None) -> str:
    return ocr_image(file_path)
