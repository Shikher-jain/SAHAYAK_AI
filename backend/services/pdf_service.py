from __future__ import annotations

from pathlib import Path

from backend.ingestion.pdf import extract_pdf_text


def process_pdf(file_path: str | Path, metadata=None) -> str:
    return extract_pdf_text(file_path)
