from typing import List


def ingest_text(text: str) -> str:
    return text.strip()


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 120) -> List[str]:
    cleaned = text.strip().split()
    chunks = []
    if not cleaned:
        return chunks
    step = max(1, chunk_size - overlap)
    for start in range(0, len(cleaned), step):
        chunk = " ".join(cleaned[start : start + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks
