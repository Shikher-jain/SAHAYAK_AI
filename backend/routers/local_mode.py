from pathlib import Path
from typing import Dict

from fastapi import APIRouter, File, UploadFile

from backend.local_stack import extractor, rag_engine
from backend.local_stack.db import add_chunk, init_db
from backend.local_stack.embedder import embed_text

BASE_DIR = Path(__file__).resolve().parents[2]
PDF_FOLDER = BASE_DIR / "data" / "sahayak_09_02" / "pdf_storage"
PDF_FOLDER.mkdir(parents=True, exist_ok=True)

init_db()
router = APIRouter(prefix="/local", tags=["local-rag"])


@router.post("/upload")
async def upload_to_local_store(file: UploadFile = File(...)) -> Dict[str, str]:
    payload = await file.read()
    filename = file.filename or "document"
    path = PDF_FOLDER / filename
    path.write_bytes(payload)

    if filename.lower().endswith(".pdf"):
        text = extractor.extract_pdf(payload)
    elif filename.lower().endswith((".png", ".jpg", ".jpeg")):
        text = extractor.extract_image(payload)
    else:
        return {"error": "Unsupported file type"}

    if not text.strip():
        return {"error": "Unable to extract text"}

    chunk_size = 600
    overlap = 120
    chunks = []
    for idx in range(0, len(text), chunk_size - overlap):
        chunk = text[idx : idx + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
            embedding = embed_text(chunk)
            add_chunk(filename, chunk, embedding)

    return {"status": "ok", "chunks_written": str(len(chunks))}


@router.get("/ask")
def local_ask(question: str):
    return {"answer": rag_engine.answer_question(question)}
