from typing import Dict

from fastapi import APIRouter, Depends, File, UploadFile

from backend.auth import api_key_auth
from backend.common.data_paths import get_local_pdf_storage_dir
from backend.ingestion.text import chunk_text
from backend.local_stack import extractor, rag_engine
from backend.local_stack.db import add_chunk_with_metadata, init_db
from backend.common.embedder import embed_text

PDF_FOLDER = get_local_pdf_storage_dir()

init_db()
router = APIRouter(prefix="/local", tags=["local-rag"], dependencies=[Depends(api_key_auth)])


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

    chunks = chunk_text(text, chunk_size=180, overlap=30)
    if not chunks:
        chunks = [text]
    for chunk in chunks:
        embedding = embed_text(chunk)
        # BUG 2 FIX: persist filename metadata for local-only uploads.
        add_chunk_with_metadata(filename, chunk, embedding, {"source": filename, "modality": "local"})

    return {"status": "ok", "chunks_written": str(len(chunks))}


@router.get("/ask")
def local_ask(question: str):
    return {"answer": rag_engine.answer_question(question)}
