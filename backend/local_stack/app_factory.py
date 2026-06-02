from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.ingestion.text import chunk_text


def create_local_rag_app(
    *,
    title: str,
    storage_dir: Path,
    init_db: Callable[[], None],
    add_chunk: Callable[[str, str, object, Dict[str, str] | None], None],
    embed_text: Callable[[str], object],
    extract_pdf: Callable[[bytes], str],
    extract_image: Callable[[bytes], str],
    answer_question: Callable[[str], str],
) -> FastAPI:
    storage_dir.mkdir(parents=True, exist_ok=True)
    app = FastAPI(title=title)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    init_db()

    @app.post("/upload")
    async def upload_file(file: UploadFile = File(...)):
        payload = await file.read()
        filename = Path(file.filename or "upload.bin").name
        save_path = storage_dir / filename
        save_path.write_bytes(payload)

        if filename.lower().endswith(".pdf"):
            text = extract_pdf(payload)
        elif filename.lower().endswith((".png", ".jpg", ".jpeg")):
            text = extract_image(payload)
        else:
            return {"error": "Unsupported file type. Please upload PDF or image (PNG/JPG)"}

        if not text.strip():
            return {"error": "No text could be extracted from the file"}

        chunks = chunk_text(text, chunk_size=180, overlap=30)
        if not chunks:
            chunks = [text]

        for chunk in chunks:
            emb = embed_text(chunk)
            # BUG 2 FIX: persist local metadata for accurate FAISS retrieval.
            add_chunk(filename, chunk, emb, {"source": filename, "modality": "local"})

        return {
            "message": f"{filename} uploaded successfully",
            "details": {
                "filename": filename,
                "text_length": len(text),
                "chunks_created": len(chunks),
            },
        }

    @app.get("/vector_store_health")
    def vector_store_health():
        return {"vector_store": "local", "status": "local-only"}

    @app.get("/ask")
    def ask(question: str):
        return {"answer": answer_question(question)}

    @app.get("/health")
    def health_check():
        return {"status": "healthy", "message": f"{title} is running"}

    @app.get("/")
    def root():
        return {
            "message": title,
            "endpoints": {
                "/upload": "POST - Upload PDF/Image files",
                "/ask": "GET - Ask questions about uploaded documents",
                "/health": "GET - Health check",
            },
        }

    return app
