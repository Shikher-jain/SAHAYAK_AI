from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from backend.ingestion.audio import transcribe_audio
from backend.ingestion.image import ocr_image_bytes
from backend.ingestion.pdf import extract_pdf_text_from_bytes
from backend.ingestion.text import ingest_text as normalize_text
from backend.ingestion.url import chunk_url
from backend.ingestion.video import transcribe_video
from backend.services import vector_service
from backend.utils.file_utils import get_tmp_path, write_bytes

router = APIRouter(tags=["multimodal-ingestion"])


async def _persist_upload(file: UploadFile) -> str:
    payload = await file.read()
    tmp_path = get_tmp_path(file.filename or "upload.bin")
    write_bytes(tmp_path, payload)
    return str(tmp_path), payload


@router.post("/audio")
async def ingest_audio_endpoint(file: UploadFile = File(...), target: str = "auto"):
    temp_path, _ = await _persist_upload(file)
    transcript = transcribe_audio(temp_path)
    metadata = {"source": file.filename, "modality": "audio"}
    records = vector_service.ingest_text(transcript, metadata=metadata, target=target)
    return {"transcription": transcript, "records": records}


@router.post("/video")
async def ingest_video_endpoint(file: UploadFile = File(...), target: str = "auto"):
    temp_path, _ = await _persist_upload(file)
    transcript = transcribe_video(temp_path)
    metadata = {"source": file.filename, "modality": "video"}
    records = vector_service.ingest_text(transcript, metadata=metadata, target=target)
    return {"transcription": transcript, "records": records}


@router.post("/image")
async def ingest_image_endpoint(file: UploadFile = File(...), target: str = "auto"):
    _, payload = await _persist_upload(file)
    text = ocr_image_bytes(payload)
    metadata = {"source": file.filename, "modality": "image"}
    records = vector_service.ingest_text(text, metadata=metadata, target=target)
    return {"ocr_text": text, "records": records}


@router.post("/pdf")
async def ingest_pdf_endpoint(file: UploadFile = File(...), target: str = "auto"):
    _, payload = await _persist_upload(file)
    text = extract_pdf_text_from_bytes(payload)
    if not text.strip():
        raise HTTPException(status_code=400, detail="No text extracted from PDF")
    metadata = {"source": file.filename, "modality": "pdf"}
    records = vector_service.ingest_text(text, metadata=metadata, target=target)
    return {"text_length": len(text), "records": records}


@router.post("/text")
async def ingest_text_endpoint(text: str = Form(...), target: str = "auto"):
    metadata = {"source": "manual", "modality": "text"}
    records = vector_service.ingest_text(normalize_text(text), metadata=metadata, target=target)
    return {"records": records}


@router.post("/url")
async def ingest_url_endpoint(url: str = Form(...), target: str = "auto"):
    chunks = chunk_url(url)
    metadata = {"source": url, "modality": "url"}
    ingested = []
    for chunk in chunks:
        ingested.extend(vector_service.ingest_text(chunk, metadata=metadata, target=target))
    return {"chunks": len(chunks), "records": ingested}
