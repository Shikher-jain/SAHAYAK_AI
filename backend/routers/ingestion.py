from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException

from backend.auth import api_key_auth
from backend.ingestion.audio import transcribe_audio
from backend.ingestion.image import ocr_image_bytes
from backend.ingestion.pdf import extract_pdf_text_from_bytes
from backend.ingestion.text import ingest_text as normalize_text
from backend.ingestion.url import fetch_url_text
from backend.ingestion.video import transcribe_video
from backend.services import audio_service, code_service, csv_service, vector_service
from backend.utils.file_utils import create_named_temp_from_bytes, safe_unlink

router = APIRouter(tags=["multimodal-ingestion"], dependencies=[Depends(api_key_auth)])


async def _persist_upload(file: UploadFile) -> tuple[str, bytes]:
    payload = await file.read()
    tmp_path = create_named_temp_from_bytes(payload, original_name=file.filename or "upload.bin")
    return str(tmp_path), payload


@router.post("/audio")
async def ingest_audio_endpoint(file: UploadFile = File(...), target: str = "auto"):
    temp_path, _ = await _persist_upload(file)
    try:
        transcript = transcribe_audio(temp_path)
        # TASK 6 FIX: attach generated timeline metadata for audio ingestion.
        timeline = audio_service.build_audio_timeline(transcript)
        metadata = {"source": file.filename, "modality": "audio", "timeline": timeline}
        records = vector_service.ingest_text(transcript, metadata=metadata, target=target)
        return {"transcription": transcript, "records": records}
    finally:
        safe_unlink(temp_path)


@router.post("/video")
async def ingest_video_endpoint(file: UploadFile = File(...), target: str = "auto"):
    temp_path, _ = await _persist_upload(file)
    try:
        transcript = transcribe_video(temp_path)
        # TASK 6 FIX: attach generated timeline metadata for video ingestion.
        timeline = audio_service.build_audio_timeline(transcript)
        metadata = {"source": file.filename, "modality": "video", "timeline": timeline}
        records = vector_service.ingest_text(transcript, metadata=metadata, target=target)
        return {"transcription": transcript, "records": records}
    finally:
        safe_unlink(temp_path)


@router.post("/image")
async def ingest_image_endpoint(file: UploadFile = File(...), target: str = "auto"):
    tmp_path, payload = await _persist_upload(file)
    try:
        text = ocr_image_bytes(payload)
        metadata = {"source": file.filename, "modality": "image"}
        records = vector_service.ingest_text(text, metadata=metadata, target=target)
        return {"ocr_text": text, "records": records}
    finally:
        safe_unlink(tmp_path)


@router.post("/pdf", dependencies=[])
async def ingest_pdf_endpoint(file: UploadFile = File(...), target: str = "auto"):
    tmp_path, payload = await _persist_upload(file)
    try:
        text = extract_pdf_text_from_bytes(payload)
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text extracted from PDF")
        metadata = {"source": file.filename, "modality": "pdf"}
        records = vector_service.ingest_text(text, metadata=metadata, target=target)
        return {"text_length": len(text), "records": records}
    finally:
        safe_unlink(tmp_path)


@router.post("/text")
async def ingest_text_endpoint(text: str = Form(...), target: str = "auto"):
    metadata = {"source": "manual", "modality": "text"}
    records = vector_service.ingest_text(normalize_text(text), metadata=metadata, target=target)
    return {"records": records}


@router.post("/url")
async def ingest_url_endpoint(url: str = Form(...), target: str = "auto"):
    try:
        text = fetch_url_text(url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    metadata = {"source": url, "modality": "url"}
    records = vector_service.ingest_text(text, metadata=metadata, target=target)
    return {"chunks": len(records), "records": records}


@router.post("/code")
async def ingest_code_endpoint(file: UploadFile = File(...), target: str = "auto"):
    temp_path, _ = await _persist_upload(file)
    try:
        metadata = {"source": file.filename, "modality": "code"}
        records = code_service.process_code(temp_path, metadata=metadata, target=target)
        return {"chunks": len(records), "records": records}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        safe_unlink(temp_path)


@router.post("/csv")
async def ingest_csv_endpoint(file: UploadFile = File(...), target: str = "auto"):
    temp_path, _ = await _persist_upload(file)
    try:
        metadata = {"source": file.filename, "modality": "csv"}
        records = csv_service.process_csv(temp_path, metadata=metadata, target=target)
        return {"chunks": len(records), "records": records}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        safe_unlink(temp_path)
