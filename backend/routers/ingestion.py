from typing import List, Optional

from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, Request

from backend.auth import api_key_auth
from backend.auth_system.auth_service import get_current_user
from backend.auth_system.models import User
from backend.ingestion.audio import transcribe_audio
from backend.ingestion.image import ocr_image_bytes
from backend.ingestion.pdf import extract_pdf_text_from_bytes
from backend.ingestion.text import ingest_text as normalize_text
from backend.ingestion.url import fetch_url_text
from backend.ingestion.video import transcribe_video
from backend.services import audio_service, code_service, csv_service, vector_service
from backend.utils.file_utils import create_named_temp_from_bytes, safe_unlink

router = APIRouter(tags=["multimodal-ingestion"], dependencies=[Depends(api_key_auth)])

from dotenv import load_dotenv
load_dotenv()

# Extension -> modality, used by the batch endpoint to auto-route each file
# to the right extractor. Kept as a simple lookup rather than magic-byte
# sniffing — good enough for user uploads where the extension is reliable.
_EXTENSION_MODALITY_MAP = {
    ".pdf": "pdf",
    ".mp3": "audio", ".wav": "audio", ".m4a": "audio", ".ogg": "audio", ".flac": "audio",
    ".mp4": "video", ".mov": "video", ".avi": "video", ".mkv": "video", ".webm": "video",
    ".png": "image", ".jpg": "image", ".jpeg": "image", ".gif": "image", ".bmp": "image", ".webp": "image",
    ".csv": "csv", ".xlsx": "csv", ".xls": "csv",
    ".py": "code", ".js": "code", ".ts": "code", ".java": "code", ".cpp": "code", ".c": "code",
    ".go": "code", ".rs": "code", ".rb": "code", ".php": "code", ".cs": "code", ".swift": "code",
    ".txt": "text", ".md": "text",
}


async def _persist_upload(file: UploadFile) -> tuple[str, bytes]:
    payload = await file.read()
    tmp_path = create_named_temp_from_bytes(payload, original_name=file.filename or "upload.bin")
    return str(tmp_path), payload

from fastapi import BackgroundTasks
import uuid
from backend.workers.tasks import run_ingestion_job
from backend.workers.ingestion import process_audio_job

@router.post("/audio")
async def ingest_audio_endpoint(background_tasks: BackgroundTasks, file: UploadFile = File(...), target: str = "auto", user: Optional[User] = Depends(get_current_user)):
    temp_path, _ = await _persist_upload(file)
    job_id = uuid.uuid4().hex
    user_id_str = str(user.id) if user else None
    background_tasks.add_task(run_ingestion_job, job_id, process_audio_job, temp_path, file.filename, target, user_id_str)
    return {"job_id": job_id, "status": "processing"}

from backend.workers.ingestion import (
    process_video_job, process_image_job, process_pdf_job,
    process_csv_job, process_code_job, process_url_job, process_youtube_job, process_text_job
)

@router.post("/video")
async def ingest_video_endpoint(background_tasks: BackgroundTasks, file: UploadFile = File(...), target: str = "auto", user: Optional[User] = Depends(get_current_user)):
    temp_path, _ = await _persist_upload(file)
    job_id = uuid.uuid4().hex
    user_id_str = str(user.id) if user else None
    background_tasks.add_task(run_ingestion_job, job_id, process_video_job, temp_path, file.filename, target, user_id_str)
    return {"job_id": job_id, "status": "processing"}

@router.post("/image")
async def ingest_image_endpoint(background_tasks: BackgroundTasks, file: UploadFile = File(...), target: str = "auto", user: Optional[User] = Depends(get_current_user)):
    tmp_path, payload = await _persist_upload(file)
    job_id = uuid.uuid4().hex
    user_id_str = str(user.id) if user else None
    background_tasks.add_task(run_ingestion_job, job_id, process_image_job, tmp_path, file.filename, target, user_id_str)
    return {"job_id": job_id, "status": "processing"}

@router.post("/pdf", dependencies=[])
async def ingest_pdf_endpoint(background_tasks: BackgroundTasks, file: UploadFile = File(...), target: str = "auto", user: Optional[User] = Depends(get_current_user)):
    tmp_path, payload = await _persist_upload(file)
    job_id = uuid.uuid4().hex
    user_id_str = str(user.id) if user else None
    background_tasks.add_task(run_ingestion_job, job_id, process_pdf_job, tmp_path, file.filename, target, user_id_str)
    return {"job_id": job_id, "status": "processing"}

from fastapi import Request
from pydantic import BaseModel, Field


class IngestTextRequest(BaseModel):
    text: str = Field(..., description="The text content to ingest")
    target: str = Field("auto", description="Target collection or 'auto'")


@router.post("/text")
async def ingest_text_endpoint(background_tasks: BackgroundTasks, payload: IngestTextRequest, user: Optional[User] = Depends(get_current_user)):
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text content is required.")
    job_id = uuid.uuid4().hex
    user_id_str = str(user.id) if user else None
    background_tasks.add_task(run_ingestion_job, job_id, process_text_job, text, payload.target.strip(), user_id_str)
    return {"job_id": job_id, "status": "processing"}


class IngestUrlRequest(BaseModel):
    url: str = Field(..., description="The URL to scrape and ingest")
    target: str = Field("auto", description="Target collection or 'auto'")


@router.post("/url")
async def ingest_url_endpoint(background_tasks: BackgroundTasks, payload: IngestUrlRequest, user: Optional[User] = Depends(get_current_user)):
    url = payload.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL is required.")
    job_id = uuid.uuid4().hex
    user_id_str = str(user.id) if user else None
    background_tasks.add_task(run_ingestion_job, job_id, process_url_job, url, payload.target.strip(), user_id_str)
    return {"job_id": job_id, "status": "processing"}


class IngestYoutubeRequest(BaseModel):
    url: str = Field(..., description="The YouTube URL to ingest")
    target: str = Field("auto", description="Target collection or 'auto'")


@router.post("/youtube")
async def ingest_youtube_endpoint(background_tasks: BackgroundTasks, payload: IngestYoutubeRequest, user: Optional[User] = Depends(get_current_user)):
    url = payload.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="YouTube URL is required.")
    
    from backend.ingestion.youtube import extract_video_id
    if not extract_video_id(url):
        raise HTTPException(status_code=400, detail="Not a recognizable YouTube URL.")
        
    job_id = uuid.uuid4().hex
    user_id_str = str(user.id) if user else None
    background_tasks.add_task(run_ingestion_job, job_id, process_youtube_job, url, payload.target.strip(), user_id_str)
    return {"job_id": job_id, "status": "processing"}

@router.post("/code")
async def ingest_code_endpoint(background_tasks: BackgroundTasks, file: UploadFile = File(...), target: str = "auto", user: Optional[User] = Depends(get_current_user)):
    temp_path, _ = await _persist_upload(file)
    job_id = uuid.uuid4().hex
    user_id_str = str(user.id) if user else None
    background_tasks.add_task(run_ingestion_job, job_id, process_code_job, temp_path, file.filename, target, user_id_str)
    return {"job_id": job_id, "status": "processing"}


@router.post("/csv")
async def ingest_csv_endpoint(background_tasks: BackgroundTasks, file: UploadFile = File(...), target: str = "auto", user: Optional[User] = Depends(get_current_user)):
    temp_path, _ = await _persist_upload(file)
    job_id = uuid.uuid4().hex
    user_id_str = str(user.id) if user else None
    background_tasks.add_task(run_ingestion_job, job_id, process_csv_job, temp_path, file.filename, target, user_id_str)
    return {"job_id": job_id, "status": "processing"}


def _detect_modality(filename: str) -> str:
    import os as _os
    ext = _os.path.splitext(filename or "")[1].lower()
    return _EXTENSION_MODALITY_MAP.get(ext, "unknown")


async def _ingest_single_file(file: UploadFile, target: str, user: Optional[User] = None) -> dict:
    """Process one file end-to-end, routed by detected modality. Returns a
    per-file result dict — never raises, so one bad file in a batch doesn't
    take down the rest."""
    modality = _detect_modality(file.filename)
    if modality == "unknown":
        return {"filename": file.filename, "status": "skipped", "error": "Unrecognized file type"}

    tmp_path = None
    try:
        tmp_path, payload = await _persist_upload(file)
        metadata = {"source": file.filename, "modality": modality}
        if user:
            metadata["user_id"] = str(user.id)

        if modality == "pdf":
            text = extract_pdf_text_from_bytes(payload)
            if not text.strip():
                return {"filename": file.filename, "status": "error", "error": "No text extracted from PDF"}
            records = vector_service.ingest_text(text, metadata=metadata, target=target)
        elif modality == "audio":
            transcript = transcribe_audio(tmp_path)
            metadata["timeline"] = audio_service.build_audio_timeline(transcript)
            records = vector_service.ingest_text(transcript, metadata=metadata, target=target)
        elif modality == "video":
            transcript = transcribe_video(tmp_path)
            metadata["timeline"] = audio_service.build_audio_timeline(transcript)
            records = vector_service.ingest_text(transcript, metadata=metadata, target=target)
        elif modality == "image":
            text = ocr_image_bytes(payload)
            records = vector_service.ingest_text(text, metadata=metadata, target=target)
        elif modality == "csv":
            records = csv_service.process_csv(tmp_path, metadata=metadata, target=target)
        elif modality == "code":
            records = code_service.process_code(tmp_path, metadata=metadata, target=target)
        elif modality == "text":
            text = payload.decode("utf-8", errors="ignore")
            records = vector_service.ingest_text(normalize_text(text), metadata=metadata, target=target)
        else:
            return {"filename": file.filename, "status": "skipped", "error": "Unrecognized file type"}

        return {"filename": file.filename, "status": "ok", "modality": modality, "chunks": len(records)}
    except Exception as exc:
        return {"filename": file.filename, "status": "error", "error": str(exc)}
    finally:
        if tmp_path:
            safe_unlink(tmp_path)


@router.post("/batch")
async def ingest_batch_endpoint(files: List[UploadFile] = File(...), target: str = "auto", user: Optional[User] = Depends(get_current_user)):
    """Upload multiple files — of different types — in a single request.
    Each file is auto-routed to the right extractor by its extension. A
    failure on one file doesn't abort the rest; check each result's
    "status" field ("ok" | "error" | "skipped")."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 files per batch upload.")

    results = [await _ingest_single_file(f, target, user=user) for f in files]
    succeeded = sum(1 for r in results if r["status"] == "ok")
    return {
        "total": len(results),
        "succeeded": succeeded,
        "failed": len(results) - succeeded,
        "results": results,
    }