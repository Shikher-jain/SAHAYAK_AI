from typing import List

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


@router.post("/audio")
async def ingest_audio_endpoint(file: UploadFile = File(...), target: str = "auto"):
    temp_path, _ = await _persist_upload(file)
    try:
        transcript = transcribe_audio(temp_path)
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


def _detect_modality(filename: str) -> str:
    import os as _os
    ext = _os.path.splitext(filename or "")[1].lower()
    return _EXTENSION_MODALITY_MAP.get(ext, "unknown")


async def _ingest_single_file(file: UploadFile, target: str) -> dict:
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
async def ingest_batch_endpoint(files: List[UploadFile] = File(...), target: str = "auto"):
    """Upload multiple files — of different types — in a single request.
    Each file is auto-routed to the right extractor by its extension. A
    failure on one file doesn't abort the rest; check each result's
    "status" field ("ok" | "error" | "skipped")."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 files per batch upload.")

    results = [await _ingest_single_file(f, target) for f in files]
    succeeded = sum(1 for r in results if r["status"] == "ok")
    return {
        "total": len(results),
        "succeeded": succeeded,
        "failed": len(results) - succeeded,
        "results": results,
    }