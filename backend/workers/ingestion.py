import logging
from typing import Optional
from backend.ingestion.audio import transcribe_audio
from backend.ingestion.image import ocr_image_bytes
from backend.ingestion.pdf import extract_pdf_text_from_bytes
from backend.ingestion.text import ingest_text as normalize_text
from backend.ingestion.url import fetch_url_text
from backend.ingestion.video import transcribe_video
from backend.services import audio_service, code_service, csv_service, vector_service
from backend.utils.file_utils import safe_unlink
from pathlib import Path

logger = logging.getLogger(__name__)

def process_audio_job(temp_path: str, filename: str, target: str, user_id: Optional[str]) -> dict:
    try:
        transcript = transcribe_audio(temp_path)
        timeline = audio_service.build_audio_timeline(transcript)
        metadata = {"source": filename, "modality": "audio", "timeline": timeline}
        if user_id:
            metadata["user_id"] = user_id
        records = vector_service.ingest_text(transcript, metadata=metadata, target=target)
        return {"transcription_length": len(transcript), "chunks": len(records)}
    finally:
        safe_unlink(temp_path)

def process_video_job(temp_path: str, filename: str, target: str, user_id: Optional[str]) -> dict:
    try:
        transcript = transcribe_video(temp_path)
        timeline = audio_service.build_audio_timeline(transcript)
        metadata = {"source": filename, "modality": "video", "timeline": timeline}
        if user_id:
            metadata["user_id"] = user_id
        records = vector_service.ingest_text(transcript, metadata=metadata, target=target)
        return {"transcription_length": len(transcript), "chunks": len(records)}
    finally:
        safe_unlink(temp_path)

def process_image_job(temp_path: str, filename: str, target: str, user_id: Optional[str]) -> dict:
    try:
        with open(temp_path, "rb") as f:
            payload = f.read()
        text = ocr_image_bytes(payload)
        metadata = {"source": filename, "modality": "image"}
        if user_id:
            metadata["user_id"] = user_id
        records = vector_service.ingest_text(text, metadata=metadata, target=target)
        return {"ocr_length": len(text), "chunks": len(records)}
    finally:
        safe_unlink(temp_path)

def process_pdf_job(temp_path: str, filename: str, target: str, user_id: Optional[str]) -> dict:
    try:
        with open(temp_path, "rb") as f:
            payload = f.read()
        text = extract_pdf_text_from_bytes(payload)
        if not text.strip():
            raise ValueError("No text extracted from PDF")
        metadata = {"source": filename, "modality": "pdf"}
        if user_id:
            metadata["user_id"] = user_id
        records = vector_service.ingest_text(text, metadata=metadata, target=target)
        return {"text_length": len(text), "chunks": len(records)}
    finally:
        safe_unlink(temp_path)

def process_csv_job(temp_path: str, filename: str, target: str, user_id: Optional[str]) -> dict:
    try:
        metadata = {"source": filename, "modality": "csv"}
        if user_id:
            metadata["user_id"] = user_id
        records = csv_service.process_csv(temp_path, metadata=metadata, target=target)
        return {"chunks": len(records)}
    finally:
        safe_unlink(temp_path)

def process_code_job(temp_path: str, filename: str, target: str, user_id: Optional[str]) -> dict:
    try:
        metadata = {"source": filename, "modality": "code"}
        if user_id:
            metadata["user_id"] = user_id
        records = code_service.process_code(temp_path, metadata=metadata, target=target)
        return {"chunks": len(records)}
    finally:
        safe_unlink(temp_path)

def process_youtube_job(url: str, target: str, user_id: Optional[str]) -> dict:
    from backend.ingestion.youtube import get_youtube_text
    text = get_youtube_text(url)
    if not text or not text.strip():
        raise ValueError("No transcribable content found for this video.")
    metadata = {"source": url, "modality": "youtube"}
    if user_id:
        metadata["user_id"] = user_id
    records = vector_service.ingest_text(text, metadata=metadata, target=target)
    return {"chunks": len(records)}

def process_url_job(url: str, target: str, user_id: Optional[str]) -> dict:
    text = fetch_url_text(url)
    metadata = {"source": url, "modality": "url"}
    if user_id:
        metadata["user_id"] = user_id
    records = vector_service.ingest_text(text, metadata=metadata, target=target)
    return {"chunks": len(records)}

def process_text_job(text: str, target: str, user_id: Optional[str]) -> dict:
    metadata = {"source": "manual", "modality": "text"}
    if user_id:
        metadata["user_id"] = user_id
    records = vector_service.ingest_text(normalize_text(text), metadata=metadata, target=target)
    return {"chunks": len(records)}
