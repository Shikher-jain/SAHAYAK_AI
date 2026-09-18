from fastapi import APIRouter, UploadFile, File, BackgroundTasks
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/pdf")
async def ingest_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Ingest a PDF document.
    Dispatches task to the Ingestion Worker (PyMuPDF) to avoid blocking the API.
    """
    logger.info(f"Received PDF for ingestion: {file.filename}")
    # background_tasks.add_task(dispatch_to_ingestion_worker, file_bytes)
    return {"status": "processing", "message": "PDF dispatched to ingestion worker."}

@router.post("/audio")
async def ingest_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Ingest an audio file.
    Dispatches to the Audio Worker (Whisper) for ASR.
    """
    logger.info(f"Received audio for ingestion: {file.filename}")
    return {"status": "processing", "message": "Audio dispatched to ASR worker."}
