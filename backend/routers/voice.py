from __future__ import annotations

import base64
from io import BytesIO
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from backend.auth import api_key_auth
from backend.ingestion.audio import transcribe_audio
from backend.services import vector_service
from backend.utils.file_utils import create_named_temp_from_bytes, safe_unlink

router = APIRouter(tags=["voice"], dependencies=[Depends(api_key_auth)])


async def _persist_upload(file: UploadFile) -> str:
    """Persist an uploaded file to a temp location and return the path."""

    payload = await file.read()
    tmp_path = create_named_temp_from_bytes(payload, original_name=file.filename or "audio.wav")
    return str(tmp_path)


@router.post("/voice/transcribe")
async def transcribe_endpoint(file: UploadFile = File(...)) -> Dict[str, str]:
    """Transcribe audio into text using the existing Whisper ingestion code."""

    tmp_path = await _persist_upload(file)
    try:
        text = transcribe_audio(tmp_path)
        return {"text": text}
    finally:
        safe_unlink(tmp_path)


@router.post("/voice/speak")
def speak_endpoint(text: str = Form(...), lang: str = Form("en")) -> Response:
    """Convert input text to speech and return mp3 bytes."""

    try:
        from gtts import gTTS
    except Exception as exc:
        raise HTTPException(status_code=503, detail="gTTS dependency is unavailable") from exc

    buffer = BytesIO()
    try:
        tts = gTTS(text=text, lang=lang)
        tts.write_to_fp(buffer)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    audio_bytes = buffer.getvalue()
    return Response(content=audio_bytes, media_type="audio/mpeg")


@router.post("/voice/voice_query")
async def voice_query(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    top_k: int = Form(5),
    target: str = Form("auto"),
) -> Dict[str, Any]:
    """Voice pipeline: transcribe → RAG query → speak answer."""

    tmp_path = await _persist_upload(file)
    try:
        query_text = transcribe_audio(tmp_path)
    finally:
        safe_unlink(tmp_path)

    rag = vector_service.rag_answer(query_text, top_k=top_k, target=target, session_id=session_id)
    answer_text = str(rag.get("answer") or "")
    sources = rag.get("sources") or []

    audio_b64: Optional[str] = None
    try:
        from gtts import gTTS

        buffer = BytesIO()
        gTTS(text=answer_text or " ", lang="en").write_to_fp(buffer)
        audio_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    except Exception:
        audio_b64 = None

    return {
        "text_answer": answer_text,
        "audio_answer": audio_b64,
        "sources": sources,
        "query": query_text,
        "session_id": session_id,
    }

