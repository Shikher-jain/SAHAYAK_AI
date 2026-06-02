from typing import List, Optional

from fastapi import APIRouter, File, UploadFile, HTTPException, Response
from pydantic import BaseModel
from io import BytesIO

from backend.ingestion.audio import transcribe_audio as whisper_transcribe
from backend.services import vector_service
from gtts import gTTS

router = APIRouter()

class TranscribeResponse(BaseModel):
    transcribed_text: str

@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_voice(audio_file: UploadFile = File(...)):
    """Transcribes an audio file using the existing Whisper model."""
    if not audio_file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only audio files are supported.")
    
    try:
        audio_bytes = await audio_file.read()
        # Assuming whisper_transcribe takes bytes and returns text
        transcribed_text = whisper_transcribe(audio_bytes)
        return TranscribeResponse(transcribed_text=transcribed_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio transcription failed: {e}")

class SpeakRequest(BaseModel):
    text: str

@router.post("/speak", response_class=Response, responses={200: {"content": {"audio/mpeg": {}}}})  # Correct response class and media type
async def speak_text(request: SpeakRequest):
    """Converts text to speech using the gTTS library and returns MP3 audio bytes."""
    try:
        tts = gTTS(text=request.text, lang="en") # Default to English
        audio_buffer = BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return Response(content=audio_buffer.read(), media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text to speech failed: {e}")

class VoiceQueryResponse(BaseModel):
    text_answer: str
    audio_answer: str  # Base64 encoded audio
    sources: list

@router.post("/voice_query", response_model=VoiceQueryResponse)
async def voice_query(audio_file: UploadFile = File(...), session_id: Optional[str] = None, language: str = "en"):
    """Pipeline: transcribe audio -> RAG query -> speak answer."""
    if not audio_file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only audio files are supported.")
    
    try:
        # 1. Transcribe audio
        audio_bytes = await audio_file.read()
        transcribed_text = whisper_transcribe(audio_bytes)

        # 2. RAG query via the unified vector service
        rag_response = vector_service.rag_answer(
            query=transcribed_text,
            top_k=5,
            session_id=session_id,
        )
        text_answer = rag_response.get("answer", "")
        sources = rag_response.get("sources", [])

        # 3. Speak answer with optional language support
        tts_lang = language if language in {"en", "hi", "es", "fr", "de"} else "en"
        tts = gTTS(text=text_answer, lang=tts_lang)
        audio_buffer = BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        # For now, returning audio as base64 string. In a real app, you might save it and return a URL.
        import base64
        audio_answer_base64 = base64.b64encode(audio_buffer.read()).decode("utf-8")

        return VoiceQueryResponse(
            text_answer=text_answer,
            audio_answer=audio_answer_base64,
            sources=sources
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice query pipeline failed: {e}")