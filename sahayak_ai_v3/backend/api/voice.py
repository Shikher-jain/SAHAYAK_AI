import io
import os
import logging
from typing import AsyncGenerator, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from groq import AsyncGroq
import edge_tts
from langchain_core.messages import HumanMessage

from sahayak_ai_v3.backend.agents.supervisor import sahayak_agent_app

logger = logging.getLogger("sahayak.voice")

router = APIRouter(prefix="/api/v3/voice", tags=["Voice & Audio"])

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MAX_AUDIO_BYTES = 10 * 1024 * 1024  # 10 MB limit

# Supported Edge-TTS Voices
DEFAULT_VOICE_EN = "en-IN-PrabhatNeural"
DEFAULT_VOICE_HI = "hi-IN-MadhurNeural"


class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = DEFAULT_VOICE_EN
    rate: Optional[str] = "+0%"
    pitch: Optional[str] = "+0Hz"


async def _generate_edge_tts_stream(text: str, voice: str, rate: str, pitch: str) -> AsyncGenerator[bytes, None]:
    """Generates real-time audio chunk stream using Edge-TTS without saving to disk."""
    try:
        communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate, pitch=pitch)
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]
    except Exception as exc:
        logger.error("Edge-TTS streaming failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Text-to-speech audio stream generation failed."
        )


@router.post("/transcribe", summary="Transcribe speech using Groq Cloud Whisper")
async def transcribe_audio(
    audio_file: UploadFile = File(..., description="Multipart audio file (.webm, .mp3, .wav, .m4a)")
):
    """Accepts an audio file upload and returns high-accuracy transcription from Groq Whisper."""
    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Groq API credentials are not configured on the server."
        )

    audio_bytes = await audio_file.read()
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Audio file exceeds the maximum allowed size of 10 MB."
        )

    if len(audio_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty audio file provided."
        )

    try:
        client = AsyncGroq(api_key=GROQ_API_KEY)
        file_payload = (audio_file.filename or "recording.webm", audio_bytes)

        transcription = await client.audio.transcriptions.create(
            file=file_payload,
            model="whisper-large-v3",
            response_format="json",
            temperature=0.0
        )
        return {
            "transcription": transcription.text,
            "filename": audio_file.filename,
            "byte_length": len(audio_bytes)
        }
    except Exception as exc:
        logger.error("Groq Whisper transcription failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Transcription failed: {str(exc)}"
        )


@router.post("/tts", summary="Stream synthesize speech from text using Edge-TTS")
async def stream_text_to_speech(payload: TTSRequest):
    """Synthesizes text into streaming MP3 audio directly via Microsoft Edge-TTS."""
    clean_text = payload.text.strip()
    if not clean_text:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Text field cannot be empty.")

    # Clamp text length to prevent memory saturation
    if len(clean_text) > 4000:
        clean_text = clean_text[:4000]

    voice = payload.voice or DEFAULT_VOICE_EN
    audio_stream = _generate_edge_tts_stream(
        text=clean_text,
        voice=voice,
        rate=payload.rate or "+0%",
        pitch=payload.pitch or "+0Hz"
    )

    return StreamingResponse(
        audio_stream,
        media_type="audio/mpeg",
        headers={
            "Content-Disposition": "inline; filename=speech.mp3",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


@router.post("/process-voice-chat", summary="End-to-end Voice In -> Agent Execution -> JSON Response")
async def process_voice_chat(
    audio_file: UploadFile = File(...),
    session_id: str = Query(default="default_session"),
    user_id: str = Query(default="anonymous_user")
):
    """
    Full conversational turn:
    1. Transcribes audio input via Groq Whisper.
    2. Invokes LangGraph multi-agent supervisor.
    3. Returns transcription, agent response text, citations, and graph state.
    """
    # 1. Transcribe Audio
    transcription_data = await transcribe_audio(audio_file)
    query_text = transcription_data.get("transcription", "").strip()

    if not query_text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Could not transcribe discernible speech from the audio sample."
        )

    # 2. Invoke LangGraph Orchestrator
    initial_state = {
        "messages": [HumanMessage(content=query_text)],
        "next_agent": "supervisor",
        "visual_citations": [],
        "graph_data": None,
        "is_emergency": False,
        "error": None,
        "user_id": user_id,
        "tier": "free",
        "iteration_count": 0
    }

    try:
        final_state = await sahayak_agent_app.ainvoke(initial_state)
        
        # Extract the final answer message from the graph
        response_message = ""
        for msg in reversed(final_state.get("messages", [])):
            if hasattr(msg, "content") and getattr(msg, "type", "") in ["ai", "AIMessage"]:
                response_message = str(msg.content)
                break

        return {
            "transcription": query_text,
            "response": response_message,
            "visual_citations": final_state.get("visual_citations", []),
            "graph_data": final_state.get("graph_data"),
            "is_emergency": final_state.get("is_emergency", False),
            "error": final_state.get("error")
        }
    except Exception as exc:
        logger.error("LangGraph execution failed during voice chat: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error running conversation workflow on transcribed speech."
        )