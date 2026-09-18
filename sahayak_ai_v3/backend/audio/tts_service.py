import re
import io
import edge_tts
from typing import AsyncGenerator

# Regex to detect Devanagari script (Hindi)
HINDI_REGEX = re.compile(r'[\u0900-\u097F]')

async def get_voice_for_text(text: str) -> str:
    """Determine the optimal neural voice based on language detection."""
    if HINDI_REGEX.search(text):
        return "hi-IN-SwaraNeural"
    return "en-IN-NeerjaNeural"

async def generate_audio_stream(text: str) -> AsyncGenerator[bytes, None]:
    """Yield audio byte chunks using edge-tts for memory-safe streaming."""
    voice = await get_voice_for_text(text)
    communicate = edge_tts.Communicate(text, voice)
    
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            yield chunk["data"]

async def generate_audio_bytes(text: str) -> bytes:
    """Helper for webhooks: returns the full audio payload in memory."""
    voice = await get_voice_for_text(text)
    communicate = edge_tts.Communicate(text, voice)
    
    buffer = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buffer.write(chunk["data"])
    return buffer.getvalue()
