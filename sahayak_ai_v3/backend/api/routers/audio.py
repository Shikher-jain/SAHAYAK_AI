from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from sahayak_ai_v3.backend.audio.tts_service import generate_audio_stream

async def synthesize_audio(req: SynthesizeRequest):

router = APIRouter(prefix="/api/v2/audio", tags=["v2 Audio"])

class SynthesizeRequest(BaseModel):
    text: str

@router.post("/synthesize")
async def synthesize_audio(req: SynthesizeRequest):
    """Streams synthesized audio directly to the client (zero RAM accumulation)."""
    return StreamingResponse(
        generate_audio_stream(req.text),
        media_type="audio/mpeg"
    )
