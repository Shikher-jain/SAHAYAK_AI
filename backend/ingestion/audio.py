"""Audio transcription — multi-tier fallback chain, using each provider's
official SDK (confirmed working via empirical test scripts, not guessed).

Priority: HF Inference API (Whisper) -> AssemblyAI -> Deepgram -> ElevenLabs
-> Sarvam AI -> local Whisper (only if ENABLE_LOCAL_ML_MODELS=true).

Each tier tried in order; any failure (missing key, network error, API
error) silently falls through to the next.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_WHISPER_MODEL = None
_HF_API_BASE = "https://router.huggingface.co/hf-inference/models"

_MIME_TYPES = {
    ".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4",
    ".flac": "audio/flac", ".ogg": "audio/ogg", ".aac": "audio/aac",
}


def _local_models_enabled() -> bool:
    return os.getenv("ENABLE_LOCAL_ML_MODELS", "false").strip().lower() == "true"


# --- Tier 1: HuggingFace Inference API (Whisper) ---

def _hf_api_transcribe(audio_path: Path) -> Optional[str]:
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        return None
    try:
        import requests
    except ImportError:
        return None

    model = os.getenv("HF_MODEL_ASR", "openai/whisper-base")
    url = f"{_HF_API_BASE}/{model}/pipeline/automatic-speech-recognition"
    content_type = _MIME_TYPES.get(audio_path.suffix.lower(), "application/octet-stream")
    headers = {"Authorization": f"Bearer {token}", "Content-Type": content_type}
    try:
        audio_bytes = audio_path.read_bytes()
        resp = requests.post(url, headers=headers, data=audio_bytes, timeout=60)
        if resp.status_code == 503:
            resp = requests.post(
                url, headers=headers, data=audio_bytes,
                params={"wait_for_model": "true"}, timeout=90,
            )
        if resp.status_code >= 400:
            logger.error("HF ASR error — status=%s body=%s", resp.status_code, resp.text[:500])
        resp.raise_for_status()
        result = resp.json()
        return result.get("text", "").strip() if isinstance(result, dict) else None
    except Exception:
        logger.exception("HF Inference API transcription failed (model=%s)", model)
        return None


# --- Tier 2: AssemblyAI (official SDK) ---

def _assemblyai_transcribe(audio_path: Path) -> Optional[str]:
    api_key = os.getenv("ASSEMBLYAI_API_KEY")
    if not api_key:
        return None
    try:
        from assemblyai import TranscriptStatus
        from assemblyai.prerecorded.v2 import Transcriber, TranscriptionConfig
    except ImportError:
        logger.warning("assemblyai package not installed; skipping tier.")
        return None
    try:
        config = TranscriptionConfig(language_detection=True)
        transcriber = Transcriber(api_key=api_key)
        transcript = transcriber.transcribe(str(audio_path), config=config)
        if transcript.status == TranscriptStatus.error:
            logger.error("AssemblyAI error: %s", transcript.error)
            return None
        return (transcript.text or "").strip() or None
    except Exception:
        logger.exception("AssemblyAI transcription failed")
        return None


# --- Tier 3: Deepgram (official SDK) ---

def _deepgram_transcribe(audio_path: Path) -> Optional[str]:
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        return None
    try:
        from deepgram import DeepgramClient
    except ImportError:
        logger.warning("deepgram-sdk package not installed; skipping tier.")
        return None
    try:
        client = DeepgramClient(api_key=api_key)
        with open(audio_path, "rb") as f:
            response = client.listen.v1.media.transcribe_file(request=f.read(), model="nova-3")
        text = response.results.channels[0].alternatives[0].transcript
        return (text or "").strip() or None
    except Exception:
        logger.exception("Deepgram transcription failed")
        return None


# --- Tier 4: ElevenLabs Scribe (official SDK) ---

def _elevenlabs_transcribe(audio_path: Path) -> Optional[str]:
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        return None
    try:
        from elevenlabs.client import ElevenLabs
    except ImportError:
        logger.warning("elevenlabs package not installed; skipping tier.")
        return None
    try:
        client = ElevenLabs(api_key=api_key)
        with open(audio_path, "rb") as f:
            transcription = client.speech_to_text.convert(
                file=f, model_id="scribe_v2", language_code="eng",
            )
        return (transcription.text or "").strip() or None
    except Exception:
        logger.exception("ElevenLabs transcription failed")
        return None


# --- Tier 5: Sarvam AI (official SDK) ---

def _sarvam_transcribe(audio_path: Path) -> Optional[str]:
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        return None
    try:
        from sarvamai import SarvamAI
    except ImportError:
        logger.warning("sarvamai package not installed; skipping tier.")
        return None
    try:
        client = SarvamAI(api_subscription_key=api_key)
        with open(audio_path, "rb") as f:
            response = client.speech_to_text.transcribe(file=f, model="saaras:v3", mode="transcribe")
        return (response.transcript or "").strip() or None
    except Exception:
        logger.exception("Sarvam AI transcription failed")
        return None


# --- Tier 6 (final): local Whisper — only if ENABLE_LOCAL_ML_MODELS=true ---

def _get_local_model(size: str = "base"):
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        try:
            import whisper
        except Exception as exc:
            raise RuntimeError(
                "Whisper dependency is unavailable. Install `openai-whisper` to enable local audio transcription."
            ) from exc
        _WHISPER_MODEL = whisper.load_model(size)
    return _WHISPER_MODEL


_TIERS = [
    ("HF Inference API", _hf_api_transcribe),
    ("AssemblyAI", _assemblyai_transcribe),
    ("Deepgram", _deepgram_transcribe),
    ("ElevenLabs", _elevenlabs_transcribe),
    ("Sarvam AI", _sarvam_transcribe),
]


def transcribe_audio(file_path: Path | str, model_size: str = "base") -> str:
    """Transcribe audio to text, trying each hosted tier in order, then
    local Whisper as a final fallback. Raises RuntimeError only if every
    tier fails/is unconfigured."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    for tier_name, tier_fn in _TIERS:
        text = tier_fn(path)
        if text:
            logger.info("Transcription succeeded via %s", tier_name)
            return text

    if _local_models_enabled():
        model = _get_local_model(model_size)
        result = model.transcribe(str(path))
        return result.get("text", "").strip()

    raise RuntimeError(
        "No transcription backend available — all hosted tiers "
        "(HF, AssemblyAI, Deepgram, ElevenLabs, Sarvam) failed/unconfigured, "
        "and local models are disabled (ENABLE_LOCAL_ML_MODELS=false)."
    )