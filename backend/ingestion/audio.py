"""Audio transcription — HF Inference API primary, local Whisper fallback.

Same pattern as everywhere else in the app: hosted API first (no local RAM/
disk cost), local model only if ENABLE_LOCAL_ML_MODELS=true. Whisper
(openai-whisper) depends on torch — without this gate, using it unconditionally
was exactly the kind of unguarded heavy-model load that caused Render's
memory limit to be hit elsewhere in the app (see hf_models.py, embedder.py).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

_WHISPER_MODEL = None

# Same router.huggingface.co domain confirmed working for embeddings — the
# old api-inference.huggingface.co subdomain no longer resolves at all.
_HF_API_BASE = "https://router.huggingface.co/hf-inference/models"


def _local_models_enabled() -> bool:
    return os.getenv("ENABLE_LOCAL_ML_MODELS", "false").strip().lower() == "true"


def _hf_api_available() -> bool:
    return bool(os.getenv("HUGGINGFACEHUB_API_TOKEN"))


def _hf_api_transcribe(audio_path: Path) -> Optional[str]:
    """Try HF Inference API's automatic-speech-recognition task.

    NOTE: unlike the text-based tasks (embeddings, summarization), audio
    tasks on HF's API take the raw file bytes as the request body directly
    — not JSON. This pattern is standard for HF's API but, unlike
    feature-extraction (embeddings), hasn't been empirically verified
    against the live endpoint yet. If this fails with an unexpected error,
    use the same diagnostic-script approach used to find the working
    embeddings URL to confirm the exact request format for this task.
    """
    if not _hf_api_available():
        return None
    try:
        import requests
    except ImportError:
        return None

    model = os.getenv("HF_MODEL_ASR", "distil-whisper/distil-medium.en")
    url = f"{_HF_API_BASE}/{model}/pipeline/automatic-speech-recognition"
    headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACEHUB_API_TOKEN', '')}"}

    try:
        audio_bytes = audio_path.read_bytes()
        _CONTENT_TYPES = {
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".flac": "audio/flac",
            ".aac": "audio/aac",
            ".m4a": "audio/mp4",  # M4A is technically an MP4 container — audio/mp4 tends to be more reliably recognized than audio/x-m4a
            ".ogg": "audio/ogg",
            ".wma": "audio/x-ms-wma",
        }
        content_type = _CONTENT_TYPES.get(audio_path.suffix.lower(), "application/octet-stream")
        req_headers = {**headers, "Content-Type": content_type}

        resp = requests.post(url, headers=req_headers, data=audio_bytes, timeout=60)
        if resp.status_code == 503:
            logger.info("HF ASR model %s is cold-starting, retrying...", model)
            resp = requests.post(
                url, headers=req_headers, data=audio_bytes,
                params={"wait_for_model": "true"}, timeout=90,
            )
        if resp.status_code >= 400:
            logger.error("HF ASR error — status=%s body=%s", resp.status_code, resp.text[:1000])
        resp.raise_for_status()
        result = resp.json()
        return result.get("text", "").strip() if isinstance(result, dict) else None
    except Exception:
        logger.exception("HF Inference API transcription failed for model %s", model)
        return None


def _get_local_model(size: str = "base"):
    """Load local Whisper (fallback path only, gated behind ENABLE_LOCAL_ML_MODELS)."""
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        try:
            import whisper  # lazy import — pulls in torch, only when actually needed
        except Exception as exc:
            raise RuntimeError(
                "Whisper dependency is unavailable. Install `openai-whisper` to enable local audio transcription."
            ) from exc
        _WHISPER_MODEL = whisper.load_model(size)
    return _WHISPER_MODEL


def transcribe_audio(file_path: Path | str, model_size: str = "base") -> str:
    """Transcribe audio to text.

    Priority: HF Inference API -> local Whisper (only if ENABLE_LOCAL_ML_MODELS=true).
    Raises RuntimeError if neither is available.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    api_text = _hf_api_transcribe(path)
    if api_text is not None:
        return api_text

    if _local_models_enabled():
        model = _get_local_model(model_size)
        result = model.transcribe(str(path))
        return result.get("text", "").strip()

    raise RuntimeError(
        "No transcription backend available: HF Inference API failed/unconfigured "
        "(check HUGGINGFACEHUB_API_TOKEN) and local models are disabled "
        "(ENABLE_LOCAL_ML_MODELS=false)."
    )
