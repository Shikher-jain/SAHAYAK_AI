from pathlib import Path
from typing import Optional

_WHISPER_MODEL = None


def _get_model(size: str = "base"):
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        try:
            import whisper
        except Exception as exc:
            raise RuntimeError(
                "Whisper dependency is unavailable. Install `openai-whisper` to enable audio transcription."
            ) from exc
        _WHISPER_MODEL = whisper.load_model(size)
    return _WHISPER_MODEL


def transcribe_audio(file_path: Path | str, model_size: str = "base") -> str:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    model = _get_model(model_size)
    result = model.transcribe(str(path))
    return result.get("text", "").strip()
