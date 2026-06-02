from pathlib import Path

from backend.ingestion.audio import transcribe_audio
from backend.utils.file_utils import get_tmp_path, safe_unlink


def extract_audio_from_video(video_path: Path | str) -> Path:
    try:
        from moviepy import VideoFileClip
    except Exception as exc:
        raise RuntimeError("moviepy is unavailable. Install `moviepy` to enable video ingestion.") from exc

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    audio_path = get_tmp_path(f"{video_path.stem}_audio.wav")
    clip = VideoFileClip(str(video_path))
    clip.audio.write_audiofile(audio_path.as_posix())
    clip.close()
    return audio_path


def transcribe_video(video_path: Path | str, model_size: str = "base") -> str:
    audio_path = extract_audio_from_video(video_path)
    try:
        return transcribe_audio(audio_path, model_size=model_size)
    finally:
        safe_unlink(audio_path)
