from pathlib import Path

from moviepy import VideoFileClip

from backend.ingestion.audio import transcribe_audio
from backend.utils.file_utils import get_tmp_path


def extract_audio_from_video(video_path: Path | str) -> Path:
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
    return transcribe_audio(audio_path, model_size=model_size)
