# from pathlib import Path
# from backend.ingestion.audio import transcribe_audio
# from backend.utils.file_utils import get_tmp_path, safe_unlink

# def extract_audio_from_video(video_path: Path | str) -> Path:
#     try:
        
#         # from moviepy import VideoFileClip
#         try:
#             from moviepy import VideoFileClip
#         except ImportError:
#             from moviepy.editor import VideoFileClip
    
#     except Exception as exc:
#         raise RuntimeError("moviepy is unavailable. Install `moviepy` to enable video ingestion.") from exc


#     video_path = Path(video_path)
#     if not video_path.exists():
#         raise FileNotFoundError(f"Video not found: {video_path}")

#     audio_path = get_tmp_path(f"{video_path.stem}_audio.wav")
#     clip = VideoFileClip(str(video_path))
#     clip.audio.write_audiofile(audio_path.as_posix())
#     clip.close()
#     return audio_path


# def transcribe_video(video_path: Path | str, model_size: str = "base") -> str:
#     audio_path = extract_audio_from_video(video_path)
#     try:
#         return transcribe_audio(audio_path, model_size=model_size)
#     finally:
#         safe_unlink(audio_path)


from pathlib import Path
from backend.ingestion.audio import transcribe_audio
from backend.utils.file_utils import get_tmp_path, safe_unlink

# Import standardly at the top level to see the true error if it fails
try:
    from moviepy.editor import VideoFileClip  # MoviePy 1.0.3 primary import
except ImportError:
    try:
        from moviepy import VideoFileClip  # MoviePy 2.0+ fallback
    except ImportError as exc:
        raise RuntimeError(
            "moviepy is unavailable. Install `moviepy` to enable video ingestion."
        ) from exc


def extract_audio_from_video(video_path: Path | str) -> Path:
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
        
    audio_path = get_tmp_path(f"{video_path.stem}_audio.wav")
    
    # Context manager ensures clip.close() is called even if write fails
    with VideoFileClip(str(video_path)) as clip:
        if clip.audio is None:
            raise ValueError(f"The video file has no audio track: {video_path}")
        # Explicitly configure standard WAV parameters for transcription stability
        clip.audio.write_audiofile(
            audio_path.as_posix(),
            codec="pcm_s16le",
            fps=16000,  # 16kHz is ideal for most ML transcription models
            logger=None # Disables noisy tqdm output in logs
        )
        
    return audio_path


def transcribe_video(video_path: Path | str, model_size: str = "base") -> str:
    audio_path = extract_audio_from_video(video_path)
    try:
        return transcribe_audio(audio_path, model_size=model_size)
    finally:
        safe_unlink(audio_path)
