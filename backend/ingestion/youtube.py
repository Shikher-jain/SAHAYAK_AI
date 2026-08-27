"""YouTube ingestion — official transcript first, Whisper/STT fallback.

Flow: URL -> video_id -> try official transcript (youtube_transcript_api,
free, fast, no audio download needed) -> if no transcript exists (captions
disabled/unavailable), download audio (yt-dlp) and transcribe via the
existing HF-API-first pipeline (backend/ingestion/audio.py — same
Groq-free, low-RAM path already used for uploaded audio/video files).

NOTE ON API STABILITY: youtube_transcript_api's and yt-dlp's exact method
signatures have changed across versions historically (similar to how HF's
Inference API domain changed this session). The code below follows the
standard/documented usage as of this writing, but — like the HF endpoint
work earlier — treat this as needing empirical verification against the
installed package versions before relying on it; if a call fails with an
AttributeError/TypeError rather than a clean "no transcript" error, check
the installed library's actual API first rather than assuming this code
is simply wrong.
"""
from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_VIDEO_ID_PATTERNS = [
    re.compile(r"(?:v=|/)([0-9A-Za-z_-]{11})(?:[&?/]|$)"),  # youtube.com/watch?v=..., youtube.com/embed/...
    re.compile(r"youtu\.be/([0-9A-Za-z_-]{11})"),             # youtu.be/...
]


def extract_video_id(url: str) -> Optional[str]:
    """Pull the 11-character video ID out of common YouTube URL formats."""
    for pattern in _VIDEO_ID_PATTERNS:
        match = pattern.search(url)
        if match:
            return match.group(1)
    return None


def _fetch_official_transcript(video_id: str) -> Optional[str]:
    """Tier 1: official captions/transcript, if the video has them.
    No audio download needed — fast and free."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        logger.warning("youtube_transcript_api not installed; skipping transcript tier.")
        return None
    try:
        segments = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join(seg.get("text", "") for seg in segments).strip()
        return text or None
    except Exception:
        # Covers TranscriptsDisabled, NoTranscriptFound, VideoUnavailable, etc.
        # — any of these just means "fall through to the STT tier", not a hard failure.
        logger.info("No official transcript available for video_id=%s; falling back to STT.", video_id)
        return None


def _download_audio(url: str) -> Optional[Path]:
    """Tier 2 (part 1): download audio-only stream via yt-dlp.
    Requires ffmpeg on the host (already installed in backend/Dockerfile)."""
    try:
        import yt_dlp
    except ImportError:
        logger.warning("yt-dlp not installed; cannot download audio for STT fallback.")
        return None

    tmp_dir = Path(tempfile.mkdtemp(prefix="yt_audio_"))
    out_template = str(tmp_dir / "audio.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": out_template,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "128",
        }],
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        # After FFmpegExtractAudio postprocessing, the file should be audio.mp3
        candidate = tmp_dir / "audio.mp3"
        if candidate.exists():
            return candidate
        # Fallback: grab whatever single file landed in tmp_dir
        found = list(tmp_dir.glob("audio.*"))
        return found[0] if found else None
    except Exception:
        logger.exception("yt-dlp audio download failed for url=%s", url)
        return None


def get_youtube_text(url: str) -> str:
    """Get transcript text for a YouTube URL — official transcript if
    available, otherwise downloads audio and transcribes it (reusing the
    existing HF-API-first Whisper pipeline). Raises ValueError if the URL
    isn't recognized, RuntimeError if both tiers fail."""
    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError(f"Could not extract a YouTube video ID from URL: {url}")

    transcript = _fetch_official_transcript(video_id)
    if transcript:
        return transcript

    audio_path = _download_audio(url)
    if audio_path is None:
        raise RuntimeError(
            "No official transcript available and audio download failed "
            "(check yt-dlp is installed and ffmpeg is available)."
        )

    from backend.ingestion.audio import transcribe_audio  # reuse existing HF-API-first pipeline
    try:
        return transcribe_audio(audio_path)
    finally:
        # Clean up the downloaded audio file + its temp directory.
        try:
            audio_path.unlink(missing_ok=True)
            audio_path.parent.rmdir()
        except OSError:
            pass