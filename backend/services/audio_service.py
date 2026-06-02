from __future__ import annotations

from pathlib import Path

from backend.ingestion.audio import transcribe_audio
from backend.processing.timeline import extract_timeline


def process_audio(file_path: str | Path, metadata=None) -> str:
    return transcribe_audio(file_path)


def build_audio_timeline(transcript: str):
    # TASK 6 FIX: generate a timeline from audio/video transcripts for metadata.
    return extract_timeline(transcript)
