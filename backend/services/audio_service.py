
def process_audio(file_path: str, metadata=None):
from backend.ingestion.audio import transcribe_audio

def process_audio(file_path: str, metadata=None):
    text = transcribe_audio(file_path)
    # Add business logic, e.g., store transcription, update DB, etc.
    return text
