import os
from dotenv import load_dotenv
from assemblyai import TranscriptStatus
from assemblyai.prerecorded.v2 import Transcriber, TranscriptionConfig

load_dotenv()
# Use a publicly-accessible URL
audio_file = "https://assembly.ai/wildfires.mp3"

# Or use a local file:
audio_file = r"D:\shikher sih\SAHAYAK_AI\Test_ingestion\documents\audio\sample.wav"

config = TranscriptionConfig(
    language_detection=True,
    speaker_labels=True,
)

transcriber = Transcriber(api_key=os.getenv("ASSEMBLYAI_API_KEY"))
transcript = transcriber.transcribe(audio_file, config=config)

if transcript.status == TranscriptStatus.error:
    raise RuntimeError(f"Transcription failed: {transcript.error}")

# Log transcript.id for every request (not just errors), with a timestamp and API region.
# It's required to fetch results, retry, or delete the transcript later, and it's the first
# thing support@assemblyai.com asks for. Delete: /pre-recorded-audio/delete-transcripts
# Troubleshooting: /pre-recorded-audio/guides/common_errors_and_solutions

print(f"\nFull Transcript:\n\n{transcript.text}")

# Optionally print speaker diarization results
# for utterance in transcript.utterances:
#     print(f"Speaker {utterance.speaker}: {utterance.text}")