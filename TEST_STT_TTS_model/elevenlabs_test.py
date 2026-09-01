import os
from io import BytesIO
import requests
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs.play import play

# Load environment variables
load_dotenv()

# Initialize the ElevenLabs client
elevenlabs = ElevenLabs(
    api_key=os.getenv("ELEVENLABS_API_KEY"),
)

transcription_text = ""

# 1. Attempt to transcribe the local file
try:
    # Fixed: Added 'r' prefix to make it a raw string to handle Windows backslashes
    file_path = r"D:\shikher sih\SAHAYAK_AI\Test_ingestion\documents\audio\sample.wav"
    
    with open(file_path, "rb") as audio_file:
        transcription = elevenlabs.speech_to_text.convert(
            file=audio_file,
            model_id="scribe_v2",
            tag_audio_events=True,
            language_code="eng",
            diarize=True,
        )
    
    # Fixed: Extract the text string from the object
    transcription_text = transcription.text
    print("Local Transcription Success:", transcription_text)

except Exception as local_error:
    print(f"Local file processing failed: {local_error}. Trying fallback URL...")
    
    # 2. Fallback: Transcribe audio from the Google Cloud URL
    audio_url = "https://storage.googleapis.com/eleven-public-cdn/audio/marketing/nicole.mp3"
    try:
        response = requests.get(audio_url)
        response.raise_for_status()
        audio_data = BytesIO(response.content)
        
        # Give the in-memory file an artificial name so Scribe knows the format context
        audio_data.name = "fallback.mp3" 

        transcription = elevenlabs.speech_to_text.convert(
            file=audio_data,
            model_id="scribe_v2",
            tag_audio_events=True,
            language_code="eng",
            diarize=True,
        )
        
        # Fixed: Extract the text string from the object
        transcription_text = transcription.text
        print("Fallback URL Transcription Success:", transcription_text)
        
    except Exception as fallback_error:
        print(f"Fallback failed: {fallback_error}")

# 3. Text-to-Speech Conversion
try:
    # If transcription failed entirely, transcription_text will be empty
    if not transcription_text:
        raise ValueError("No text transcribed from sources.")
        
    print("Generating speech for transcribed text...")
    audio = elevenlabs.text_to_speech.convert(
        text=transcription_text,
        voice_id="JBFqnCBsd6RMkjVDRZzb",  # "George"
        model_id="eleven_v3",
        output_format="mp3_44100_128",
    )
except Exception as tts_error:
    print(f"TTS failed for transcribed text ({tts_error}). Using fallback static phrase.")
    audio = elevenlabs.text_to_speech.convert(
        text="The first move is what sets everything in motion.",
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        model_id="eleven_v3",
        output_format="mp3_44100_128",
    )

# Play the resulting audio
play(audio)
