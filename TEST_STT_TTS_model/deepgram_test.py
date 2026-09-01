from deepgram import DeepgramClient
from dotenv import load_dotenv
import os

load_dotenv()

client = DeepgramClient(api_key=os.getenv("DEEPGRAM_API_KEY"))

with open(r"D:\shikher sih\SAHAYAK_AI\Test_ingestion\documents\audio\sample.wav", "rb") as audio_file:
    response = client.listen.v1.media.transcribe_file(
        request=audio_file.read(),
        model="nova-3"
    )
    print(response.results.channels[0].alternatives[0].transcript)