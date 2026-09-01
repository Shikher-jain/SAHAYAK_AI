import os

from sarvamai import SarvamAI

from dotenv import load_dotenv

load_dotenv(override=True)

def transcribe_audio(audio_path: str):
    api_key = os.getenv("SARVAM_API_KEY")

    if not api_key:
        raise RuntimeError(
            "SARVAM_API_KEY environment variable is not set."
        )

    client = SarvamAI(
        api_subscription_key=api_key,
    )

    with open(audio_path, "rb") as audio_file:
        response = client.speech_to_text.transcribe(
            file=audio_file,
            model="saaras:v3",
            mode="transcribe",
        )

    return response


if __name__ == "__main__":
    response = transcribe_audio("Test_ingestion/documents/audio/sample.mp3")

    print("Transcript:", response.transcript)
    print("Language:", response.language_code)