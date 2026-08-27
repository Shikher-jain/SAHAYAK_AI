import os
import tempfile
import unittest
from unittest.mock import patch
from dotenv import load_dotenv
load_dotenv()
from backend.ingestion.audio import transcribe_audio


class TestAudioIngestion(unittest.TestCase):

    def setUp(self):
        handle = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        handle.write(b"RIFF....WAVEfmt ")
        handle.close()
        self.test_audio_path = handle.name

    # -------------------------
    # ✅ CASE 1: HF API SUCCESS
    # -------------------------
    def test_transcribe_audio_hf_success(self):
        with patch("backend.ingestion.audio._hf_api_transcribe", return_value="hello world"):
            result = transcribe_audio(self.test_audio_path)

        self.assertEqual(result, "hello world")

    # -------------------------
    # ✅ CASE 2: HF FAIL → LOCAL SUCCESS
    # -------------------------
    def test_transcribe_audio_local_fallback(self):

        class FakeModel:
            def transcribe(self, path):
                return {"text": "local transcription"}

        with patch("backend.ingestion.audio._hf_api_transcribe", return_value=None), \
             patch("backend.ingestion.audio._local_models_enabled", return_value=True), \
             patch("backend.ingestion.audio._get_local_model", return_value=FakeModel()):

            result = transcribe_audio(self.test_audio_path)

        self.assertEqual(result, "local transcription")

    # -------------------------
    # ❌ CASE 3: BOTH FAIL
    # -------------------------
    def test_transcribe_audio_no_backend(self):
        with patch("backend.ingestion.audio._hf_api_transcribe", return_value=None), \
             patch("backend.ingestion.audio._local_models_enabled", return_value=False):

            with self.assertRaises(RuntimeError):
                transcribe_audio(self.test_audio_path)

    # -------------------------
    # ❌ CASE 4: FILE NOT FOUND
    # -------------------------
    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            transcribe_audio("non_existent.wav")

    def tearDown(self):
        if os.path.exists(self.test_audio_path):
            os.remove(self.test_audio_path)


if __name__ == "__main__":
    unittest.main()