import os
import tempfile
import unittest
from unittest.mock import patch

from backend.ingestion.audio import transcribe_audio


class TestAudioIngestion(unittest.TestCase):
    def setUp(self):
        handle = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        handle.write(b"RIFF....WAVE")
        handle.close()
        self.test_audio = handle.name

    def test_transcribe_audio(self):
        fake_model = type("FakeModel", (), {"transcribe": lambda self, path: {"text": "hello world"}})()
        with patch("backend.ingestion.audio._get_model", return_value=fake_model):
            result = transcribe_audio(self.test_audio)
        self.assertEqual(result, "hello world")

    def tearDown(self):
        if os.path.exists(self.test_audio):
            os.remove(self.test_audio)


if __name__ == "__main__":
    unittest.main()