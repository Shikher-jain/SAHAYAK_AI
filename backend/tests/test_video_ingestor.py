import unittest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch
from dotenv import load_dotenv
load_dotenv()
from backend.ingestion.video import transcribe_video

class TestVideoIngestion(unittest.TestCase):
    def setUp(self):
        handle = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        handle.write(b"00fakevideo")
        handle.close()
        self.test_video = handle.name

    def test_transcribe_video(self):
        fake_audio = Path(self.test_video).with_suffix(".wav")
        with patch("backend.ingestion.video.extract_audio_from_video", return_value=fake_audio), patch(
            "backend.ingestion.video.transcribe_audio", return_value="spoken content"
        ), patch("backend.ingestion.video.safe_unlink"):
            result = transcribe_video(self.test_video)
        self.assertEqual(result, "spoken content")

    def tearDown(self):
        if os.path.exists(self.test_video):
            os.remove(self.test_video)

if __name__ == "__main__":
    unittest.main()
