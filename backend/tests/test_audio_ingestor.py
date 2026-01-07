import unittest
from backend.ingestion.audio import AudioIngestor
import numpy as np
import soundfile as sf
import os

class TestAudioIngestor(unittest.TestCase):
    def setUp(self):
        self.ingestor = AudioIngestor()
        self.test_audio = "test_audio.wav"
        # Generate a silent audio file
        data = np.zeros(int(16000 * 1), dtype=np.float32)  # 1 second at 16kHz
        sf.write(self.test_audio, data, 16000)

    def test_ingest_audio(self):
        self.ingestor.ingest_audio(self.test_audio, metadata={"id": "test_audio"})
        # No assertion: just check no error

    def tearDown(self):
        if os.path.exists(self.test_audio):
            os.remove(self.test_audio)

if __name__ == "__main__":
    unittest.main()
