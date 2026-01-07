import unittest
from backend.ingestion.video import VideoIngestor
import os

class TestVideoIngestor(unittest.TestCase):
    def setUp(self):
        self.ingestor = VideoIngestor()
        self.test_video = "test_video.mp4"
        # Create a dummy video file
        with open(self.test_video, "wb") as f:
            f.write(b"00fakevideo")

    def test_ingest_video(self):
        result = self.ingestor.ingest_video(self.test_video, metadata={"id": "test_video"})
        self.assertEqual(result["status"], "success")

    def tearDown(self):
        if os.path.exists(self.test_video):
            os.remove(self.test_video)

if __name__ == "__main__":
    unittest.main()
