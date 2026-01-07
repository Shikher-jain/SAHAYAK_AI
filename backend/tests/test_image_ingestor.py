import unittest
from backend.ingestion.image import ImageIngestor
from PIL import Image
import numpy as np
import os

class TestImageIngestor(unittest.TestCase):
    def setUp(self):
        self.ingestor = ImageIngestor()
        self.test_img = "test_img.png"
        img = Image.fromarray(np.zeros((224,224,3), dtype=np.uint8))
        img.save(self.test_img)

    def test_ingest_image(self):
        self.ingestor.ingest_image(self.test_img, metadata={"id": "test_img"})
        # No assertion: just check no error

    def tearDown(self):
        if os.path.exists(self.test_img):
            os.remove(self.test_img)

if __name__ == "__main__":
    unittest.main()
