import unittest
from io import BytesIO
from unittest.mock import patch

from PIL import Image

from backend.ingestion.image import ocr_image_bytes

class TestImageIngestion(unittest.TestCase):
    def test_ocr_image_bytes(self):
        image = Image.new("RGB", (32, 32), color=(255, 255, 255))
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        payload = buffer.getvalue()

        with patch("backend.ingestion.image.pytesseract.image_to_string", return_value="detected text"):
            result = ocr_image_bytes(payload)
        self.assertEqual(result, "detected text")

if __name__ == "__main__":
    unittest.main()
