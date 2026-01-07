from io import BytesIO
from pathlib import Path

import pytesseract
from PIL import Image


def ocr_image(image_path: Path | str) -> str:
    image = Image.open(Path(image_path))
    return pytesseract.image_to_string(image)


def ocr_image_bytes(data: bytes, suffix: str = "png") -> str:
    image = Image.open(BytesIO(data))
    return pytesseract.image_to_string(image)
