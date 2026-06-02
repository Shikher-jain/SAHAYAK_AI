from io import BytesIO
from pathlib import Path

from PIL import Image


def _image_to_string(image: Image.Image) -> str:
    try:
        import pytesseract
    except Exception as exc:
        raise RuntimeError(
            "pytesseract is unavailable. Install `pytesseract` and Tesseract OCR to enable image ingestion."
        ) from exc
    return pytesseract.image_to_string(image)


def ocr_image(image_path: Path | str) -> str:
    image = Image.open(Path(image_path))
    return _image_to_string(image)


def ocr_image_bytes(data: bytes, suffix: str = "png") -> str:
    image = Image.open(BytesIO(data))
    return _image_to_string(image)
