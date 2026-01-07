from io import BytesIO
from pathlib import Path

import pdfplumber


def extract_pdf_text(pdf_path: Path | str) -> str:
    path = Path(pdf_path)
    with pdfplumber.open(path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


def extract_pdf_text_from_bytes(payload: bytes) -> str:
    with pdfplumber.open(BytesIO(payload)) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)
