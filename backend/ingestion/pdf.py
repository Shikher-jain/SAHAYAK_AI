from collections import Counter
from io import BytesIO
import logging
import math
import os
import re
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HuggingFace OCR fallback configuration
# ---------------------------------------------------------------------------
HF_API_TOKEN: str | None = os.getenv("HF_API_TOKEN")
HF_OCR_MODEL: str = os.getenv("HF_OCR_MODEL", "baidu/Unlimited-OCR")
HF_OCR_ENDPOINT: str = f"https://api-inference.huggingface.co/models/{HF_OCR_MODEL}"
# Pages whose rendered pixmap exceeds this many pixels skip local Tesseract
# and go straight to the HF API (default: 1 Mpx ≈ 1000×1000).
OCR_MAX_PIXELS: int = int(os.getenv("OCR_MAX_PIXELS", "1000000"))

# ---------------------------------------------------------------------------
# Text-cleaning constants
# ---------------------------------------------------------------------------
HEADER_FOOTER_THRESHOLD = 0.6
HEADER_FOOTER_MAX_LENGTH = 120
UNICODE_BULLET_CODES = (0x2022, 0x2023, 0x25E6, 0x2043, 0x2219)
UNICODE_BULLETS = "".join(chr(code) for code in UNICODE_BULLET_CODES)
BULLET_PATTERN = re.compile(rf"^\s*(?:[-*]|(?:\d+|[A-Za-z])[.)]|[{UNICODE_BULLETS}])\s+")
LEADING_LABEL_PATTERN = re.compile(
    r"^\s*(?:Figure|Table|Listing|Appendix)\s+\d+[:.-]\s*", re.IGNORECASE
)
NON_ASCII_PATTERN = re.compile(r"[^\x09\x0A\x0D\x20-\x7E]")
NOISE_PATTERNS = [
    re.compile(r"^\s*page\s+\d+(\s+of\s+\d+)?\s*$", re.IGNORECASE),
    re.compile(r"^\s*confidential.*$", re.IGNORECASE),
    re.compile(r"^\s*copyright\s+\d{4}.*$", re.IGNORECASE),
    re.compile(r"^\s*all rights reserved.*$", re.IGNORECASE),
]

# Below this many characters, a page's text-layer extraction is treated as
# "failed" (rather than "genuinely a near-empty page") and the fallback
# chain kicks in.
MIN_MEANINGFUL_CHARS = 20


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_pdf_text(pdf_path: "Path | str") -> str:
    payload = Path(pdf_path).read_bytes()
    return extract_pdf_text_from_bytes(payload)


def extract_pdf_text_from_bytes(payload: bytes) -> str:
    pages = _extract_pages(payload)
    return _clean_document_text(pages)


# ---------------------------------------------------------------------------
# Internal extraction pipeline
# ---------------------------------------------------------------------------

def _extract_pages(payload: bytes) -> list[str]:
    """Extract text per page with automatic fallback: pdfplumber -> PyMuPDF
    -> OCR (Tesseract or HF API). A single PDF can have some pages with a
    normal text layer and others that are scanned images — every page still
    gets text; the caller never needs to know which path was taken."""
    try:
        import pdfplumber
    except Exception as exc:
        raise RuntimeError(
            "pdfplumber is unavailable. Install `pdfplumber` to enable PDF ingestion."
        ) from exc

    fitz_doc = _open_fitz(payload)  # None if PyMuPDF isn't installed/usable
    try:
        pages_text: list[str] = []
        with pdfplumber.open(BytesIO(payload)) as pdf:
            for i, page in enumerate(pdf.pages):
                text = (page.extract_text() or "").strip()
                if len(text) < MIN_MEANINGFUL_CHARS:
                    fallback_text = _fallback_page_text(fitz_doc, i)
                    if fallback_text:
                        text = fallback_text
                pages_text.append(text)
        return pages_text
    finally:
        if fitz_doc is not None:
            fitz_doc.close()


def _open_fitz(payload: bytes):
    """Open the PDF with PyMuPDF for the fallback path. Returns None (not an
    exception) if PyMuPDF is unavailable or the file can't be opened —
    callers treat that as 'no fallback available'."""
    try:
        import pymupdf as fitz  # noqa: PLC0415
    except Exception:
        return None
    try:
        return fitz.open(stream=payload, filetype="pdf")
    except Exception:
        return None


def _fallback_page_text(fitz_doc, page_index: int) -> str:
    """Tier 2: PyMuPDF's own text extraction.
    Tier 3: OCR (Tesseract locally, or HF API when the image is too large)."""
    if fitz_doc is None or page_index >= len(fitz_doc):
        return ""
    page = fitz_doc[page_index]

    text = (page.get_text() or "").strip()
    if len(text) >= MIN_MEANINGFUL_CHARS:
        return text

    return _ocr_page(page)


# ---------------------------------------------------------------------------
# OCR – Tesseract (local) with HF API fallback
# ---------------------------------------------------------------------------

def _ocr_page(page) -> str:
    """Render *page* to a pixmap and OCR it.

    Strategy:
    1. Render at 80 DPI (low memory footprint).
    2. If the rendered size is within OCR_MAX_PIXELS, try local Tesseract.
    3. On any failure (OOM, missing binary, huge page) fall back to the
       HuggingFace Inference API.
    """
    try:
        from backend.ingestion.image import _image_to_string
        from PIL import Image  # noqa: PLC0415
    except Exception:
        logger.warning("PIL / pytesseract not available; skipping local OCR.")
        from backend.ingestion.image import _image_to_string  # type: ignore
        Image = None  # will trigger HF fallback below

    pix = None
    try:
        pix = page.get_pixmap(dpi=80)
    except Exception as exc:
        logger.warning("Pixmap generation failed: %s — trying HF OCR", exc)
        return hf_ocr(None)

    png_bytes = pix.tobytes("png")

    # If the page is very large, skip Tesseract and go straight to HF OCR
    if pix.width * pix.height > OCR_MAX_PIXELS:
        logger.info(
            "Page too large (%dx%d px) for local OCR — using HF OCR.",
            pix.width, pix.height,
        )
        return hf_ocr(png_bytes)

    try:
        image = Image.open(BytesIO(png_bytes))  # type: ignore[union-attr]
        text = _image_to_string(image).strip()
        if text:
            logger.debug("OCR method: local Tesseract")
            return text
    except Exception as exc:
        logger.info("Local OCR failed (%s) — falling back to HF OCR.", exc)

    return hf_ocr(png_bytes)


def hf_ocr(image_bytes: "bytes | None") -> str:
    """Send *image_bytes* to the HuggingFace Inference API for OCR.

    Returns an empty string (never raises) so the ingestion pipeline can
    continue even when the HF service is unreachable or the token is missing.

    The response format differs by model:
    - baidu/Unlimited-OCR returns ``{"text": "..."}``
    - Other models may return a list; we try both shapes.
    """
    if not HF_API_TOKEN:
        logger.warning("HF_API_TOKEN is not set; cannot use HF OCR fallback.")
        return ""
    if not image_bytes:
        logger.warning("hf_ocr called with no image bytes.")
        return ""

    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "image/png",
    }
    try:
        resp = requests.post(
            HF_OCR_ENDPOINT,
            headers=headers,
            data=image_bytes,
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()

        # Handle {"text": "..."} (Baidu model)
        if isinstance(payload, dict):
            return payload.get("text", "").strip()
        # Handle [{"generated_text": "..."}] (some HF pipelines)
        if isinstance(payload, list) and payload:
            first = payload[0]
            if isinstance(first, dict):
                return (
                    first.get("generated_text", first.get("text", ""))
                ).strip()
            if isinstance(first, str):
                return first.strip()
        logger.warning("Unexpected HF OCR response shape: %r", payload)
        return ""
    except Exception as exc:
        logger.warning("HuggingFace OCR failed: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Text-cleaning pipeline
# ---------------------------------------------------------------------------

def _clean_document_text(pages: list[str]) -> str:
    if not pages:
        return ""
    headers, footers = _detect_repeated_edges(pages)
    cleaned_pages = [_clean_page_text(page, headers, footers) for page in pages]
    cleaned_pages = [page for page in cleaned_pages if page]
    if not cleaned_pages:
        return ""
    return _normalize_whitespace("\n\n".join(cleaned_pages))


def _detect_repeated_edges(pages: list[str]) -> tuple[set[str], set[str]]:
    header_counter: Counter[str] = Counter()
    footer_counter: Counter[str] = Counter()
    for raw in pages:
        lines = _prepare_lines(raw)
        if not lines:
            continue
        header_counter[_normalize_edge_line(lines[0])] += 1
        footer_counter[_normalize_edge_line(lines[-1])] += 1
    threshold = max(2, math.ceil(len(pages) * HEADER_FOOTER_THRESHOLD))
    header_lines = {
        line
        for line, count in header_counter.items()
        if count >= threshold and len(line) <= HEADER_FOOTER_MAX_LENGTH
    }
    footer_lines = {
        line
        for line, count in footer_counter.items()
        if count >= threshold and len(line) <= HEADER_FOOTER_MAX_LENGTH
    }
    return header_lines, footer_lines


def _clean_page_text(raw_page: str, headers: set[str], footers: set[str]) -> str:
    lines = _prepare_lines(raw_page)
    if not lines:
        return ""
    while lines and _normalize_edge_line(lines[0]) in headers:
        lines.pop(0)
    while lines and _normalize_edge_line(lines[-1]) in footers:
        lines.pop()
    cleaned: list[str] = []
    for line in lines:
        sanitized = _sanitize_line(line)
        if not sanitized or _is_noise_line(sanitized):
            continue
        cleaned.append(sanitized)
    return "\n".join(cleaned)


def _prepare_lines(raw: str) -> list[str]:
    return [line.strip() for line in raw.splitlines() if line and line.strip()]


def _normalize_edge_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip().lower()


def _sanitize_line(line: str) -> str:
    line = BULLET_PATTERN.sub("", line)
    line = LEADING_LABEL_PATTERN.sub("", line)
    line = NON_ASCII_PATTERN.sub(" ", line)
    line = re.sub(r"\s+", " ", line)
    return line.strip().strip("-•*")


def _is_noise_line(line: str) -> bool:
    if not line:
        return True
    lowered = line.lower()
    if len(lowered) <= 2:
        return True
    for pattern in NOISE_PATTERNS:
        if pattern.match(line):
            return True
    return False


def _normalize_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()