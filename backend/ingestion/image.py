"""
image.py — OCR pipeline with Tesseract-first, HF Inference API fallback.

Local Tesseract is tried first (fast, free). If the binary is missing or OCR
returns nothing, the call is transparently forwarded to the HuggingFace
Inference API (model configured via HF_OCR_MODEL env var).
"""
from __future__ import annotations

import logging
import os
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HuggingFace OCR fallback (same vars as pdf.py – shared from env)
# ---------------------------------------------------------------------------
_HF_API_TOKEN: str | None = os.getenv("HF_API_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
_HF_OCR_MODEL: str = os.getenv("HF_OCR_MODEL", "baidu/Unlimited-OCR")
_HF_OCR_ENDPOINT: str = f"https://api-inference.huggingface.co/models/{_HF_OCR_MODEL}"


def _hf_ocr_bytes(image_bytes: bytes) -> str:
    """Call the HuggingFace Inference API with raw PNG bytes. Returns '' on
    any error so callers never need to handle exceptions."""
    if not _HF_API_TOKEN:
        logger.warning("HF_API_TOKEN is not set; cannot use HF OCR fallback.")
        return ""
    headers = {
        "Authorization": f"Bearer {_HF_API_TOKEN}",
        "Content-Type": "image/png",
    }
    try:
        resp = requests.post(_HF_OCR_ENDPOINT, headers=headers, data=image_bytes, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        if isinstance(payload, dict):
            return payload.get("text", "").strip()
        if isinstance(payload, list) and payload:
            first = payload[0]
            if isinstance(first, dict):
                return (first.get("generated_text") or first.get("text") or "").strip()
            if isinstance(first, str):
                return first.strip()
        logger.warning("Unexpected HF OCR response shape: %r", payload)
        return ""
    except Exception as exc:
        logger.warning("HuggingFace OCR API failed: %s", exc)
        return ""


def _image_to_string(image: Image.Image) -> str:
    """OCR a PIL Image. Tries local Tesseract first; falls back to HF API."""
    # --- Tier 1: local Tesseract ---
    try:
        import pytesseract
        text = pytesseract.image_to_string(image).strip()
        if text:
            logger.debug("OCR method: local Tesseract")
            return text
    except Exception as exc:
        logger.info("Local Tesseract OCR failed (%s) — falling back to HF OCR.", exc)

    # --- Tier 2: HuggingFace Inference API ---
    png_buf = BytesIO()
    image.save(png_buf, format="PNG")
    text = _hf_ocr_bytes(png_buf.getvalue())
    if text:
        logger.debug("OCR method: HuggingFace API")
    return text


def ocr_image(image_path: "Path | str") -> str:
    image = Image.open(Path(image_path))
    return _image_to_string(image)


def ocr_image_bytes(data: bytes, suffix: str = "png") -> str:
    image = Image.open(BytesIO(data))
    return _image_to_string(image)
