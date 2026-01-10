from collections import Counter
from io import BytesIO
import math
from pathlib import Path
import re

import pdfplumber

HEADER_FOOTER_THRESHOLD = 0.6
HEADER_FOOTER_MAX_LENGTH = 120
UNICODE_BULLET_CODES = (0x2022, 0x2023, 0x25E6, 0x2043, 0x2219)
UNICODE_BULLETS = "".join(chr(code) for code in UNICODE_BULLET_CODES)
BULLET_PATTERN = re.compile(rf"^\s*(?:[-*]|(?:\d+|[A-Za-z])[.)]|[{UNICODE_BULLETS}])\s+")
LEADING_LABEL_PATTERN = re.compile(r"^\s*(?:Figure|Table|Listing|Appendix)\s+\d+[:.-]\s*", re.IGNORECASE)
NON_ASCII_PATTERN = re.compile(r"[^\x09\x0A\x0D\x20-\x7E]")
NOISE_PATTERNS = [
    re.compile(r"^\s*page\s+\d+(\s+of\s+\d+)?\s*$", re.IGNORECASE),
    re.compile(r"^\s*confidential.*$", re.IGNORECASE),
    re.compile(r"^\s*copyright\s+\d{4}.*$", re.IGNORECASE),
    re.compile(r"^\s*all rights reserved.*$", re.IGNORECASE),
]


def extract_pdf_text(pdf_path: Path | str) -> str:
    path = Path(pdf_path)
    with pdfplumber.open(path) as pdf:
        pages = _extract_pages(pdf)
    return _clean_document_text(pages)


def extract_pdf_text_from_bytes(payload: bytes) -> str:
    with pdfplumber.open(BytesIO(payload)) as pdf:
        pages = _extract_pages(pdf)
    return _clean_document_text(pages)


def _extract_pages(pdf: pdfplumber.PDF) -> list[str]:
    return [page.extract_text() or "" for page in pdf.pages]


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
