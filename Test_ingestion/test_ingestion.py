"""
Sahayak AI — end-to-end ingestion test.

Tests every supported input type against the live backend (local or Render)
in a single run and prints a PASS/FAIL/SKIP summary.

Usage:
    python test_ingestion.py                    # tests localhost:8000
    python test_ingestion.py --url https://sahayak-ai-yxl8.onrender.com
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Pre-flight check — verifies env vars + dependencies before running any
# actual ingestion tests, so a missing package/token is caught in 2 seconds
# instead of after a slow HTTP request fails deep in the pipeline.
# ---------------------------------------------------------------------------

def preflight_check() -> bool:
    print("=" * 60)
    print("PRE-FLIGHT CHECK")
    print("=" * 60)
    ok = True

    required_env = ["JWT_SECRET_KEY", "HUGGINGFACEHUB_API_TOKEN", "GROQ_API_KEY"]
    optional_env = ["QDRANT_URL", "QDRANT_API_KEY", "AUTH_DATABASE_URL", "SAHAYAK_API_KEY"]

    for var in required_env:
        val = os.getenv(var, "").strip()
        status = "OK" if val else "MISSING"
        print(f"  [{status}] {var}")
        if not val:
            ok = False

    for var in optional_env:
        val = os.getenv(var, "").strip()
        print(f"  [{'set' if val else 'not set (ok if intentional)'}] {var}")

    required_packages = {
        "requests": "requests",
        "yt_dlp": "yt-dlp",
        "youtube_transcript_api": "youtube_transcript_api",
        "moviepy": "moviepy",
        "pytesseract": "pytesseract",
        "PIL": "Pillow",
        "pdfplumber": "pdfplumber",
        "fitz": "PyMuPDF",
    }
    for module_name, pip_name in required_packages.items():
        try:
            __import__(module_name)
            print(f"  [OK] {pip_name}")
        except ImportError:
            print(f"  [MISSING] {pip_name} — run: pip install {pip_name}")
            ok = False

    print("=" * 60)
    if not ok:
        print("Pre-flight check FAILED — fix the above before running tests.\n")
    else:
        print("Pre-flight check passed.\n")
    return ok


if not preflight_check():
    sys.exit(1)


DOCS = Path(__file__).parent / "documents"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--url", default=os.getenv("BACKEND_URL", "http://127.0.0.1:8000"))
parser.add_argument("--target", default="auto")
args = parser.parse_args()

BASE_URL = args.url.rstrip("/")
API_KEY = os.getenv("SAHAYAK_API_KEY", "").strip()
HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}

results = []  # (label, status, detail)


def record(label: str, status: str, detail: str = "") -> None:
    results.append((label, status, detail))
    icon = {"PASS": "\u2705", "FAIL": "\u274c", "SKIP": "\u23ed\ufe0f"}.get(status, "?")
    print(f"{icon} {label}: {status}" + (f" — {detail}" if detail else ""))


def post_file(label: str, endpoint: str, file_path: Path, mime: str, extra_data: Optional[dict] = None) -> None:
    if not file_path.exists():
        record(label, "SKIP", f"sample file not found: {file_path}")
        return
    try:
        with open(file_path, "rb") as f:
            files = {"file": (file_path.name, f, mime)}
            resp = requests.post(
                f"{BASE_URL}{endpoint}",
                params={"target": args.target},
                files=files,
                data=extra_data or {},
                headers=HEADERS,
                timeout=600,
            )
        if resp.status_code == 200:
            record(label, "PASS", f"HTTP 200 — {resp.text[:120]}")
        else:
            record(label, "FAIL", f"HTTP {resp.status_code} — {resp.text[:200]}")
    except Exception as exc:
        record(label, "FAIL", f"{type(exc).__name__}: {exc}")


def post_form(label: str, endpoint: str, data: dict) -> None:
    try:
        resp = requests.post(
            f"{BASE_URL}{endpoint}",
            params={"target": args.target},
            data=data,
            headers=HEADERS,
            timeout=600,
        )
        if resp.status_code == 200:
            record(label, "PASS", f"HTTP 200 — {resp.text[:120]}")
        else:
            record(label, "FAIL", f"HTTP {resp.status_code} — {resp.text[:200]}")
    except Exception as exc:
        record(label, "FAIL", f"{type(exc).__name__}: {exc}")


print(f"Testing backend: {BASE_URL}\n")

# --- Health check first ---
try:
    r = requests.get(f"{BASE_URL}/health", timeout=30)
    record("Health check", "PASS" if r.status_code == 200 else "FAIL", r.text[:150])
except Exception as exc:
    record("Health check", "FAIL", str(exc))
    print("\nBackend unreachable — aborting further tests.")
    sys.exit(1)

# --- PDF (4 variants) ---
post_file("PDF — text-based", "/ingest/pdf", DOCS / "pdf" / "text_pdf.pdf", "application/pdf")
post_file("PDF — multi-column", "/ingest/pdf", DOCS / "pdf" / "column_topic_pdf.pdf", "application/pdf")
post_file("PDF — scanned (OCR path)", "/ingest/pdf", DOCS / "pdf" / "scan_pdf.pdf", "application/pdf")
post_file("PDF — embedded image", "/ingest/pdf", DOCS / "pdf" / "image_pdf.pdf", "application/pdf")

# --- Images (3 variants) ---
post_file("Image — clean (OCR)", "/ingest/image", DOCS / "images" / "sample.png", "image/png")
post_file("Image — noisy scan (OCR)", "/ingest/image", DOCS / "images" / "scanned.jpg", "image/jpeg")
post_file("Image — handwriting-style (OCR)", "/ingest/image", DOCS / "images" / "handwritten.jpg", "image/jpeg")

# --- Audio / Video — need real files supplied by you ---
post_file("Audio (.wav)", "/ingest/audio", DOCS / "audio" / "sample.wav", "audio/wav")
post_file("Audio (.mp3)", "/ingest/audio", DOCS / "audio" / "sample.mp3", "audio/mpeg")
post_file("Video (.mp4)", "/ingest/video", DOCS / "video" / "sample.mp4", "video/mp4")

# --- CSV ---
post_file("CSV", "/ingest/csv", DOCS / "csv" / "csv_data.csv", "text/csv")

# --- Code (3 languages) ---
post_file("Code — Python", "/ingest/code", DOCS / "code" / "sample.py", "text/x-python")
post_file("Code — JavaScript", "/ingest/code", DOCS / "code" / "sample.js", "application/javascript")
post_file("Code — C++", "/ingest/code", DOCS / "code" / "sample.cpp", "text/x-c++src")

# --- Raw text ---
text_file = DOCS / "text" / "sample.txt"
if text_file.exists():
    post_form("Raw text", "/ingest/text", {"text": text_file.read_text(encoding="utf-8")})
else:
    record("Raw text", "SKIP", "sample.txt not found")

# --- URL (web scraping) + YouTube (from links.txt) ---
links_file = DOCS / "links" / "links.txt"
if links_file.exists():
    urls = [
        line.strip() for line in links_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    for url in urls:
        if "youtube.com" in url or "youtu.be" in url:
            if "https://www.youtube.com/watch?v=a-UNLln4ZcE" in url:
                record(f"YouTube — {url[:50]}", "SKIP", "placeholder URL — replace with a real video ID in links.txt")
            else:
                post_form(f"YouTube — {url[:50]}", "/ingest/youtube", {"url": url})
        else:
            post_form(f"URL — {url[:50]}", "/ingest/url", {"url": url})
else:
    record("URL/YouTube", "SKIP", "links.txt not found")

# --- Summary ---
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
passed = sum(1 for _, s, _ in results if s == "PASS")
failed = sum(1 for _, s, _ in results if s == "FAIL")
skipped = sum(1 for _, s, _ in results if s == "SKIP")
print(f"Total: {len(results)}  |  Passed: {passed}  |  Failed: {failed}  |  Skipped: {skipped}")
if failed:
    print("\nFailed tests:")
    for label, status, detail in results:
        if status == "FAIL":
            print(f"  - {label}: {detail}")