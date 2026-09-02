FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies:
#   tesseract-ocr        — local OCR (tier-1, fast)
#   libtesseract-dev     — required by pytesseract
#   tesseract-ocr-eng    — English language data pack
#   ffmpeg               — audio/video transcription (moviepy, pydub)
#   libgl1               — OpenCV dependency used by some PDF/image libs
#   libglib2.0-0         — glib runtime required by libgl1
#   poppler-utils        — pdfplumber / pdf2image backend
#   libmagic1            — python-magic for file-type detection
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libtesseract-dev \
    tesseract-ocr-eng \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

EXPOSE 8000

# Shell form (not exec-form array) so $PORT actually expands —
# Render sets PORT dynamically for web services; local docker-compose
# doesn't set it, so it falls back to 8000 there.
CMD uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}