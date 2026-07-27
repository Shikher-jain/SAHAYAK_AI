FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    tesseract-ocr \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

EXPOSE 8000

# Shell form (not exec-form array) so $PORT actually expands — Render sets
# PORT dynamically for web services; local `docker-compose up` doesn't set
# it, so it falls back to 8000 there.
CMD uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}