from __future__ import annotations

_summarizer = None


def _get_summarizer():
    global _summarizer
    if _summarizer is None:
        try:
            from transformers import pipeline

            _summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
            _summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        except Exception:
            _summarizer = None
    return _summarizer


def summarize_text(text: str, max_length: int = 150, min_length: int = 50) -> str:
    if not text or not text.strip():
        return ""
    summarizer = _get_summarizer()
    if summarizer is None:
        sentences = [segment.strip() for segment in text.split(".") if segment.strip()]
        return ". ".join(sentences[:3]).strip() + ("." if sentences else "")
    result = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return result[0]["summary_text"]
