"""
Text chunking for RAG ingestion.

Strategies:
  fixed     — character windows (chunk_size=500, overlap=50)
  recursive — paragraph → sentence → word; never splits mid-sentence
  semantic  — merge consecutive sentences when embedding similarity is high
"""
from __future__ import annotations

import re
from typing import List

import numpy as np

# Strategy 1 defaults (character-based fixed windows).
DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 50

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def ingest_text(text: str) -> str:
    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    strategy: str = "fixed",
) -> List[str]:
    cleaned = text.strip()
    if not cleaned:
        return []
    if strategy == "recursive":
        return _chunk_recursive(cleaned, chunk_size)
    if strategy == "semantic":
        return _chunk_semantic(cleaned, chunk_size)
    return _chunk_fixed(cleaned, chunk_size, overlap)


def _chunk_fixed(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Strategy 1: sliding character windows with overlap."""
    if len(text) <= chunk_size:
        return [text]
    chunks: List[str] = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(text), step):
        piece = text[start : start + chunk_size].strip()
        if piece:
            chunks.append(piece)
    return chunks


def _split_sentences(text: str) -> List[str]:
    sentences: List[str] = []
    for part in _SENTENCE_SPLIT.split(text.strip()):
        cleaned = part.strip()
        if cleaned:
            sentences.append(cleaned)
    return sentences


def _split_paragraphs(text: str) -> List[str]:
    return [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]


def _split_words_to_char_limit(text: str, chunk_size: int) -> List[str]:
    """Word-boundary splits for text that exceeds chunk_size (recursive fallback)."""
    words = text.split()
    if not words:
        return []
    parts: List[str] = []
    current: List[str] = []
    current_len = 0
    for word in words:
        if len(word) > chunk_size:
            if current:
                parts.append(" ".join(current))
                current = []
                current_len = 0
            for start in range(0, len(word), chunk_size):
                piece = word[start : start + chunk_size]
                if piece:
                    parts.append(piece)
            continue
        extra = len(word) if not current else len(word) + 1
        if current and current_len + extra > chunk_size:
            parts.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += extra
    if current:
        parts.append(" ".join(current))
    return parts


def _sentences_for_recursive(text: str, chunk_size: int) -> List[str]:
    """Flatten paragraphs to sentences; oversized sentences split at word boundaries."""
    units: List[str] = []
    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        paragraphs = [text]
    for paragraph in paragraphs:
        for sentence in _split_sentences(paragraph):
            if len(sentence) <= chunk_size:
                units.append(sentence)
            else:
                units.extend(_split_words_to_char_limit(sentence, chunk_size))
    return units


def _chunk_recursive(text: str, chunk_size: int) -> List[str]:
    """Strategy 2: paragraph → sentence → word; never breaks inside a sentence."""
    units = _sentences_for_recursive(text, chunk_size)
    if not units:
        return _split_words_to_char_limit(text, chunk_size)
    return _merge_units_by_char_limit(units, chunk_size)


def _merge_units_by_char_limit(units: List[str], chunk_size: int) -> List[str]:
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for unit in units:
        unit_len = len(unit)
        separator = 1 if current else 0
        if current and current_len + separator + unit_len > chunk_size:
            chunks.append(" ".join(current))
            current = [unit]
            current_len = unit_len
        else:
            if current:
                current_len += separator + unit_len
            else:
                current_len = unit_len
            current.append(unit)
    if current:
        chunks.append(" ".join(current))
    return chunks


def _chunk_semantic(text: str, chunk_size: int, similarity_threshold: float = 0.75) -> List[str]:
    """
    Strategy 3: group consecutive sentences with similar embeddings.

    Cosine similarity between normalized sentence vectors; merge while under chunk_size.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []
    if len(sentences) == 1:
        return sentences if len(sentences[0]) <= chunk_size else _split_words_to_char_limit(sentences[0], chunk_size)

    from backend.common.embedder import embed_texts

    vectors = embed_texts(sentences)
    # L2-normalized vectors → dot product equals cosine similarity.
    similarities = np.sum(vectors[:-1] * vectors[1:], axis=1)

    chunks: List[str] = []
    current_sentences: List[str] = []

    for idx, sentence in enumerate(sentences):
        if not current_sentences:
            current_sentences = [sentence]
            continue

        merged = " ".join(current_sentences + [sentence])
        should_merge = (
            idx > 0
            and similarities[idx - 1] >= similarity_threshold
            and len(merged) <= chunk_size
        )
        if should_merge:
            current_sentences.append(sentence)
        else:
            chunks.append(" ".join(current_sentences))
            current_sentences = [sentence]

    if current_sentences:
        chunks.append(" ".join(current_sentences))
    return chunks
