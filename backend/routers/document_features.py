from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Form, HTTPException

from backend.auth import api_key_auth
from backend.common.hf_models import get_hf_models
from backend.services import vector_service

router = APIRouter(tags=["document-features"], dependencies=[Depends(api_key_auth)])


def _resolve_document_text(document_id: Optional[str], text: Optional[str], target: str) -> str:
    """Resolve text from either a document_id (source) or raw text input."""

    if text and text.strip():
        return text.strip()
    if document_id and document_id.strip():
        resolved = vector_service.get_document_text(document_id.strip(), target=target)
        if resolved.strip():
            return resolved
        raise HTTPException(status_code=404, detail="No content found for document_id")
    raise HTTPException(status_code=400, detail="Provide either document_id or text")


def _extract_key_points(summary: str, max_points: int = 5) -> List[str]:
    """Derive short key points from a summary using sentence splitting."""

    normalized = re.sub(r"\s+", " ", summary or "").strip()
    if not normalized:
        return []
    sentences = [seg.strip() for seg in re.split(r"(?<=[.!?])\s+", normalized) if seg.strip()]
    points: List[str] = []
    for sentence in sentences:
        if sentence in points:
            continue
        points.append(sentence)
        if len(points) >= max_points:
            break
    if not points:
        points = [normalized]
    return points


@router.post("/document/summarize")
def summarize_document(
    document_id: Optional[str] = Form(None),
    text: Optional[str] = Form(None),
    target: str = Form("auto"),
) -> Dict[str, Any]:
    """Summarize a document (by id/source) or provided text using BART."""

    payload = _resolve_document_text(document_id, text, target=target)
    models = get_hf_models()
    summarizer = models.summarizer()
    if summarizer is None:
        summary = vector_service.summarize_text(payload)
    else:
        try:
            result = summarizer(payload, max_length=220, min_length=60, do_sample=False)
            summary = str(result[0].get("summary_text", "")).strip()
        except Exception:
            summary = vector_service.summarize_text(payload)
    return {"summary": summary, "key_points": _extract_key_points(summary)}


@router.post("/document/qna")
def document_qna(
    document_id: str = Form(...),
    question: str = Form(...),
    target: str = Form("auto"),
) -> Dict[str, Any]:
    """Answer a question against a stored document using RoBERTa QnA."""

    document_text, chunks = vector_service.get_document_chunks(document_id.strip(), target=target)
    if not document_text.strip():
        raise HTTPException(status_code=404, detail="No content found for document_id")

    models = get_hf_models()
    qa = models.qna()
    if qa is None:
        rag = vector_service.rag_answer(question, top_k=5, target=target, session_id=None)
        return {"answer": rag.get("answer", ""), "confidence": 0.0, "source_chunk": ""}

    best_answer = ""
    best_score = -1.0
    best_chunk = ""
    for chunk in chunks:
        context = chunk.strip()
        if not context:
            continue
        try:
            out = qa(question=question, context=context)
            score = float(out.get("score") or 0.0)
            answer = str(out.get("answer") or "").strip()
        except Exception:
            continue
        if score > best_score and answer:
            best_score = score
            best_answer = answer
            best_chunk = context

    if not best_answer:
        return {"answer": "", "confidence": 0.0, "source_chunk": ""}
    return {"answer": best_answer, "confidence": max(0.0, best_score), "source_chunk": best_chunk}


@router.post("/document/notes")
def document_notes(document_id: str = Form(...), target: str = Form("auto")) -> Dict[str, Any]:
    """Generate structured markdown notes for a stored document using Flan-T5."""

    text = vector_service.get_document_text(document_id.strip(), target=target)
    if not text.strip():
        raise HTTPException(status_code=404, detail="No content found for document_id")

    models = get_hf_models()
    gen = models.text_generator()
    if gen is None:
        raise HTTPException(status_code=503, detail="Text generation model unavailable")

    prompt = (
        "Create structured notes in MARKDOWN with exactly these sections:\n"
        "## Title\n"
        "## Key Concepts\n"
        "## Summary\n"
        "## Important Points\n\n"
        "Use concise bullet points where appropriate.\n\n"
        f"Text:\n{text}\n\nNotes:"
    )
    try:
        out = gen(prompt, max_new_tokens=320, do_sample=False)
        notes = str(out[0].get("generated_text", "")).strip()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"notes": notes}


@router.post("/document/explain")
def explain_text(
    text: str = Form(...),
    level: str = Form("beginner"),
) -> Dict[str, Any]:
    """Explain text at a requested difficulty level using Flan-T5."""

    level_norm = (level or "beginner").strip().lower()
    if level_norm not in {"beginner", "intermediate", "expert"}:
        raise HTTPException(status_code=400, detail="level must be beginner/intermediate/expert")

    models = get_hf_models()
    gen = models.text_generator()
    if gen is None:
        raise HTTPException(status_code=503, detail="Text generation model unavailable")

    prompt = (
        f"Explain the following text at a {level_norm} level. "
        "Include 2 short examples.\n\n"
        f"Text:\n{text}\n\nExplanation:"
    )
    try:
        out = gen(prompt, max_new_tokens=260, do_sample=False)
        explanation = str(out[0].get("generated_text", "")).strip()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"explanation": explanation, "examples": []}

