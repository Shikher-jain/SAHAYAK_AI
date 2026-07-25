"""Document intelligence endpoints — summarize, QnA, notes, explain.

Backend priority: Groq (default, low-RAM, free-tier friendly) -> local HF
model (only if ENABLE_LOCAL_ML_MODELS=true, e.g. on a dev machine with
enough RAM). This mirrors the same priority pattern already used in
backend/rag/generator.py. See backend/common/hf_models.py for why local
models are gated off by default.
"""
import json
import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel

from backend.common.groq_client import groq_complete
from backend.common.hf_models import HFModels
from backend.common.rate_limit import limiter
from backend.services import vector_service

logger = logging.getLogger(__name__)
router = APIRouter()


class DocumentSummarizeRequest(BaseModel):
    document_id: Optional[str] = None
    text: Optional[str] = None


class DocumentSummarizeResponse(BaseModel):
    summary: str
    key_points: List[str]


@router.post("/document/summarize", response_model=DocumentSummarizeResponse)
@limiter.limit("20/minute")
async def summarize_document(request: Request, payload: DocumentSummarizeRequest):
    """Summarize a document or given text."""
    if not payload.document_id and not payload.text:
        raise HTTPException(status_code=400, detail="Either document_id or text must be provided.")

    content_to_summarize = payload.text
    if payload.document_id and not payload.text:
        content_to_summarize, _ = vector_service.get_document_chunks(payload.document_id)
    if not content_to_summarize or not content_to_summarize.strip():
        raise HTTPException(status_code=400, detail="No content available to summarize.")

    # --- Try Groq first ---
    system_prompt = (
        "You summarize documents. Respond ONLY with valid JSON in this exact shape: "
        '{"summary": "<2-4 sentence summary>", "key_points": ["<point 1>", "<point 2>", ...]}. '
        "No markdown fences, no extra text outside the JSON."
    )
    raw = groq_complete(system_prompt, content_to_summarize[:8000], temperature=0.2, max_tokens=600)
    if raw:
        try:
            parsed = json.loads(raw)
            return DocumentSummarizeResponse(
                summary=parsed.get("summary", "").strip(),
                key_points=[p.strip() for p in parsed.get("key_points", []) if p.strip()],
            )
        except (json.JSONDecodeError, AttributeError):
            # Groq answered but not in the requested JSON shape — use the raw text
            # as the summary rather than discarding a perfectly usable answer.
            logger.warning("Groq summarize response wasn't valid JSON; using raw text as summary.")
            return DocumentSummarizeResponse(summary=raw.strip(), key_points=[])

    # --- Fall back to local model only if explicitly enabled ---
    summarizer_pipeline = HFModels.get_summarizer()
    if not summarizer_pipeline:
        raise HTTPException(
            status_code=503,
            detail="Summarization unavailable: Groq call failed and no local model is enabled. "
                   "Check GROQ_API_KEY, or set ENABLE_LOCAL_ML_MODELS=true on a machine with enough RAM.",
        )
    try:
        summary_result = summarizer_pipeline(content_to_summarize, max_length=150, min_length=50, do_sample=False)
        summary_text = summary_result[0]["summary_text"]
        key_points = [sentence.strip() for sentence in summary_text.split(".") if sentence.strip()]
        return DocumentSummarizeResponse(summary=summary_text, key_points=key_points)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {e}")


class DocumentQnARequest(BaseModel):
    document_id: Optional[str] = None
    question: str


class DocumentQnAResponse(BaseModel):
    answer: str
    confidence: float
    source_chunk: Optional[str]


@router.post("/document/qna", response_model=DocumentQnAResponse)
@limiter.limit("20/minute")
async def document_qna(request: Request, payload: DocumentQnARequest):
    """Answer a question based on document content."""
    if not payload.document_id:
        raise HTTPException(status_code=400, detail="document_id must be provided.")

    document_content, _ = vector_service.get_document_chunks(payload.document_id)
    if not document_content or not document_content.strip():
        raise HTTPException(status_code=400, detail="No content found for the given document_id.")

    # --- Try Groq first ---
    system_prompt = (
        "Answer the user's question using ONLY the provided context. "
        "If the answer isn't in the context, say so explicitly rather than guessing."
    )
    user_prompt = f"Context:\n{document_content[:8000]}\n\nQuestion: {payload.question}"
    answer = groq_complete(system_prompt, user_prompt, temperature=0.1, max_tokens=400)
    if answer:
        # NOTE: unlike the extractive RoBERTa-QnA model this replaces, Groq doesn't
        # return a real confidence score — it's a generative answer, not a span
        # extracted with a probability. We report a fixed indicative value rather
        # than fabricate false precision; treat this field as approximate.
        not_found = "not in the context" in answer.lower() or "doesn't contain" in answer.lower()
        return DocumentQnAResponse(
            answer=answer,
            confidence=0.3 if not_found else 0.8,
            source_chunk=document_content[:500],
        )

    # --- Fall back to local model only if explicitly enabled ---
    qna_pipeline = HFModels.get_qna()
    if not qna_pipeline:
        raise HTTPException(
            status_code=503,
            detail="QnA unavailable: Groq call failed and no local model is enabled. "
                   "Check GROQ_API_KEY, or set ENABLE_LOCAL_ML_MODELS=true on a machine with enough RAM.",
        )
    try:
        qna_result = qna_pipeline(question=payload.question, context=document_content)
        return DocumentQnAResponse(
            answer=qna_result["answer"],
            confidence=qna_result["score"],
            source_chunk=document_content[:500],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"QnA failed: {e}")


class DocumentNotesResponse(BaseModel):
    notes: str  # Markdown formatted notes


@router.post("/document/notes", response_model=DocumentNotesResponse)
@limiter.limit("20/minute")
async def generate_document_notes(request: Request, document_id: str):
    """Generate structured markdown notes for a document."""
    document_content, _ = vector_service.get_document_chunks(document_id)
    if not document_content or not document_content.strip():
        raise HTTPException(status_code=400, detail="No content found for the given document_id.")

    system_prompt = (
        "You generate structured study notes in markdown, with a Title, Key Concepts, "
        "Summary, and Important Points. Respond with markdown only, no commentary."
    )
    notes = groq_complete(system_prompt, document_content[:8000], temperature=0.3, max_tokens=800)
    if notes:
        return DocumentNotesResponse(notes=notes)

    text_generation_pipeline = HFModels.get_text_generation()
    if not text_generation_pipeline:
        raise HTTPException(
            status_code=503,
            detail="Note generation unavailable: Groq call failed and no local model is enabled. "
                   "Check GROQ_API_KEY, or set ENABLE_LOCAL_ML_MODELS=true on a machine with enough RAM.",
        )
    try:
        prompt = (
            "Generate structured notes for the following document, including a Title, "
            f"Key Concepts, Summary, and Important Points, formatted in markdown:\n\nDocument: {document_content}"
        )
        generated_notes = text_generation_pipeline(prompt, max_length=500, num_return_sequences=1)
        return DocumentNotesResponse(notes=generated_notes[0]["generated_text"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Note generation failed: {e}")


class DocumentExplainRequest(BaseModel):
    text: str
    level: str  # beginner/intermediate/expert


class DocumentExplainResponse(BaseModel):
    explanation: str
    examples: List[str]


@router.post("/document/explain", response_model=DocumentExplainResponse)
@limiter.limit("20/minute")
async def explain_text(request: Request, payload: DocumentExplainRequest):
    """Explain a given text at the requested level (beginner/intermediate/expert)."""
    system_prompt = (
        f"Explain the given text at a {payload.level} level. Respond ONLY with valid JSON: "
        '{"explanation": "<explanation>", "examples": ["<example 1>", "<example 2>"]}. '
        "No markdown fences, no extra text outside the JSON."
    )
    raw = groq_complete(system_prompt, payload.text[:4000], temperature=0.4, max_tokens=600)
    if raw:
        try:
            parsed = json.loads(raw)
            return DocumentExplainResponse(
                explanation=parsed.get("explanation", "").strip(),
                examples=[e.strip() for e in parsed.get("examples", []) if e.strip()],
            )
        except (json.JSONDecodeError, AttributeError):
            logger.warning("Groq explain response wasn't valid JSON; using raw text as explanation.")
            return DocumentExplainResponse(explanation=raw.strip(), examples=[])

    text_generation_pipeline = HFModels.get_text_generation()
    if not text_generation_pipeline:
        raise HTTPException(
            status_code=503,
            detail="Explanation unavailable: Groq call failed and no local model is enabled. "
                   "Check GROQ_API_KEY, or set ENABLE_LOCAL_ML_MODELS=true on a machine with enough RAM.",
        )
    try:
        prompt = f"Explain the following text at a {payload.level} level and provide examples:\n\nText: {payload.text}\n\nExplanation and Examples:"
        generated_explanation = text_generation_pipeline(prompt, max_length=300, num_return_sequences=1)
        explanation_text = generated_explanation[0]["generated_text"]
        examples = []
        if "Examples:" in explanation_text:
            parts = explanation_text.split("Examples:", 1)
            explanation_text = parts[0].strip()
            examples = [e.strip() for e in parts[1].split("\n") if e.strip()]
        return DocumentExplainResponse(explanation=explanation_text, examples=examples)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {e}")