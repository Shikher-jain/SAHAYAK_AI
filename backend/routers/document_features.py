from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel
from typing import List, Optional

from backend.common.hf_models import HFModels
from backend.common.rate_limit import limiter
from backend.services import vector_service

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
    """Summarize a document or given text using the BART model."""
    if not payload.document_id and not payload.text:
        raise HTTPException(status_code=400, detail="Either document_id or text must be provided.")

    # TASK 17 FIX: Fetch actual document content from vector store when document_id is provided.
    content_to_summarize = payload.text
    if payload.document_id and not payload.text:
        content_to_summarize, _ = vector_service.get_document_chunks(payload.document_id)
    if not content_to_summarize or not content_to_summarize.strip():
        raise HTTPException(status_code=400, detail="No content available to summarize.")

    summarizer_pipeline = HFModels.get_summarizer()
    if not summarizer_pipeline:
        raise HTTPException(status_code=500, detail="Summarization model not available.")

    try:
        summary_result = summarizer_pipeline(content_to_summarize, max_length=150, min_length=50, do_sample=False)
        summary_text = summary_result[0]["summary_text"]
        
        # Simple key point extraction (can be improved)
        key_points = [sentence.strip() for sentence in summary_text.split('.') if sentence.strip()]

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
    """Answer a question based on document content using the RoBERTa QnA model."""
    if not payload.document_id:
        raise HTTPException(status_code=400, detail="document_id must be provided.")

    # TASK 17 FIX: Fetch actual document content for QnA.
    document_content = ""
    if payload.document_id:
        document_content, _ = vector_service.get_document_chunks(payload.document_id)
    if not document_content.strip():
        raise HTTPException(status_code=400, detail="No content found for the given document_id.")

    qna_pipeline = HFModels.get_qna()
    if not qna_pipeline:
        raise HTTPException(status_code=500, detail="QnA model not available.")
    
    try:
        qna_result = qna_pipeline(question=payload.question, context=document_content)
        return DocumentQnAResponse(
            answer=qna_result["answer"],
            confidence=qna_result["score"],
            source_chunk=qna_result["context"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"QnA failed: {e}")

class DocumentNotesResponse(BaseModel):
    notes: str # Markdown formatted notes

@router.post("/document/notes", response_model=DocumentNotesResponse)
@limiter.limit("20/minute")
async def generate_document_notes(request: Request, document_id: str):
    """Generate structured notes for a document using flan-t5."""
    # TASK 19 FIX: Fetch actual document content for notes generation.
    document_content = ""
    if document_id:
        document_content, _ = vector_service.get_document_chunks(document_id)
    if not document_content.strip():
        raise HTTPException(status_code=400, detail="No content found for the given document_id.")

    text_generation_pipeline = HFModels.get_text_generation()
    if not text_generation_pipeline:
        raise HTTPException(status_code=500, detail="Text generation model not available.")

    prompt = f"Generate structured notes for the following document, including a Title, Key Concepts, Summary, and Important Points, formatted in markdown:\n\nDocument: {document_content}"

    try:
        generated_notes = text_generation_pipeline(prompt, max_length=500, num_return_sequences=1)
        notes_text = generated_notes[0]["generated_text"]
        return DocumentNotesResponse(notes=notes_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Note generation failed: {e}")

class DocumentExplainRequest(BaseModel):
    text: str
    level: str # beginner/intermediate/expert

class DocumentExplainResponse(BaseModel):
    explanation: str
    examples: List[str]

@router.post("/document/explain", response_model=DocumentExplainResponse)
@limiter.limit("20/minute")
async def explain_text(request: Request, payload: DocumentExplainRequest):
    """Explain a given text at the requested level (beginner/intermediate/expert)."""
    text_generation_pipeline = HFModels.get_text_generation()
    if not text_generation_pipeline:
        raise HTTPException(status_code=500, detail="Text generation model not available.")

    prompt = f"Explain the following text at a {payload.level} level and provide examples:\n\nText: {payload.text}\n\nExplanation and Examples:"

    try:
        generated_explanation = text_generation_pipeline(prompt, max_length=300, num_return_sequences=1)
        explanation_text = generated_explanation[0]["generated_text"]
        
        # Simple extraction of examples (can be improved)
        examples = []
        if "Examples:" in explanation_text:
            parts = explanation_text.split("Examples:", 1)
            explanation_text = parts[0].strip()
            examples = [e.strip() for e in parts[1].split('\n') if e.strip()]

        return DocumentExplainResponse(explanation=explanation_text, examples=examples)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {e}")