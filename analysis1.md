# Sahayak AI — Complete Project Analysis

## Overview

**Sahayak AI** is a **multimodal RAG (Retrieval-Augmented Generation) knowledge assistant** that consolidates four legacy projects into one platform. It ingests diverse content (PDF, audio, video, images, URLs, text), embeds them into vector space, stores them in a **dual-backend** (Qdrant + SQLite/FAISS fallback), and answers user questions using retrieved context.

**Stack:** FastAPI backend, Streamlit frontend, sentence-transformers for embeddings, Whisper for audio, Tesseract for OCR, BART for summarization, Qdrant + FAISS for vector search.

## Architecture

```text
User → Streamlit UI → FastAPI Backend
                          ├── /ingest/*   → Ingestion modules → vector_service → Qdrant / SQLite+FAISS
                          ├── /search/*   → vector_service → retriever → generator
                          ├── /summaries  → BART summarizer
                          ├── /local/*    → local_stack (standalone SQLite+FAISS)
                          ├── /finetune/* → dataset_service (JSONL LoRA pairs)
                          └── /admin/*    → Qdrant health + metadata
```

| Layer | Key Files |
|-------|-----------|
| **Entry point** | `backend/main.py` — mounts 6 routers, CORS, global error handler |
| **Routers** | `ingestion.py`, `search.py`, `summarize.py`, `admin.py`, `local_mode.py`, `finetune.py` |
| **Orchestration** | `vector_service.py` — central hub, manages dual-backend ingest/search/RAG |
| **Vector stores** | `qdrant_store.py` (primary), `local_stack/db.py` (fallback SQLite+FAISS) |
| **Ingestion** | `pdf.py`, `audio.py`, `video.py`, `image.py`, `url.py`, `text.py` |
| **RAG** | `retriever.py`, `generator.py`, `embedder.py`, `search.py` |
| **Frontend** | `frontend/app.py` — Streamlit UI with 3 modes (Pro Dashboard, Simplified, HuggingFace Mini) |

## What Works Well

| Area | Details |
|------|---------|
| **PDF ingestion** | `backend/ingestion/pdf.py` is the most polished module — sophisticated cleaning pipeline that strips repeated headers/footers, normalizes whitespace, removes noise lines |
| **Dual-backend fallback** | `vector_service.py` elegantly routes to Qdrant or SQLite/FAISS with `target=auto|qdrant|local` |
| **Qdrant integration** | `qdrant_store.py` has graceful degradation — if Qdrant is unreachable, the system falls back silently |
| **Output sanitization** | `_sanitize_output()` / `_sanitize_record()` strip emojis and normalize whitespace from all responses |
| **Frontend** | `frontend/app.py` is a complete Streamlit UI with configurable backend URL, health badges, error handling, and 3 UI modes |
| **Fine-tune API** | Clean Pydantic-validated endpoints for collecting training pairs into JSONL |
| **Test coverage** (partial) | `test_pdf_cleaning.py` and `test_response_sanitization.py` are meaningful and functional |

## Top 10 Bugs & Issues

| # | Severity | Issue | Location |
|---|----------|-------|----------|
| 1 | **Critical** | **BART summarization used as QA** — `rag_answer()` summarizes context instead of answering the question. The model generates a summary, not an answer. | `vector_service.py` |
| 2 | **Critical** | **Double storage in auto mode** — when `target="auto"` and Qdrant is available, `ingest_text()` stores in **both** Qdrant AND local DB | `vector_service.py` |
| 3 | **High** | **3 of 6 tests broken** — `test_audio_ingestor`, `test_image_ingestor`, `test_video_ingestor` import non-existent `*Ingestor` classes | `backend/tests/` |
| 4 | **High** | **FAISS index rebuilt per query** — `build_faiss_index()` loads all embeddings from SQLite on every search → O(n) per query | `local_stack/db.py` |
| 5 | **High** | **Authentication disabled** — `auth.py` exists with a hardcoded API key `"your_api_key"` but is never wired into any endpoint | `backend/auth.py` |
| 6 | **Medium** | **Massive code duplication** — `finetune_stack/` is ~95% copy of `local_stack/`; `processing/` and `rag/` modules have concatenated duplicate function definitions | Multiple files |
| 7 | **Medium** | **Multiple embedding models** — `all-MiniLM-L6-v2` (384-dim), `BAAI/bge-m3` (1024-dim), `clip-ViT-B-32` used across different code paths → dimension mismatches will cause errors if mixed | `rag/embedder.py`, `processing/embeddings.py` |
| 8 | **Medium** | **No temp file cleanup** — uploaded files, extracted audio WAVs, and temp uploads accumulate forever in `~/.sahayak_ai/tmp/` | `file_utils.py`, `video.py` |
| 9 | **Medium** | **Two PDF libraries** — `pdfplumber` (with cleaning) in main pipeline vs `PyMuPDF/fitz` (no cleaning) in local_stack — inconsistent text quality | `ingestion/pdf.py` vs `local_stack/extractor.py` |
| 10 | **Medium** | **`rag/generator.py` is broken** — contains concatenated duplicate function definitions with syntax errors; references a 14GB `Qwen2.5-7B` model, never used | `rag/generator.py` |

## Security Concerns

| Risk | Detail |
|------|--------|
| **SSRF** | `url.py` and `local_stack/extractor.py` fetch arbitrary user-supplied URLs with no validation or allow-list |
| **CORS wide open** | `allow_origins=["*"]` in `main.py` — all origins allowed |
| **No auth enforced** | Admin endpoints, ingestion, search — all publicly accessible |
| **Error info leak** | Global exception handler returns raw `str(exc)` to clients |
| **Pickle deserialization** | SQLite blobs store pickled numpy arrays — potential RCE if DB is tampered with |
| **No input sanitization** | Uploaded filenames used directly — potential path traversal |

## Dead/Unused Code

| Module | Status |
|--------|--------|
| `backend/auth.py` | Never imported by any router |
| `backend/analytics.py` | Never instantiated |
| `backend/rag/generator.py` | Broken syntax, never called |
| `backend/rag/query_rewrite.py` | Not integrated |
| `backend/rag/recommend.py` | Not integrated |
| `backend/rag/duplicate.py` | Not integrated |
| `backend/processing/*` | All 4 files unused by any active pipeline |
| `backend/utils/api_utils.py` | Never imported |
| `scripts/*` | All 4 script files are empty stubs |
| `notebooks/demo.ipynb` | Empty notebook |

## Code Quality Summary

| Metric | Assessment |
|--------|-----------|
| **Modularity** | Good router/service/store separation |
| **Duplication** | High — `finetune_stack` ≈ `local_stack`, many files have copy-pasted definitions |
| **Logging** | Inconsistent — mix of `print()`, `logging`, and emoji-laden console output |
| **Type hints** | Partial — routers have some, `local_stack/` and `finetune_stack/` have none |
| **Error handling** | Inconsistent — some modules catch everything, others let exceptions propagate |
| **Test coverage** | Low — only 2 of 6 tests work; scripts, processing, RAG modules have zero tests |
| **Documentation** | `README.md` and `design.md` are thorough; inline docstrings vary |

## Recommendations (Priority Order)

1. **Fix RAG answer generation** — replace BART summarization with a proper QA model or integrate the existing `Generator` class with a smaller model.
2. **Fix double-storage bug** in `vector_service.ingest_text()` for `target="auto"`.
3. **Fix broken tests** — update imports to match actual function signatures.
4. **Deduplicate `finetune_stack`/`local_stack`** — extract shared base classes or use a factory pattern.
5. **Clean up duplicated definitions** in `rag/`, `processing/`, `services/` files.
6. **Add URL validation** and SSRF protection to URL ingestion endpoints.
7. **Implement temp file cleanup** — use `tempfile.NamedTemporaryFile` with auto-delete or a periodic cleanup task.
8. **Cache FAISS index** instead of rebuilding per query.
9. **Wire authentication** into sensitive endpoints.
10. **Standardize on one embedding model** and one PDF library across all code paths.
