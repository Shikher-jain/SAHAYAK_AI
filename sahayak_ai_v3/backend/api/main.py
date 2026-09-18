import logging
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from backend.api.routers import chat, ingest
from backend.core.config import settings

# Import new observability modules
from sahayak_ai_v3.backend.core.logging import setup_structured_logging
from sahayak_ai_v3.backend.core.middleware import CorrelationIdMiddleware
from sahayak_ai_v3.backend.core.exceptions import register_exception_handlers

# Initialize root logger with structured JSON and PII sanitization
setup_structured_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize connections but DO NOT load heavy ML models here.
    # Heavy models (Whisper, OCR, Vision) belong in isolated worker processes.
    logger.info("Initializing Sahayak AI V3 Orchestrator...")
    yield
    # Shutdown
    logger.info("Shutting down Orchestrator...")

app = FastAPI(
    title="Sahayak AI V3 - Orchestrator",
    description="Next-Generation Enterprise Multimodal RAG with CRAG and GraphRAG",
    lifespan=lifespan
)

# Add observability middleware
app.add_middleware(CorrelationIdMiddleware)

# Register custom exception handlers
register_exception_handlers(app)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router, prefix="/chat", tags=["Chat"])
app.include_router(ingest.router, prefix="/ingest", tags=["Ingestion"])

# Render Keep-Alive Router
from sahayak_ai_v3.backend.api.health import router as health_router
app.include_router(health_router)

# v2_advanced-only agents: loaded lazily so the legacy env boots without regressions
# even if the v2 deps (langgraph, stripe) are not installed (Rule 2).
if settings.SAHAYAK_PIPELINE_VERSION == "v2_advanced":
    try:
        from sahayak_ai_v3.backend.api.routers import v2_chat, payments, audio
        from sahayak_ai_v3.backend.webhooks import telegram, whatsapp

        app.include_router(v2_chat.router, tags=["v2 Chat"])
        app.include_router(payments.router, tags=["v2 Payments"])
        app.include_router(audio.router)
        app.include_router(telegram.router)
        app.include_router(whatsapp.router)
        logger.info("v2_advanced routers registered (supervisor + payments + audio + webhooks).")
    except ImportError as e:
        logger.warning(
            "v2_advanced routers NOT loaded: %s. Install sahayak_ai_v3/requirements.txt "
            "deps in the v2 runtime environment.", e
        )

@app.get("/health")
async def health_check():
    """Lightweight health check for the orchestrator."""
    return {"status": "healthy", "version": "v3.0.0"}
