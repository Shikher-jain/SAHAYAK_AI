import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
# load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR / ".env", override=True)


from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from backend.common.logging_config import configure_logging
from backend.common.rate_limit import limiter
from backend.routers import admin, auth, document_features, finetune, ingestion, local_mode, pages, search, stats, summarize
from backend.routers import voice # Import the voice router separately
from backend.routers import learning, quiz, courses, help_bot, counselor, books, stories, commerce, roadmaps, knowledge, progress, sync
from backend.utils.file_utils import cleanup_tmp_dir
from backend.auth_system.database import init_db as init_auth_db
configure_logging()
logger = logging.getLogger(__name__)

APP_TITLE = "Sahayak AI Platform"

app = FastAPI(title=APP_TITLE)

# Rate limiting — keyed by client IP. `limiter` itself lives in
# backend/common/rate_limit.py (imported by routers too) to avoid a circular
# import, since main.py imports the routers below.
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

cleanup_tmp_dir(max_age_seconds=24 * 3600)
init_auth_db()  # Initialize JWT auth database tables

# TASK 24: GZip compression for responses > 1KB.
app.add_middleware(GZipMiddleware, minimum_size=1024)

# CORS — origins come from ALLOWED_ORIGINS env var (comma-separated).
# Defaults to local Streamlit and React dev servers. allow_origins=["*"] combined with
# allow_credentials=True is both a security hole and invalid per the CORS spec
# (browsers reject wildcard-origin + credentials), so it's intentionally not used.
_default_origins = "http://localhost:8501,http://127.0.0.1:8501,http://localhost:5173,http://127.0.0.1:5173,http://localhost:5174,http://127.0.0.1:5174,http://localhost:3000,http://127.0.0.1:3000"
_allowed_origins = os.getenv("ALLOWED_ORIGINS", _default_origins).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _allowed_origins if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# TASK 24: Request timing middleware for performance monitoring.
@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = round(time.perf_counter() - start, 4)
    response.headers["X-Process-Time"] = str(elapsed)
    return response

@app.get("/")
def root():
    return {
        "name": APP_TITLE,
        "services": [   
            "/ingest",
            "/search",
            "/summaries",
            "/document",
            "/voice",
            "/local",
            "/finetune",
            "/admin",
        ],
    }

def _check_auth_db() -> bool:
    try:
        from sqlalchemy import text
        from backend.auth_system.database import engine
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        logger.exception("Auth DB health check failed")
        return False


def _check_vector_backend() -> bool:
    try:
        from backend.services.vector_service import check_vector_db
        # Either backend ("qdrant" or "faiss") being resolvable counts as healthy;
        # faiss is a valid local fallback, so this isn't a hard failure either way.
        check_vector_db()
        return True
    except Exception:
        logger.exception("Vector backend health check failed")
        return False


@app.get("/health")
def health():
    checks = {"auth_database": _check_auth_db(), "vector_backend": _check_vector_backend()}
    ok = all(checks.values())
    return JSONResponse(
        status_code=200 if ok else 503,
        content={"status": "healthy" if ok else "degraded", "checks": checks},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # Log the full exception server-side; never leak internals (paths, DB errors,
    # library details) to the client.
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again later."},
    )

app.include_router(admin.router)
app.include_router(auth.router)  # JWT auth: /auth/*
app.include_router(ingestion.router, prefix="/ingest")
app.include_router(search.router, prefix="/search")
app.include_router(summarize.router, prefix="/summaries")
app.include_router(document_features.router, prefix="/document")
app.include_router(voice.router, prefix="/voice")
app.include_router(local_mode.router)
app.include_router(finetune.router) 
app.include_router(pages.router)  # Static pages: /pages/*
app.include_router(stats.router)  # Dashboard stats: /stats/*
app.include_router(learning.router)  # Learning modes
app.include_router(quiz.router)  # Interactive quiz
app.include_router(courses.router)  # Course catalog
app.include_router(help_bot.router)  # Help center
app.include_router(counselor.router)  # AI counselor
app.include_router(books.router)  # Online books / NCERT
app.include_router(stories.router)  # User stories
app.include_router(commerce.router)  # E-commerce / pricing
app.include_router(roadmaps.router)  # Learning roadmaps
app.include_router(knowledge.router)  # Knowledge graph
app.include_router(progress.router)  # Progress tracking
app.include_router(sync.router)  # Cloud sync