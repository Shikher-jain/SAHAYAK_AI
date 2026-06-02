from pathlib import Path
import time
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from backend.routers import admin, auth, document_features, finetune, ingestion, local_mode, pages, search, stats, summarize
from backend.routers import voice # Import the voice router separately
from backend.routers import learning, quiz, courses, help_bot, counselor, books, stories, commerce, roadmaps, knowledge, progress, sync
from backend.utils.file_utils import cleanup_tmp_dir
from backend.auth_system.database import init_db as init_auth_db

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

APP_TITLE = "Sahayak AI Platform"

app = FastAPI(title=APP_TITLE)

cleanup_tmp_dir(max_age_seconds=24 * 3600)
init_auth_db()  # Initialize JWT auth database tables

# TASK 24: GZip compression for responses > 1KB.
app.add_middleware(GZipMiddleware, minimum_size=1024)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": str(exc)})

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