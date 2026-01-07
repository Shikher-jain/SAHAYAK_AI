from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.routers import admin, finetune, ingestion, local_mode, search, summarize

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

APP_TITLE = "Sahayak AI Platform"

app = FastAPI(title=APP_TITLE)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "name": APP_TITLE,
        "services": [
            "/ingest",
            "/search",
            "/summaries",
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
app.include_router(ingestion.router, prefix="/ingest")
app.include_router(search.router, prefix="/search")
app.include_router(summarize.router, prefix="/summaries")
app.include_router(local_mode.router)
app.include_router(finetune.router)

