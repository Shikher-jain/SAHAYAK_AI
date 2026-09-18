import asyncio
import logging
import time
from typing import Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# A dedicated thread pool for CPU-bound ingestion tasks (e.g., transcription, OCR, embedding)
# to prevent blocking the main FastAPI event loop.
_ingestion_pool = ThreadPoolExecutor(max_workers=4)

# Simple in-memory job tracker
_jobs: Dict[str, Dict[str, Any]] = {}

def get_job_status(job_id: str) -> Optional[Dict[str, Any]]:
    return _jobs.get(job_id)

async def run_in_thread(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_ingestion_pool, lambda: func(*args, **kwargs))

async def run_ingestion_job(job_id: str, func, *args, **kwargs):
    """
    Runs an ingestion task in a thread pool and updates the job tracker.
    """
    _jobs[job_id] = {"status": "processing", "started_at": time.time()}
    try:
        logger.info(f"Starting ingestion job {job_id}")
        result = await run_in_thread(func, *args, **kwargs)
        _jobs[job_id]["status"] = "completed"
        _jobs[job_id]["result"] = result
        _jobs[job_id]["completed_at"] = time.time()
        logger.info(f"Completed ingestion job {job_id}")
    except Exception as e:
        logger.exception(f"Failed ingestion job {job_id}")
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = str(e)
        _jobs[job_id]["completed_at"] = time.time()
