"""Cloud sync router — export, import, and backup management."""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from backend.auth_system.auth_service import require_user
from backend.auth_system.models import User
from backend.services import sync_service

router = APIRouter(prefix="/sync", tags=["sync"])


class ImportRequest(BaseModel):
    file_path: str
    target: str = "auto"


@router.post("/export")
def export_data(user: User = Depends(require_user)):
    """Export all local data to a JSON backup file."""
    return sync_service.export_data(str(user.id))


@router.post("/import")
def import_data(req: ImportRequest, user: User = Depends(require_user)):
    """Import data from a previously exported JSON file."""
    return sync_service.import_data(req.file_path, target=req.target)


@router.get("/status")
def sync_status():
    """Get current sync status and statistics."""
    return sync_service.get_status()


@router.get("/exports")
def list_exports(user: User = Depends(require_user)) -> List[Dict[str, str]]:
    """List all available export files."""
    return sync_service.list_exports()
